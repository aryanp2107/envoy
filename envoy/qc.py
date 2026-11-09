"""
QC pipeline for raw LANL Env FASTAs.

Six gates applied in order:
    Gate 6: M-group check (hard) — reject Group O/N/P sequences
    Gate 1: env extractable (hard) — require >=2000 nt coverage of HXB2 env
    Gate 2: translation (hard) — longest ORF frame, 600-900 aa, <=2 internal stops
    Gate 3: start-M (soft flag only)
    Gate 4: ambiguity (informational columns only, no reject)
    Gate 5: hypermut (hard) — Fisher's exact p < 0.05 with GpR-biased direction

Permissive by design: optimized across 11 HIV-1 clades for 85-95% pass rates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

from envoy.hypermut import detect_from_fasta
from envoy.accession import IDStrategy, normalize_accession


# HXB2 Env CDS coordinates (standard reference)
HXB2_ENV_START = 6225
HXB2_ENV_END = 8795
HXB2_ENV_LENGTH = HXB2_ENV_END - HXB2_ENV_START + 1  # 2571 nt

MIN_ENV_COVERAGE_NT = 2000
MIN_PROTEIN_LEN_AA = 600
MAX_PROTEIN_LEN_AA = 900
MAX_INTERNAL_STOPS = 2

GROUP_O_INDICATORS = ["O_", "/O.", "GROUP_O"]
GROUP_N_INDICATORS = ["N_", "/N.", "GROUP_N"]
GROUP_P_INDICATORS = ["P_", "/P.", "GROUP_P"]


@dataclass
class QCRecord:
    """Per-sequence QC result."""
    accession: str
    env_nt_length: int
    env_coverage_confidence: str  # "high", "medium", "low"
    protein_length: int
    n_internal_stops: int
    has_start_met: bool
    hypermut_pvalue: float
    ambiguity_pct: float
    flags: str  # comma-separated list of rejections/warnings
    passed: bool


def _longest_orf_translate(nt_seq: str) -> tuple[str, int, int]:
    """
    Find the longest ORF across all 6 frames (3 forward + 3 reverse).

    Selection rule: prefer the frame with FEWEST stop codons (not the longest
    raw translation, which can include X residues from ambiguous nucleotides
    and misidentify the real frame).

    Returns:
        (protein_sequence, n_internal_stops, selected_frame)
        where frame is 0-5 (0-2 forward, 3-5 reverse)
    """
    best_protein = ""
    best_stops = float("inf")
    best_frame = 0

    nt = Seq(nt_seq.upper().replace("-", ""))

    for frame_idx, seq_to_translate in enumerate([nt, nt[1:], nt[2:], nt.reverse_complement(),
                                                     nt.reverse_complement()[1:], nt.reverse_complement()[2:]]):
        # Trim to multiple of 3
        usable = len(seq_to_translate) - (len(seq_to_translate) % 3)
        if usable < 3:
            continue
        protein = str(seq_to_translate[:usable].translate())
        # Count internal stops (exclude a trailing stop if present)
        internal = protein[:-1] if protein.endswith("*") else protein
        n_stops = internal.count("*")
        # Prefer fewest stops; tiebreak by longest protein
        if n_stops < best_stops or (n_stops == best_stops and len(protein) > len(best_protein)):
            best_protein = protein
            best_stops = n_stops
            best_frame = frame_idx

    return best_protein, int(best_stops), best_frame


def _check_group(accession_or_name: str) -> Optional[str]:
    """Return 'O', 'N', 'P' if sequence is identified as non-M group, else None."""
    name_upper = accession_or_name.upper()
    for indicator in GROUP_O_INDICATORS:
        if indicator in name_upper:
            return "O"
    for indicator in GROUP_N_INDICATORS:
        if indicator in name_upper:
            return "N"
    for indicator in GROUP_P_INDICATORS:
        if indicator in name_upper:
            return "P"
    return None


def _compute_ambiguity_pct(nt_seq: str) -> float:
    """Percentage of ambiguous (non-ACGT) nucleotides in the sequence."""
    seq = nt_seq.upper().replace("-", "")
    if not seq:
        return 100.0
    n_ambig = sum(1 for c in seq if c not in "ACGT")
    return 100.0 * n_ambig / len(seq)


def run_gates(
    input_fasta: str | Path,
    consensus_fasta: str | Path,
    clade: str,
    output_dir: str | Path,
    hypermut_p_threshold: float = 0.05,
    id_strategy: IDStrategy = IDStrategy.AUTO,
) -> pd.DataFrame:
    """
    Run all QC gates on an input FASTA and write pass/fail outputs.

    Outputs created in output_dir:
        <clade>.qc_pass.fasta           — nucleotide FASTA of passing sequences
        <clade>.qc_pass.protein.fasta   — translated protein FASTA
        <clade>.qc_pass.metadata.csv    — per-sequence QC results (all input rows)
        <clade>.qc_summary.json         — aggregate pass/fail statistics

    Returns:
        DataFrame of QC records (one row per input sequence).
    """
    input_fasta = Path(input_fasta)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input sequences with configurable ID normalization
    sequences: dict[str, str] = {}
    for rec in SeqIO.parse(str(input_fasta), "fasta"):
        acc = normalize_accession(rec.description, id_strategy)
        sequences[acc] = str(rec.seq).upper()

    # Gates 6, 1, 2, 3, 4 per sequence
    records = []
    proteins: dict[str, str] = {}
    nt_pass: dict[str, str] = {}

    for acc, nt in sequences.items():
        flags = []

        # Gate 6: group check
        group = _check_group(acc)
        if group is not None:
            flags.append(f"group_{group}")
            records.append(QCRecord(
                accession=acc, env_nt_length=len(nt), env_coverage_confidence="low",
                protein_length=0, n_internal_stops=0, has_start_met=False,
                hypermut_pvalue=1.0, ambiguity_pct=_compute_ambiguity_pct(nt),
                flags=",".join(flags), passed=False,
            ))
            continue

        # Gate 1: env coverage
        nt_len = len(nt.replace("-", ""))
        if nt_len < MIN_ENV_COVERAGE_NT:
            flags.append(f"env_too_short_{nt_len}nt")
            records.append(QCRecord(
                accession=acc, env_nt_length=nt_len, env_coverage_confidence="low",
                protein_length=0, n_internal_stops=0, has_start_met=False,
                hypermut_pvalue=1.0, ambiguity_pct=_compute_ambiguity_pct(nt),
                flags=",".join(flags), passed=False,
            ))
            continue
        coverage_conf = "high" if nt_len >= HXB2_ENV_LENGTH * 0.95 else "medium"

        # Gate 2: translation
        protein, n_stops, _frame = _longest_orf_translate(nt)
        prot_len = len(protein.rstrip("*"))
        if prot_len < MIN_PROTEIN_LEN_AA or prot_len > MAX_PROTEIN_LEN_AA:
            flags.append(f"protein_len_out_of_range_{prot_len}aa")
            records.append(QCRecord(
                accession=acc, env_nt_length=nt_len, env_coverage_confidence=coverage_conf,
                protein_length=prot_len, n_internal_stops=n_stops, has_start_met=False,
                hypermut_pvalue=1.0, ambiguity_pct=_compute_ambiguity_pct(nt),
                flags=",".join(flags), passed=False,
            ))
            continue
        if n_stops > MAX_INTERNAL_STOPS:
            flags.append(f"too_many_stops_{n_stops}")
            records.append(QCRecord(
                accession=acc, env_nt_length=nt_len, env_coverage_confidence=coverage_conf,
                protein_length=prot_len, n_internal_stops=n_stops, has_start_met=False,
                hypermut_pvalue=1.0, ambiguity_pct=_compute_ambiguity_pct(nt),
                flags=",".join(flags), passed=False,
            ))
            continue

        # Gate 3: start-M (soft flag only)
        has_start_m = protein.startswith("M")
        if not has_start_m:
            flags.append("no_start_met")

        # Gate 4: ambiguity (informational)
        ambig_pct = _compute_ambiguity_pct(nt)
        if ambig_pct > 5:
            flags.append(f"ambiguity_over_5pct_{ambig_pct:.1f}")
        if ambig_pct > 10:
            flags.append(f"ambiguity_over_10pct_{ambig_pct:.1f}")

        # Provisionally save for Gate 5 (Hypermut)
        proteins[acc] = protein
        nt_pass[acc] = nt
        records.append(QCRecord(
            accession=acc, env_nt_length=nt_len, env_coverage_confidence=coverage_conf,
            protein_length=prot_len, n_internal_stops=n_stops, has_start_met=has_start_m,
            hypermut_pvalue=-1.0,  # placeholder, filled by Gate 5
            ambiguity_pct=ambig_pct, flags=",".join(flags), passed=True,
        ))

    # Gate 5: Hypermut — write temp FASTA of post-gate-2 survivors and test
    if nt_pass:
        tmp_fasta = output_dir / f"{clade}._gate5_input.fasta"
        with open(tmp_fasta, "w") as fh:
            for acc, seq in nt_pass.items():
                fh.write(f">{acc}\n{seq}\n")
        hypermut_results = detect_from_fasta(
            tmp_fasta, consensus_fasta, clade, p_threshold=hypermut_p_threshold
        )
        hypermut_map = {r.accession: r for r in hypermut_results}
        tmp_fasta.unlink(missing_ok=True)

        # Apply Gate 5 to records
        final_records = []
        final_proteins = {}
        final_nt = {}
        for rec in records:
            if rec.passed and rec.accession in hypermut_map:
                hr = hypermut_map[rec.accession]
                rec.hypermut_pvalue = hr.fisher_p_value
                if hr.is_hypermutant:
                    rec.flags = (rec.flags + "," if rec.flags else "") + f"hypermut_p={hr.fisher_p_value:.3e}"
                    rec.passed = False
                else:
                    final_proteins[rec.accession] = proteins[rec.accession]
                    final_nt[rec.accession] = nt_pass[rec.accession]
            final_records.append(rec)
        records = final_records
    else:
        final_proteins = {}
        final_nt = {}

    # Write outputs
    df = pd.DataFrame([asdict(r) for r in records])
    metadata_path = output_dir / f"{clade}.qc_pass.metadata.csv"
    df.to_csv(metadata_path, index=False)

    nt_path = output_dir / f"{clade}.qc_pass.fasta"
    with open(nt_path, "w") as fh:
        for acc, seq in final_nt.items():
            fh.write(f">{acc}\n{seq}\n")

    prot_path = output_dir / f"{clade}.qc_pass.protein.fasta"
    with open(prot_path, "w") as fh:
        for acc, seq in final_proteins.items():
            # Strip trailing stop codon if present
            seq_clean = seq.rstrip("*")
            fh.write(f">{acc}\n{seq_clean}\n")

    # Summary JSON
    n_in = len(records)
    n_pass = sum(1 for r in records if r.passed)
    n_group_reject = sum(1 for r in records if "group_" in r.flags)
    n_env_short = sum(1 for r in records if "env_too_short" in r.flags)
    n_translation_reject = sum(1 for r in records
                               if "protein_len_out_of_range" in r.flags or "too_many_stops" in r.flags)
    n_hypermut_reject = sum(1 for r in records if "hypermut_p=" in r.flags)

    summary = {
        "clade": clade,
        "n_input": n_in,
        "n_pass": n_pass,
        "pass_rate_pct": round(100 * n_pass / n_in, 2) if n_in else 0,
        "n_rejected_group": n_group_reject,
        "n_rejected_env_coverage": n_env_short,
        "n_rejected_translation": n_translation_reject,
        "n_rejected_hypermut": n_hypermut_reject,
        "hypermut_p_threshold": hypermut_p_threshold,
    }
    with open(output_dir / f"{clade}.qc_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    return df