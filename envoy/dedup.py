"""
CD-HIT nucleotide deduplication with smart-pick representative selection.

Wraps the `cd-hit-est` binary (must be on PATH). Runs at 98% identity by default,
parses the .clstr output, applies three smart-pick rules in priority order:
    1. Prefer cluster members with populated patient codes
    2. Prefer non-AMP breakthrough sequences
    3. Longest sequence as tiebreaker
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from Bio import SeqIO

from envoy.metadata_io import (
    find_column,
    flag_amp_breakthroughs,
    flag_blank_patient_code,
    resolve_standard_columns,
)


@dataclass
class ClusterCollapse:
    """One CD-HIT cluster summary."""
    cluster_id: int
    n_members: int
    representative: str
    members: list[str]
    smart_pick_reason: str  # which rule triggered the representative choice


def _parse_clstr(clstr_path: Path) -> list[list[tuple[str, int]]]:
    """
    Parse a CD-HIT .clstr file.

    Returns a list of clusters; each cluster is a list of (accession, length) tuples.
    """
    clusters = []
    current: list[tuple[str, int]] = []
    seq_pattern = re.compile(r"\s*\d+\s+(\d+)(?:aa|nt),\s*>(\S+)\.{3}")
    with open(clstr_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">Cluster"):
                if current:
                    clusters.append(current)
                current = []
            else:
                m = seq_pattern.match(line)
                if m:
                    length = int(m.group(1))
                    acc = m.group(2)
                    current.append((acc, length))
    if current:
        clusters.append(current)
    return clusters


def _smart_pick(members: list[tuple[str, int]], metadata: pd.DataFrame,
                acc_col: str, patient_col: Optional[str], name_col: Optional[str]) -> tuple[str, str]:
    """
    Apply smart-pick rules to select the best cluster representative.

    Returns (representative_accession, reason).
    """
    # Build lookup of members to metadata rows
    member_accs = [m[0] for m in members]
    sub = metadata[metadata[acc_col].astype(str).isin(member_accs)].copy()

    if sub.empty:
        # No metadata for any member — default to longest
        longest = max(members, key=lambda x: x[1])
        return longest[0], "fallback_longest_no_metadata"

    # Rule 1: prefer members with populated patient codes
    if patient_col is not None:
        with_patient_mask = ~flag_blank_patient_code(sub, patient_col)
        if with_patient_mask.any():
            sub = sub[with_patient_mask]

    # Rule 2: prefer non-AMP breakthrough sequences
    if name_col is not None:
        amp_mask = flag_amp_breakthroughs(sub, name_col)
        if (~amp_mask).any():
            sub = sub[~amp_mask]

    # Rule 3: tiebreak on length
    length_map = {acc: length for acc, length in members}
    sub = sub.copy()
    sub["_length"] = sub[acc_col].astype(str).map(length_map).fillna(0).astype(int)
    sub = sub.sort_values("_length", ascending=False)

    chosen = sub.iloc[0][acc_col]
    # Build reason string
    reasons = []
    if patient_col is not None:
        n_with_patient = (~flag_blank_patient_code(
            metadata[metadata[acc_col].astype(str).isin(member_accs)], patient_col
        )).sum()
        if n_with_patient < len(member_accs):
            reasons.append(f"patient_code_present")
    if name_col is not None:
        n_amp = flag_amp_breakthroughs(
            metadata[metadata[acc_col].astype(str).isin(member_accs)], name_col
        ).sum()
        if n_amp > 0:
            reasons.append("non_amp_preferred")
    reasons.append("longest_tiebreak")
    return str(chosen), "+".join(reasons)


def run(
    input_nt_fasta: str | Path,
    input_protein_fasta: str | Path,
    metadata_csv: str | Path,
    raw_metadata_tsv: str | Path,
    clade: str,
    output_dir: str | Path,
    identity_threshold: float = 0.98,
    n_threads: int = 4,
) -> pd.DataFrame:
    """
    Run CD-HIT-EST deduplication with smart-pick representative selection.

    Args:
        input_nt_fasta: nucleotide FASTA (from `envoy qc`)
        input_protein_fasta: matched protein FASTA (from `envoy qc`)
        metadata_csv: per-sequence QC metadata CSV (from `envoy qc`)
        raw_metadata_tsv: original LANL metadata TSV (for geography/patient info)
        clade: clade name for output file naming
        output_dir: directory to write deduplicated outputs
        identity_threshold: CD-HIT identity threshold (default 0.98)
        n_threads: CPU threads for CD-HIT

    Outputs:
        <clade>.cdhit98.fasta           — representative nucleotide sequences
        <clade>.cdhit98.protein.fasta   — matched representative proteins
        <clade>.cdhit98.metadata.csv    — filtered metadata for representatives
        <clade>.cdhit98.clstr           — CD-HIT cluster log
        <clade>.dedup_audit.json        — collapse stats and smart-pick decisions

    Returns:
        DataFrame of representative metadata.
    """
    if shutil.which("cd-hit-est") is None:
        raise RuntimeError(
            "cd-hit-est not found on PATH. Install with: brew install cd-hit "
            "(macOS) or conda install -c bioconda cd-hit (Linux/conda)."
        )

    input_nt_fasta = Path(input_nt_fasta)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cdhit_out = output_dir / f"{clade}.cdhit98.fasta"
    cdhit_clstr = output_dir / f"{clade}.cdhit98.fasta.clstr"

    # Word size by identity: see CD-HIT docs
    if identity_threshold >= 0.97:
        word_size = "10"
    elif identity_threshold >= 0.95:
        word_size = "9"
    elif identity_threshold >= 0.90:
        word_size = "8"
    else:
        word_size = "7"

    cmd = [
        "cd-hit-est",
        "-i", str(input_nt_fasta),
        "-o", str(cdhit_out),
        "-c", str(identity_threshold),
        "-n", word_size,
        "-T", str(n_threads),
        "-M", "0",  # unlimited memory
        "-d", "0",  # full header in .clstr
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"cd-hit-est failed:\n{result.stderr}")

    # Rename .clstr to standard location
    produced_clstr = Path(str(cdhit_out) + ".clstr")
    if produced_clstr.exists() and produced_clstr != cdhit_clstr:
        produced_clstr.rename(cdhit_clstr)

    clusters_raw = _parse_clstr(cdhit_clstr)

    # Load metadata for smart-pick
    raw_meta = pd.read_csv(raw_metadata_tsv, sep="\t", skiprows=2, low_memory=False)
    cols = resolve_standard_columns(raw_meta)
    acc_col = cols["accession"]
    patient_col = cols["patient"]
    name_col = cols["name"]

    if acc_col is None:
        raise RuntimeError(
            f"Could not find accession column in {raw_metadata_tsv}. "
            f"Columns found: {list(raw_meta.columns)}"
        )

    collapses: list[ClusterCollapse] = []
    representatives: list[str] = []
    for cluster_id, members in enumerate(clusters_raw):
        if len(members) == 1:
            rep = members[0][0]
            collapses.append(ClusterCollapse(
                cluster_id=cluster_id, n_members=1, representative=rep,
                members=[rep], smart_pick_reason="singleton",
            ))
            representatives.append(rep)
            continue
        rep, reason = _smart_pick(members, raw_meta, acc_col, patient_col, name_col)
        collapses.append(ClusterCollapse(
            cluster_id=cluster_id, n_members=len(members),
            representative=rep, members=[m[0] for m in members],
            smart_pick_reason=reason,
        ))
        representatives.append(rep)

    # Rewrite cdhit_out FASTA to contain the smart-picked representatives
    # (CD-HIT by default picks the longest; our smart-pick may differ)
    all_nt: dict[str, str] = {}
    for rec in SeqIO.parse(str(input_nt_fasta), "fasta"):
        acc = rec.id.split(".")[-1] if "." in rec.id else rec.id
        all_nt[acc] = str(rec.seq).upper()

    with open(cdhit_out, "w") as fh:
        for rep in representatives:
            if rep in all_nt:
                fh.write(f">{rep}\n{all_nt[rep]}\n")

    # Matching protein FASTA
    all_prot: dict[str, str] = {}
    for rec in SeqIO.parse(str(input_protein_fasta), "fasta"):
        acc = rec.id.split(".")[-1] if "." in rec.id else rec.id
        all_prot[acc] = str(rec.seq)

    prot_out = output_dir / f"{clade}.cdhit98.protein.fasta"
    with open(prot_out, "w") as fh:
        for rep in representatives:
            if rep in all_prot:
                fh.write(f">{rep}\n{all_prot[rep]}\n")

    # Filtered metadata
    filtered_meta = raw_meta[raw_meta[acc_col].astype(str).isin(representatives)].copy()
    meta_out = output_dir / f"{clade}.cdhit98.metadata.csv"
    filtered_meta.to_csv(meta_out, index=False)

    # Audit JSON
    n_in = sum(len(c.members) for c in collapses)
    n_out = len(collapses)
    audit = {
        "clade": clade,
        "identity_threshold": identity_threshold,
        "n_input": n_in,
        "n_output": n_out,
        "collapse_rate_pct": round(100 * (1 - n_out / n_in), 2) if n_in else 0,
        "n_singletons": sum(1 for c in collapses if c.n_members == 1),
        "n_multimembers": sum(1 for c in collapses if c.n_members > 1),
        "largest_cluster_size": max(c.n_members for c in collapses) if collapses else 0,
        "smart_pick_reasons": {},  # populated below
    }
    reason_counts: dict[str, int] = {}
    for c in collapses:
        if c.n_members > 1:
            reason_counts[c.smart_pick_reason] = reason_counts.get(c.smart_pick_reason, 0) + 1
    audit["smart_pick_reasons"] = reason_counts

    with open(output_dir / f"{clade}.dedup_audit.json", "w") as fh:
        json.dump(audit, fh, indent=2)

    return filtered_meta