"""
Rose & Korber 2000 APOBEC hypermutation detection.

Context-aware G→A substitution test using a Fisher's exact test comparing
G→A rates in APOBEC-preferred contexts (GpR: GA, GG) vs control (GpY: GC, GT).
Hypermutants are called at p < 0.05.

Reference: Rose PP, Korber BT. (2000) Detecting hypermutations in viral sequences
with an emphasis on G→A hypermutation. Bioinformatics 16: 400-401.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from Bio import SeqIO
from Bio.Align import PairwiseAligner
from scipy.stats import fisher_exact


# Clade → consensus reference accession mapping (within LANL consensus env file).
# Fallbacks are used for clades without their own consensus.
CLADE_TO_REFERENCE = {
    "A1": "CONSENSUS_A1",
    "A2": "CONSENSUS_A2",
    "A6": "CONSENSUS_A1",  # fallback: A6 has no dedicated consensus
    "B": "CONSENSUS_B",
    "C": "CONSENSUS_C",
    "D": "CONSENSUS_D",
    "F1": "CONSENSUS_F1",
    "F2": "CONSENSUS_F2",
    "G": "CONSENSUS_G",
    "CRF01_AE": "CONSENSUS_01_AE",
    "CRF02_AG": "CONSENSUS_02_AG",
    "CRF07_BC": "CONSENSUS_08_BC",  # fallback: no CRF07 consensus, use related
    "CRF08_BC": "CONSENSUS_08_BC",
}


@dataclass
class HypermutResult:
    """Outcome of the Rose & Korber hypermut test for one sequence."""
    accession: str
    n_ga_substitutions_gpr: int
    n_g_positions_gpr: int
    n_ga_substitutions_gpy: int
    n_g_positions_gpy: int
    fisher_p_value: float
    is_hypermutant: bool  # True if p < 0.05 AND excess is GpR-biased

    def __str__(self) -> str:
        return (
            f"{self.accession}: GpR {self.n_ga_substitutions_gpr}/{self.n_g_positions_gpr}, "
            f"GpY {self.n_ga_substitutions_gpy}/{self.n_g_positions_gpy}, "
            f"p={self.fisher_p_value:.3e}, hypermutant={self.is_hypermutant}"
        )


def load_reference_sequence(consensus_fasta: str | Path, reference_id: str) -> str:
    """
    Load a specific consensus sequence from the LANL consensus FASTA.

    Args:
        consensus_fasta: path to LANL consensus env alignment FASTA
        reference_id: ID of the consensus to load (e.g. 'CONSENSUS_B')

    Returns:
        Reference sequence as a string with alignment gaps removed.
    """
    for rec in SeqIO.parse(str(consensus_fasta), "fasta"):
        if reference_id in rec.id or reference_id in rec.description:
            return str(rec.seq).replace("-", "").upper()
    raise ValueError(f"Reference {reference_id} not found in {consensus_fasta}")


def align_and_scan(query: str, reference: str) -> tuple[int, int, int, int]:
    """
    Align query to reference and count G→A substitutions by GpR/GpY context.

    Returns:
        (n_ga_gpr, n_g_positions_gpr, n_ga_gpy, n_g_positions_gpy)
        where GpR context = G followed by A or G in reference
              GpY context = G followed by C or T in reference
    """
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.match_score = 2
    aligner.mismatch_score = -1

    try:
        alignment = aligner.align(reference, query)[0]
    except Exception:
        # Degenerate alignment — return zeros
        return 0, 0, 0, 0

    ref_aln = str(alignment[0])
    qry_aln = str(alignment[1])

    n_ga_gpr = n_g_gpr = 0
    n_ga_gpy = n_g_gpy = 0

    for i in range(len(ref_aln) - 1):
        r = ref_aln[i]
        q = qry_aln[i]
        # Next reference base (skip gaps)
        next_r = None
        for j in range(i + 1, len(ref_aln)):
            if ref_aln[j] != "-":
                next_r = ref_aln[j]
                break
        if next_r is None:
            continue
        # Only consider positions where reference is G
        if r != "G":
            continue
        is_gpr = next_r in ("A", "G")
        is_gpy = next_r in ("C", "T")
        if is_gpr:
            n_g_gpr += 1
            if q == "A":
                n_ga_gpr += 1
        elif is_gpy:
            n_g_gpy += 1
            if q == "A":
                n_ga_gpy += 1

    return n_ga_gpr, n_g_gpr, n_ga_gpy, n_g_gpy


def test_sequence(accession: str, query: str, reference: str, p_threshold: float = 0.05) -> HypermutResult:
    """
    Run the Rose & Korber Fisher's exact test on one query sequence.

    Returns a HypermutResult with the counts, p-value, and a binary
    is_hypermutant call (True iff p < p_threshold AND GpR excess is in the
    hypermutant direction, i.e. n_ga_gpr * n_g_gpy > n_ga_gpy * n_g_gpr).
    """
    n_ga_gpr, n_g_gpr, n_ga_gpy, n_g_gpy = align_and_scan(query, reference)

    # Build the 2x2 contingency table for Fisher's exact
    # Rows: GpR vs GpY context
    # Cols: G→A vs G stayed G
    table = [
        [n_ga_gpr, n_g_gpr - n_ga_gpr],
        [n_ga_gpy, n_g_gpy - n_ga_gpy],
    ]

    # Validate table — must have all non-negative entries and at least some signal
    if any(cell < 0 for row in table for cell in row):
        return HypermutResult(accession, n_ga_gpr, n_g_gpr, n_ga_gpy, n_g_gpy, 1.0, False)
    if n_g_gpr == 0 or n_g_gpy == 0:
        return HypermutResult(accession, n_ga_gpr, n_g_gpr, n_ga_gpy, n_g_gpy, 1.0, False)

    _, p_value = fisher_exact(table, alternative="greater")

    # Direction check: hypermutation enriches G→A in GpR relative to GpY
    direction_ok = (n_ga_gpr * n_g_gpy) > (n_ga_gpy * n_g_gpr)
    is_hyper = (p_value < p_threshold) and direction_ok

    return HypermutResult(accession, n_ga_gpr, n_g_gpr, n_ga_gpy, n_g_gpy, p_value, is_hyper)


def detect_from_fasta(
    query_fasta: str | Path,
    consensus_fasta: str | Path,
    clade: str,
    p_threshold: float = 0.05,
) -> list[HypermutResult]:
    """
    Run hypermut detection on every sequence in query_fasta against the clade reference.

    Args:
        query_fasta: FASTA of nucleotide sequences to test
        consensus_fasta: LANL consensus env alignment FASTA
        clade: clade name (e.g. 'B', 'CRF01_AE') for reference lookup
        p_threshold: Fisher's exact p-value threshold for hypermutant call

    Returns:
        List of HypermutResult, one per input sequence.
    """
    if clade not in CLADE_TO_REFERENCE:
        raise ValueError(
            f"No reference mapping for clade {clade}. "
            f"Known clades: {sorted(CLADE_TO_REFERENCE.keys())}"
        )
    reference_id = CLADE_TO_REFERENCE[clade]
    reference = load_reference_sequence(consensus_fasta, reference_id)

    results = []
    for rec in SeqIO.parse(str(query_fasta), "fasta"):
        result = test_sequence(rec.id, str(rec.seq).upper(), reference, p_threshold)
        results.append(result)
    return results