"""
Sequence accession normalization.

Handles the diversity of FASTA header conventions in HIV sequence data:
    - LANL: >B.US.1990.WEAU16.U21135  → U21135
    - GenBank: >U21135.1 Human immunodeficiency... → U21135
    - Verbatim: >my_sequence_001 → my_sequence_001
    - Pipe-delimited: >B|US|1990|U21135 → U21135
"""

import re
from enum import Enum

# GenBank accession pattern: 1-2 letters + 5-8 digits, optional .version
GENBANK_PATTERN = re.compile(r"^([A-Z]{1,2}\d{5,8})(?:\.\d+)?$")

# LANL-style: contains dots, probably has country/year/name structure
LANL_PATTERN = re.compile(r"^[A-Z0-9_]+\.[A-Z]{2,3}\.\d{4}\.[^.]+\.[A-Z0-9]+$")


class IDStrategy(str, Enum):
    AUTO = "auto"
    LANL = "lanl"
    GENBANK = "genbank"
    VERBATIM = "verbatim"


def normalize_accession(header: str, strategy: IDStrategy = IDStrategy.AUTO) -> str:
    """
    Normalize a FASTA header into a canonical accession.

    The accession is used as a primary key across QC → dedup → embed → cluster.
    All stages must agree on the normalization, or metadata merges break.

    Args:
        header: raw FASTA header (without the leading '>')
        strategy: how to normalize. AUTO tries LANL → GenBank → verbatim.

    Returns:
        Normalized accession string (always stripped of whitespace and version).
    """
    first_token = header.split()[0] if header else ""

    if strategy == IDStrategy.VERBATIM:
        return first_token

    if strategy == IDStrategy.LANL:
        return first_token.split(".")[-1]

    if strategy == IDStrategy.GENBANK:
        m = GENBANK_PATTERN.match(first_token)
        return m.group(1) if m else first_token

    # AUTO: try each in priority order
    if LANL_PATTERN.match(first_token):
        return first_token.split(".")[-1]
    if "|" in first_token:
        # HIV-TRACE / pipe-delimited: take last field, then recursively normalize
        last_field = first_token.split("|")[-1]
        return normalize_accession(last_field, IDStrategy.AUTO)
    m = GENBANK_PATTERN.match(first_token)
    if m:
        return m.group(1)
    return first_token


def normalize_batch(headers: list[str], strategy: IDStrategy = IDStrategy.AUTO) -> list[str]:
    """
    Normalize a batch of headers. Checks for duplicates after normalization.

    Raises ValueError if normalization collapses two distinct headers to the
    same accession (users should either pick a different strategy or clean
    their input).
    """
    normalized = [normalize_accession(h, strategy) for h in headers]
    seen = {}
    for orig, norm in zip(headers, normalized):
        if norm in seen and seen[norm] != orig:
            raise ValueError(
                f"Accession collision after normalization: "
                f"'{seen[norm]}' and '{orig}' both normalize to '{norm}'. "
                f"Try --id-strategy verbatim to preserve original IDs."
            )
        seen[norm] = orig
    return normalized