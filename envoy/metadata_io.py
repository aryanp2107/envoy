"""
LANL metadata TSV parser and helpers.

LANL's "Save background info" exports are tab-delimited despite a .txt extension
and typically have leading non-data lines (blank line + "Number of records
retrieved" header) before the real column header. This module handles that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


EXPECTED_MIN_COLUMNS = 4

# Canonical column name candidates used by LANL exports
_ACCESSION_COLS = ["Accession", "accession", "GenBank Accession"]
_COUNTRY_COLS = ["Country", "country", "Sampling Country"]
_YEAR_COLS = ["Sampling Year", "Year", "year", "Collection Year"]
_PATIENT_COLS = ["Patient Code", "Patient", "PAT id(SSAM)"]
_NAME_COLS = ["Name", "name", "Isolate Name"]


def read_lanl_metadata(path: str | Path, max_skiprows: int = 5) -> pd.DataFrame:
    """
    Read a LANL metadata TSV, auto-detecting the header row offset.

    Tries skiprows in [0, 1, ..., max_skiprows] and picks the first that
    yields a parseable TSV with at least EXPECTED_MIN_COLUMNS columns.

    Args:
        path: path to the LANL metadata TSV
        max_skiprows: upper bound on lines to skip before expecting the header

    Returns:
        Parsed DataFrame.

    Raises:
        ValueError: if no skiprows setting produces a valid TSV header.
    """
    path = Path(path)
    for skip in range(0, max_skiprows + 1):
        try:
            df = pd.read_csv(path, sep="\t", skiprows=skip, low_memory=False)
            if len(df.columns) >= EXPECTED_MIN_COLUMNS:
                return df
        except Exception:
            continue
    raise ValueError(
        f"Could not parse {path} as a tab-delimited LANL metadata file. "
        f"Tried skiprows 0 through {max_skiprows}."
    )


def find_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    """Return the first column in `columns` matching (case-insensitive) a candidate."""
    for cand in candidates:
        for c in columns:
            if c.strip().lower() == cand.lower():
                return c
    return None


def resolve_standard_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    """
    Resolve LANL column names to a standard set.

    Returns a dict with keys 'accession', 'country', 'year', 'patient', 'name',
    each mapped to the actual column name in `df` (or None if absent).
    """
    cols = list(df.columns)
    return {
        "accession": find_column(cols, _ACCESSION_COLS),
        "country": find_column(cols, _COUNTRY_COLS),
        "year": find_column(cols, _YEAR_COLS),
        "patient": find_column(cols, _PATIENT_COLS),
        "name": find_column(cols, _NAME_COLS),
    }


def flag_amp_breakthroughs(df: pd.DataFrame, name_col: Optional[str] = None) -> pd.Series:
    """
    Return a boolean Series marking AMP trial breakthrough sequences (V703/V704).

    AMP V703 (HVTN 703) and V704 (HVTN 704) trial breakthrough sequences are
    identifiable by V70[34]_ prefix in LANL isolate names. Returned mask is True
    for breakthrough sequences, False otherwise.
    """
    if name_col is None:
        name_col = find_column(list(df.columns), _NAME_COLS)
    if name_col is None or name_col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[name_col].astype(str).str.contains(r"V70[34]_", regex=True, na=False)


def flag_blank_patient_code(df: pd.DataFrame, patient_col: Optional[str] = None) -> pd.Series:
    """Return a boolean Series marking rows with blank/missing patient codes."""
    if patient_col is None:
        patient_col = find_column(list(df.columns), _PATIENT_COLS)
    if patient_col is None or patient_col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[patient_col].isna() | (df[patient_col].astype(str).str.strip() == "")