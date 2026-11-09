"""
ENVOY: embedding-based selection of authentic HIV-1 Env representatives.

Public API (available once Delivery 2+ lands):
    envoy.qc          — QC gates + Hypermut
    envoy.dedup       — CD-HIT 98% deduplication
    envoy.embed       — ESM-2 embedding with device auto-detect
    envoy.cluster     — PAM k-medoids + k-sweep + N+2 backups
    envoy.figures     — per-clade and cross-clade figures
    envoy.divergence  — identity-vs-embedding divergence analysis
"""

from envoy.accession import IDStrategy, normalize_accession

__version__ = "0.1.0"

__all__ = ["__version__"]