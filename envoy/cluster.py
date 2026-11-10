"""
PAM k-medoids clustering with random baselines and N+2 backup selection.

The core clustering workflow:
    1. Load embeddings and build a cosine distance matrix
    2. Run PAM (Partitioning Around Medoids) for one or more k values
    3. Compare against random baselines (uniform-without-replacement sampling)
    4. Compute silhouette, nearest-medoid, and inertia metrics
    5. Select N+2 backup sequences per medoid for wet-lab redundancy
    6. Report MAFFT-aligned pairwise identity between deployment medoids
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances

from envoy.accession import IDStrategy, normalize_accession


logger = logging.getLogger(__name__)


DEFAULT_SEED = 20261026  # matches Phase B.2 rigor; reproducible across runs
DEFAULT_K = 20           # pre-registered deployment k
DEFAULT_N_BASELINES = 3000  # random baseline draws per k
DEFAULT_N_BACKUPS = 2    # N+2 backup scheme per medoid


@dataclass
class PAMResult:
    """One PAM run at a specific k."""
    k: int
    medoid_indices: np.ndarray       # (k,) indices into the embedding array
    medoid_accessions: list[str]     # (k,) accessions corresponding to medoids
    assignments: np.ndarray          # (n,) cluster label per sequence
    cluster_sizes: list[int]         # (k,) size of each cluster
    inertia: float                   # sum of distances to assigned medoid


@dataclass
class ClusterMetrics:
    """Quality metrics for one clustering result."""
    k: int
    silhouette: float
    mean_nearest_medoid_distance: float
    inertia: float
    # Random baseline comparison (populated if baselines run)
    beat_pct_silhouette: Optional[float] = None
    beat_pct_nearest: Optional[float] = None
    beat_pct_inertia: Optional[float] = None
    random_silhouette_mean: Optional[float] = None
    random_silhouette_std: Optional[float] = None
    random_nearest_mean: Optional[float] = None
    random_nearest_std: Optional[float] = None


@dataclass
class DeploymentPanel:
    """The final selected panel with backups."""
    k: int
    medoid_accessions: list[str]
    cluster_assignments: dict[str, int]  # accession → cluster id
    backups_per_cluster: dict[int, list[str]]  # cluster id → list of backup accessions
    metrics: ClusterMetrics
    pairwise_identity: Optional[np.ndarray] = None  # (k, k) MAFFT-aligned identity matrix


# ============================================================================
# Distance matrix
# ============================================================================


def compute_cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine distance matrix.

    Args:
        embeddings: (n, embed_dim) embedding matrix

    Returns:
        (n, n) symmetric distance matrix with zeros on diagonal
    """
    return cosine_distances(embeddings)


# ============================================================================
# PAM k-medoids
# ============================================================================


def _pam_build(distance_matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Kaufman-Rousseeuw BUILD phase: deterministic greedy medoid selection.

    Picks the sequence with minimum total distance to all others as first medoid,
    then iteratively adds the sequence that most reduces total distance from each
    point to its nearest medoid. Fully deterministic — no randomness.

    Args:
        distance_matrix: (n, n) precomputed symmetric distances
        k: number of medoids to select

    Returns:
        (k,) array of medoid indices
    """
    n = distance_matrix.shape[0]
    medoids = np.zeros(k, dtype=np.int64)

    # First medoid: minimize sum of distances to all other points
    totals = distance_matrix.sum(axis=1)
    medoids[0] = int(np.argmin(totals))

    # Track distance from each point to its nearest-so-far medoid
    nearest = distance_matrix[:, medoids[0]].copy()

    # Pick remaining k-1 medoids greedily
    for i in range(1, k):
        # For each candidate, compute the cost reduction if it becomes a new medoid.
        # Cost reduction = sum over points of max(0, nearest[p] - dist(p, candidate))
        gains = np.zeros(n)
        for c in range(n):
            if c in medoids[:i]:
                continue
            delta = nearest - distance_matrix[:, c]
            gains[c] = np.maximum(delta, 0).sum()
        # Pick the candidate with maximum gain; tiebreak by lowest index for determinism
        best = int(np.argmax(gains))
        medoids[i] = best
        # Update nearest
        new_dists = distance_matrix[:, best]
        nearest = np.minimum(nearest, new_dists)

    return medoids


def _pam_swap(distance_matrix: np.ndarray, medoids: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Kaufman-Rousseeuw SWAP phase: local refinement.

    Iteratively considers swapping each medoid with each non-medoid; accepts
    the swap that most reduces total cost (sum of distances to nearest medoid).
    Terminates when no beneficial swap exists.

    Args:
        distance_matrix: (n, n) symmetric distances
        medoids: (k,) initial medoid indices (typically from BUILD)

    Returns:
        (medoid_indices, n_swaps_performed)
    """
    n = distance_matrix.shape[0]
    k = len(medoids)
    medoids = medoids.copy()
    n_swaps = 0

    max_iterations = 100  # safety: real PAM converges in O(k) iterations
    for iteration in range(max_iterations):
        # Current cost: sum of distances from each point to its nearest medoid
        dists_to_medoids = distance_matrix[:, medoids]  # (n, k)
        nearest = dists_to_medoids.min(axis=1)
        current_cost = nearest.sum()

        best_swap_gain = 0.0
        best_m_idx = -1
        best_candidate = -1

        # Second-nearest distances (needed to compute cost after removing a medoid)
        sorted_dists = np.sort(dists_to_medoids, axis=1)
        second_nearest = sorted_dists[:, 1] if k >= 2 else np.full(n, np.inf)

        # Point-to-medoid-index assignment
        nearest_medoid_idx = dists_to_medoids.argmin(axis=1)

        for m_idx in range(k):
            for c in range(n):
                if c in medoids:
                    continue
                # New cost if we replace medoids[m_idx] with c
                # For points currently nearest to a different medoid:
                #   new_dist_to_c < nearest[p]  →  gain = nearest[p] - new_dist_to_c
                #   else no change
                # For points currently nearest to medoids[m_idx]:
                #   new nearest = min(dist(p, c), second_nearest[p])
                dists_to_c = distance_matrix[:, c]
                new_nearest = nearest.copy()
                # Points currently assigned to medoids[m_idx]: must find new nearest
                assigned_to_this = (nearest_medoid_idx == m_idx)
                new_nearest[assigned_to_this] = np.minimum(
                    dists_to_c[assigned_to_this], second_nearest[assigned_to_this]
                )
                # Points assigned elsewhere may improve if c is closer
                other = ~assigned_to_this
                new_nearest[other] = np.minimum(new_nearest[other], dists_to_c[other])
                new_cost = new_nearest.sum()
                gain = current_cost - new_cost
                if gain > best_swap_gain:
                    best_swap_gain = gain
                    best_m_idx = m_idx
                    best_candidate = c

        if best_m_idx == -1:
            break  # no beneficial swap

        medoids[best_m_idx] = best_candidate
        n_swaps += 1

    return medoids, n_swaps


def run_pam(
    distance_matrix: np.ndarray,
    accessions: list[str],
    k: int,
    seed: int = DEFAULT_SEED,  # kept for API consistency; BUILD is deterministic
) -> PAMResult:
    """
    Run PAM k-medoids on a precomputed distance matrix.

    Uses Kaufman-Rousseeuw BUILD + SWAP. BUILD is a deterministic greedy
    initialization; SWAP iteratively refines by trying all medoid/non-medoid
    swaps and accepting the best. Both phases are deterministic, so output
    depends only on the distance matrix (seed unused).

    Args:
        distance_matrix: (n, n) precomputed pairwise distances
        accessions: (n,) list of accessions in the same order as the matrix
        k: number of medoids
        seed: kept for API compatibility; unused by deterministic BUILD+SWAP

    Returns:
        PAMResult with medoid indices, assignments, and inertia
    """
    n = distance_matrix.shape[0]
    if k > n:
        raise ValueError(f"k={k} exceeds population size n={n}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    # BUILD phase
    medoids = _pam_build(distance_matrix, k)

    # SWAP phase
    medoids, n_swaps = _pam_swap(distance_matrix, medoids)
    logger.debug("PAM at k=%d converged after %d swaps", k, n_swaps)

    # Final assignment: each point to its nearest medoid
    dists_to_medoids = distance_matrix[:, medoids]
    assignments = dists_to_medoids.argmin(axis=1)
    nearest = dists_to_medoids.min(axis=1)
    inertia = float(nearest.sum())

    medoid_accs = [accessions[i] for i in medoids]
    cluster_sizes = [int((assignments == c).sum()) for c in range(k)]

    return PAMResult(
        k=k,
        medoid_indices=medoids,
        medoid_accessions=medoid_accs,
        assignments=assignments,
        cluster_sizes=cluster_sizes,
        inertia=inertia,
    )


# ============================================================================
# Metrics
# ============================================================================


def _compute_core_metrics(
    distance_matrix: np.ndarray,
    assignments: np.ndarray,
    medoid_indices: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute silhouette, mean nearest-medoid distance, and inertia.

    Returns:
        (silhouette, mean_nearest_medoid, inertia)
    """
    n = distance_matrix.shape[0]
    k = len(medoid_indices)

    # Silhouette (uses precomputed distance matrix)
    if k >= 2 and len(set(assignments.tolist())) >= 2:
        sil = float(silhouette_score(distance_matrix, assignments, metric="precomputed"))
    else:
        sil = 0.0

    # Mean nearest-medoid distance
    dists_to_medoids = distance_matrix[:, medoid_indices]  # (n, k)
    nearest_dists = dists_to_medoids.min(axis=1)
    mean_nearest = float(nearest_dists.mean())

    # Inertia (sum of distances to assigned medoid)
    inertia = 0.0
    for i in range(n):
        c = assignments[i]
        if c < k:
            inertia += distance_matrix[i, medoid_indices[c]]

    return sil, mean_nearest, float(inertia)


def compute_metrics(distance_matrix: np.ndarray, pam_result: PAMResult) -> ClusterMetrics:
    """Compute silhouette, nearest-medoid, and inertia for a PAM result."""
    sil, nearest, inertia = _compute_core_metrics(
        distance_matrix, pam_result.assignments, pam_result.medoid_indices
    )
    return ClusterMetrics(
        k=pam_result.k,
        silhouette=sil,
        mean_nearest_medoid_distance=nearest,
        inertia=inertia,
    )


# ============================================================================
# Random baselines
# ============================================================================


def compute_random_baseline_metrics(
    distance_matrix: np.ndarray,
    k: int,
    n_baselines: int = DEFAULT_N_BASELINES,
    seed: int = DEFAULT_SEED,
) -> dict[str, np.ndarray]:
    """
    Generate random panels (uniform without replacement) and compute metrics.

    Each baseline samples k indices from the full population without replacement,
    treats them as medoids, assigns remaining points to the nearest sampled index,
    and computes the same metrics as PAM. This is the correct null distribution
    for "what if we had picked any random k sequences as representatives."

    Args:
        distance_matrix: (n, n) precomputed distances
        k: number of medoids to sample
        n_baselines: number of random draws
        seed: random state

    Returns:
        Dict with keys 'silhouette', 'nearest', 'inertia', each mapping to a
        (n_baselines,) array of metric values from random panels.
    """
    rng = np.random.default_rng(seed)
    n = distance_matrix.shape[0]

    sils = np.zeros(n_baselines, dtype=np.float64)
    nearests = np.zeros(n_baselines, dtype=np.float64)
    inertias = np.zeros(n_baselines, dtype=np.float64)

    for b in range(n_baselines):
        sampled = rng.choice(n, size=k, replace=False)
        # Assign all points to nearest sampled index
        dists_to_sampled = distance_matrix[:, sampled]  # (n, k)
        assignments = dists_to_sampled.argmin(axis=1)
        sil, nearest, inertia = _compute_core_metrics(
            distance_matrix, assignments, sampled
        )
        sils[b] = sil
        nearests[b] = nearest
        inertias[b] = inertia

    return {"silhouette": sils, "nearest": nearests, "inertia": inertias}


def beat_percentile(observed: float, baseline: np.ndarray, higher_is_better: bool = True) -> float:
    """
    Return the fraction (in [0, 1]) of baseline panels the observed metric beats.

    higher_is_better=True (silhouette): fraction of baselines strictly below observed.
    higher_is_better=False (nearest, inertia): fraction of baselines strictly above.
    """
    if higher_is_better:
        return float((baseline < observed).sum()) / len(baseline)
    return float((baseline > observed).sum()) / len(baseline)


def attach_baseline_metrics(
    metrics: ClusterMetrics, baselines: dict[str, np.ndarray]
) -> ClusterMetrics:
    """Compute beat-percentiles and baseline stats; attach to metrics in-place."""
    metrics.beat_pct_silhouette = beat_percentile(
        metrics.silhouette, baselines["silhouette"], higher_is_better=True
    )
    metrics.beat_pct_nearest = beat_percentile(
        metrics.mean_nearest_medoid_distance, baselines["nearest"], higher_is_better=False
    )
    metrics.beat_pct_inertia = beat_percentile(
        metrics.inertia, baselines["inertia"], higher_is_better=False
    )
    metrics.random_silhouette_mean = float(baselines["silhouette"].mean())
    metrics.random_silhouette_std = float(baselines["silhouette"].std())
    metrics.random_nearest_mean = float(baselines["nearest"].mean())
    metrics.random_nearest_std = float(baselines["nearest"].std())
    return metrics


# ============================================================================
# k-sweep orchestrator
# ============================================================================


@dataclass
class KSweepResult:
    """Full k-sweep output."""
    k_values: list[int]
    pam_results: dict[int, PAMResult]       # k → PAMResult
    metrics: dict[int, ClusterMetrics]      # k → ClusterMetrics
    baselines: dict[int, dict[str, np.ndarray]]  # k → baseline arrays
    distance_matrix: np.ndarray
    accessions: list[str]
    n_sequences: int


def run_k_sweep(
    embeddings: np.ndarray,
    accessions: list[str],
    k_values: list[int],
    n_baselines: int = DEFAULT_N_BASELINES,
    seed: int = DEFAULT_SEED,
) -> KSweepResult:
    """
    Run PAM at multiple k values with random baselines.

    Args:
        embeddings: (n, embed_dim) matrix
        accessions: (n,) list of accessions
        k_values: list of k to sweep (e.g. [5, 10, 15, 20, 25, 30])
        n_baselines: random baselines per k
        seed: random state

    Returns:
        KSweepResult with PAM and metrics for each k.
    """
    n = len(accessions)
    if embeddings.shape[0] != n:
        raise ValueError(
            f"Embeddings rows ({embeddings.shape[0]}) does not match "
            f"accession list length ({n})"
        )

    logger.info("Computing distance matrix (%d x %d)...", n, n)
    distance_matrix = compute_cosine_distance_matrix(embeddings)

    pam_results: dict[int, PAMResult] = {}
    all_metrics: dict[int, ClusterMetrics] = {}
    all_baselines: dict[int, dict[str, np.ndarray]] = {}

    for k in k_values:
        if k > n:
            logger.warning("k=%d exceeds n=%d; skipping.", k, n)
            continue
        if k > n // 4:
            logger.warning("k=%d > n/4=%d (overclustered regime).", k, n // 4)

        logger.info("Running PAM at k=%d...", k)
        pam = run_pam(distance_matrix, accessions, k, seed=seed)
        metrics = compute_metrics(distance_matrix, pam)

        logger.info("Computing %d random baselines at k=%d...", n_baselines, k)
        baselines = compute_random_baseline_metrics(
            distance_matrix, k, n_baselines=n_baselines, seed=seed
        )
        metrics = attach_baseline_metrics(metrics, baselines)

        pam_results[k] = pam
        all_metrics[k] = metrics
        all_baselines[k] = baselines

        logger.info(
            "k=%d: silhouette=%.4f (beats %.1f%% random), nearest=%.4f (beats %.1f%%)",
            k, metrics.silhouette, 100 * metrics.beat_pct_silhouette,
            metrics.mean_nearest_medoid_distance, 100 * metrics.beat_pct_nearest,
        )

    return KSweepResult(
        k_values=list(pam_results.keys()),
        pam_results=pam_results,
        metrics=all_metrics,
        baselines=all_baselines,
        distance_matrix=distance_matrix,
        accessions=accessions,
        n_sequences=n,
    )


# ============================================================================
# N+2 backup selection
# ============================================================================


def compute_backups(
    distance_matrix: np.ndarray,
    pam_result: PAMResult,
    accessions: list[str],
    n_backups: int = DEFAULT_N_BACKUPS,
) -> dict[int, list[str]]:
    """
    For each cluster, find the N+<n_backups> closest sequences to the medoid.

    Args:
        distance_matrix: (n, n) precomputed distances
        pam_result: a PAMResult from run_pam
        accessions: (n,) list of accessions matching the distance matrix rows
        n_backups: number of backups per medoid (default 2 for N+2 scheme)

    Returns:
        Dict mapping cluster id → list of accessions (medoid first, then backups
        in order of increasing distance from the medoid).
    """
    backups: dict[int, list[str]] = {}
    for cluster_id in range(pam_result.k):
        medoid_idx = pam_result.medoid_indices[cluster_id]
        # Find members of this cluster, excluding the medoid itself
        member_mask = (pam_result.assignments == cluster_id) & (
            np.arange(len(accessions)) != medoid_idx
        )
        member_indices = np.where(member_mask)[0]
        if len(member_indices) == 0:
            backups[cluster_id] = []
            continue
        # Sort by distance to the medoid, take top n_backups
        dists = distance_matrix[medoid_idx, member_indices]
        sorted_order = np.argsort(dists)
        n_to_take = min(n_backups, len(sorted_order))
        selected = member_indices[sorted_order[:n_to_take]]
        backups[cluster_id] = [accessions[i] for i in selected]
    return backups


# ============================================================================
# MAFFT-aligned pairwise identity
# ============================================================================


def compute_pairwise_identity(
    sequences: list[str],
    labels: Optional[list[str]] = None,
    mafft_path: str = "mafft",
) -> np.ndarray:
    """
    Compute pairwise amino-acid identity (%) on MAFFT-aligned sequences.

    Alignment is required because raw position-by-position comparison of unaligned
    HIV Env sequences produces artificially low identity due to length-variable
    loops (V1/V2/V4). Standard practice for within-clade panel identity reporting
    (DeCamp 2014, Hraber 2017).

    Args:
        sequences: list of protein sequences, one per medoid
        labels: optional names for FASTA headers (defaults to seq_0, seq_1, ...)
        mafft_path: path to the mafft binary (default assumes on PATH)

    Returns:
        (k, k) symmetric identity matrix in percent, with 100 on the diagonal.
    """
    k = len(sequences)
    if k < 2:
        return np.array([[100.0]]) if k == 1 else np.zeros((0, 0))

    if shutil.which(mafft_path) is None:
        raise RuntimeError(
            f"mafft not found at '{mafft_path}'. Install with: brew install mafft "
            f"(macOS) or conda install -c bioconda mafft (Linux)."
        )

    if labels is None:
        labels = [f"seq_{i}" for i in range(k)]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", delete=False
    ) as tmp_in:
        for label, seq in zip(labels, sequences):
            tmp_in.write(f">{label}\n{seq}\n")
        tmp_in_path = tmp_in.name

    try:
        result = subprocess.run(
            [mafft_path, "--auto", "--quiet", tmp_in_path],
            capture_output=True, text=True, check=True,
        )
        aligned_fasta = result.stdout
    finally:
        Path(tmp_in_path).unlink(missing_ok=True)

    # Parse aligned FASTA
    aligned_seqs = []
    current: list[str] = []
    for line in aligned_fasta.split("\n"):
        if line.startswith(">"):
            if current:
                aligned_seqs.append("".join(current).upper())
            current = []
        elif line.strip():
            current.append(line.strip())
    if current:
        aligned_seqs.append("".join(current).upper())

    if len(aligned_seqs) != k:
        raise RuntimeError(
            f"MAFFT returned {len(aligned_seqs)} sequences, expected {k}"
        )

    # Identity = matching positions / positions where both sequences have non-gap
    identity = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                identity[i, j] = 100.0
                continue
            s1, s2 = aligned_seqs[i], aligned_seqs[j]
            matches = 0
            counted = 0
            for a, b in zip(s1, s2):
                if a == "-" or b == "-":
                    continue  # skip gap positions
                counted += 1
                if a == b:
                    matches += 1
            identity[i, j] = 100.0 * matches / counted if counted > 0 else 0.0

    return identity


# ============================================================================
# I/O helpers for the CLI layer
# ============================================================================


def load_embeddings(
    npy_path: str | Path,
    accession_order_path: Optional[str | Path] = None,
    id_strategy: IDStrategy = IDStrategy.AUTO,
) -> tuple[np.ndarray, list[str]]:
    """
    Load an embedding .npy and matching accession-order file.

    If accession_order_path is None, it's inferred from npy_path by replacing
    the .npy extension with .accession_order.txt.

    Returns:
        (embeddings, accessions) — matrix and matching accession list in order.
    """
    npy_path = Path(npy_path)
    embeddings = np.load(npy_path)

    if accession_order_path is None:
        accession_order_path = npy_path.with_suffix(".accession_order.txt")
    accession_order_path = Path(accession_order_path)

    if not accession_order_path.exists():
        raise FileNotFoundError(
            f"Accession-order file not found: {accession_order_path}. "
            f"Expected alongside {npy_path} as <npy_stem>.accession_order.txt"
        )

    with open(accession_order_path) as fh:
        accessions = [normalize_accession(line.strip(), id_strategy) for line in fh if line.strip()]

    if len(accessions) != embeddings.shape[0]:
        raise ValueError(
            f"Embedding rows ({embeddings.shape[0]}) does not match "
            f"accession count ({len(accessions)})"
        )
    return embeddings, accessions


def load_protein_fasta(
    fasta_path: str | Path,
    id_strategy: IDStrategy = IDStrategy.AUTO,
) -> dict[str, str]:
    """Load a protein FASTA into an accession → sequence dict."""
    from Bio import SeqIO

    seqs: dict[str, str] = {}
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        acc = normalize_accession(rec.description, id_strategy)
        seqs[acc] = str(rec.seq).replace("*", "").upper()
    return seqs


def write_sweep_metrics_csv(sweep: KSweepResult, output_path: str | Path) -> None:
    """Write per-k metrics to CSV."""
    rows = []
    for k in sweep.k_values:
        m = sweep.metrics[k]
        row = {
            "k": m.k,
            "silhouette": m.silhouette,
            "mean_nearest_medoid_distance": m.mean_nearest_medoid_distance,
            "inertia": m.inertia,
            "beat_pct_silhouette": m.beat_pct_silhouette,
            "beat_pct_nearest": m.beat_pct_nearest,
            "beat_pct_inertia": m.beat_pct_inertia,
            "random_silhouette_mean": m.random_silhouette_mean,
            "random_silhouette_std": m.random_silhouette_std,
            "random_nearest_mean": m.random_nearest_mean,
            "random_nearest_std": m.random_nearest_std,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def write_pam_result(pam: PAMResult, output_dir: str | Path) -> None:
    """Write medoids and assignments for one PAM run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    k = pam.k
    medoids_df = pd.DataFrame({
        "cluster_id": list(range(k)),
        "medoid_accession": pam.medoid_accessions,
        "cluster_size": pam.cluster_sizes,
    })
    medoids_df.to_csv(output_dir / f"k{k:02d}_medoids.csv", index=False)

    # Assignments
    pd.DataFrame({
        "accession_index": list(range(len(pam.assignments))),
        "cluster_id": pam.assignments.tolist(),
    }).to_csv(output_dir / f"k{k:02d}_assignments.csv", index=False)


def write_deployment_panel(
    panel: DeploymentPanel,
    accessions: list[str],
    protein_seqs: Optional[dict[str, str]],
    output_dir: str | Path,
) -> None:
    """Write deployment medoids, backups, and derived outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Medoids CSV
    medoid_rows = []
    for cluster_id, medoid_acc in enumerate(panel.medoid_accessions):
        medoid_rows.append({
            "cluster_id": cluster_id,
            "medoid_accession": medoid_acc,
            "cluster_size": sum(1 for v in panel.cluster_assignments.values() if v == cluster_id),
        })
    pd.DataFrame(medoid_rows).to_csv(output_dir / "medoids.csv", index=False)

    # Backups CSV
    backup_rows = []
    for cluster_id, backups in panel.backups_per_cluster.items():
        for rank, acc in enumerate(backups):
            backup_rows.append({
                "cluster_id": cluster_id,
                "backup_rank": rank + 1,
                "backup_accession": acc,
            })
    pd.DataFrame(backup_rows).to_csv(output_dir / "backups.csv", index=False)

    # Assignments CSV
    pd.DataFrame([
        {"accession": acc, "cluster_id": cid}
        for acc, cid in panel.cluster_assignments.items()
    ]).to_csv(output_dir / "assignments.csv", index=False)

    # Medoids FASTA (if protein sequences supplied)
    if protein_seqs is not None:
        with open(output_dir / "medoids.fasta", "w") as fh:
            for cluster_id, medoid_acc in enumerate(panel.medoid_accessions):
                if medoid_acc in protein_seqs:
                    fh.write(f">{medoid_acc} cluster={cluster_id}\n{protein_seqs[medoid_acc]}\n")

    # Metrics JSON
    metrics_dict = asdict(panel.metrics)
    with open(output_dir / "metrics.json", "w") as fh:
        json.dump(metrics_dict, fh, indent=2)

    # Pairwise identity (if computed)
    if panel.pairwise_identity is not None:
        np.save(output_dir / "pairwise_identity.npy", panel.pairwise_identity)


# ============================================================================
# Full-run orchestrator
# ============================================================================


def run(
    embeddings_path: str | Path,
    output_dir: str | Path,
    k: int = DEFAULT_K,
    k_sweep_values: Optional[list[int]] = None,
    n_baselines: int = DEFAULT_N_BASELINES,
    n_backups: int = DEFAULT_N_BACKUPS,
    seed: int = DEFAULT_SEED,
    protein_fasta: Optional[str | Path] = None,
    compute_identity: bool = True,
    id_strategy: IDStrategy = IDStrategy.AUTO,
) -> DeploymentPanel:
    """
    Full clustering workflow: sweep, deployment selection, backups, identity.

    Args:
        embeddings_path: path to <clade>.esm2_650m.npy (accession-order file auto-detected)
        output_dir: directory for outputs (sweep/, deployment/, figures/ subdirs created)
        k: deployment k (default 20)
        k_sweep_values: list of k to sweep for figure generation (default [5, 10, 15, 20, 25, 30])
        n_baselines: random baselines per k
        n_backups: backups per deployment medoid
        seed: random state
        protein_fasta: if provided, enables pairwise identity + medoid FASTA output
        compute_identity: whether to compute MAFFT-aligned medoid identity
        id_strategy: accession normalization strategy

    Returns:
        DeploymentPanel at the chosen k.
    """
    output_dir = Path(output_dir)
    sweep_dir = output_dir / "sweep"
    deploy_dir = output_dir / "deployment"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    deploy_dir.mkdir(parents=True, exist_ok=True)

    if k_sweep_values is None:
        k_sweep_values = [5, 10, 15, 20, 25, 30]
    if k not in k_sweep_values:
        k_sweep_values = sorted(set(k_sweep_values + [k]))

    embeddings, accessions = load_embeddings(embeddings_path, id_strategy=id_strategy)
    n = len(accessions)
    logger.info("Loaded %d sequences, embedding dim %d", n, embeddings.shape[1])

    # k-sweep
    sweep = run_k_sweep(
        embeddings=embeddings,
        accessions=accessions,
        k_values=k_sweep_values,
        n_baselines=n_baselines,
        seed=seed,
    )
    write_sweep_metrics_csv(sweep, sweep_dir / "metrics.csv")
    for k_val, pam in sweep.pam_results.items():
        write_pam_result(pam, sweep_dir / "pam_results")

    # Deployment at the chosen k
    if k not in sweep.pam_results:
        raise ValueError(f"Deployment k={k} not in sweep results {list(sweep.pam_results.keys())}")

    pam = sweep.pam_results[k]
    metrics = sweep.metrics[k]

    backups = compute_backups(sweep.distance_matrix, pam, accessions, n_backups=n_backups)

    cluster_assignments = {accessions[i]: int(pam.assignments[i]) for i in range(n)}

    # Pairwise identity on deployment medoids (if requested + protein FASTA supplied)
    pairwise_id: Optional[np.ndarray] = None
    protein_seqs: Optional[dict[str, str]] = None
    if compute_identity and protein_fasta is not None:
        logger.info("Loading protein FASTA for medoid identity...")
        protein_seqs = load_protein_fasta(protein_fasta, id_strategy=id_strategy)
        medoid_prots = []
        missing = []
        for acc in pam.medoid_accessions:
            if acc in protein_seqs:
                medoid_prots.append(protein_seqs[acc])
            else:
                missing.append(acc)
        if missing:
            logger.warning(
                "Protein sequence missing for %d medoids: %s. Skipping identity.",
                len(missing), missing[:3],
            )
        else:
            logger.info("Computing MAFFT-aligned pairwise identity for %d medoids...", k)
            pairwise_id = compute_pairwise_identity(medoid_prots, labels=pam.medoid_accessions)
            logger.info(
                "Median medoid identity: %.1f%%, max: %.1f%%",
                float(np.median(pairwise_id[np.triu_indices(k, k=1)])),
                float(pairwise_id[np.triu_indices(k, k=1)].max()),
            )

    panel = DeploymentPanel(
        k=k,
        medoid_accessions=pam.medoid_accessions,
        cluster_assignments=cluster_assignments,
        backups_per_cluster=backups,
        metrics=metrics,
        pairwise_identity=pairwise_id,
    )

    write_deployment_panel(panel, accessions, protein_seqs, deploy_dir)

    logger.info("Deployment panel written to %s", deploy_dir)
    return panel