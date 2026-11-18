"""
End-to-end smoke test on any clade in ~/genseq/ data.

Usage:
    python tests/smoke_test_clade.py <CLADE_NAME> [--k K] [--n-baselines N]

Example:
    python tests/smoke_test_clade.py A6
    python tests/smoke_test_clade.py CRF01_AE --n-baselines 2000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from envoy.cluster import (
    DEFAULT_SEED,
    compute_cosine_distance_matrix,
    load_embeddings,
    load_protein_fasta,
    run as run_cluster,
)
from envoy.figures import (
    plot_dendrogram,
    plot_pairwise_identity,
    plot_pca_2x2,
    plot_sweep_metrics,
)


# Pre-registered deployment k per clade (from paper's threshold table)
DEPLOYMENT_K = {
    "A1": 20, "A6": 20, "B": 20, "C": 20, "D": 20, "G": 20,
    "CRF01_AE": 20, "CRF02_AG": 20, "CRF07_BC": 20,
    "F1": 15, "CRF08_BC": 10,
}

GENSEQ_ROOT = Path.home() / "genseq"


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end ENVOY smoke test")
    parser.add_argument("clade", help="Clade name (e.g., A6, CRF01_AE)")
    parser.add_argument("--k", type=int, default=None,
                        help="Deployment k (defaults to pre-registered value)")
    parser.add_argument("--n-baselines", type=int, default=1000,
                        help="Random baselines per k (default 1000 for speed)")
    args = parser.parse_args()

    clade = args.clade
    k = args.k if args.k is not None else DEPLOYMENT_K.get(clade, 20)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    emb_path = GENSEQ_ROOT / f"data/processed/embeddings/2026-11-05/{clade}.esm2_650m.npy"
    prot_path = GENSEQ_ROOT / f"data/processed/dedup/2026-11-05/{clade}/{clade}.cdhit98.protein.fasta"

    if not emb_path.exists():
        logger.error("Embeddings not found: %s", emb_path)
        return 1
    if not prot_path.exists():
        logger.error("Protein FASTA not found: %s", prot_path)
        return 1

    output_root = Path(__file__).parent.parent / "smoke_test_output" / clade

    logger.info("=" * 70)
    logger.info("ENVOY smoke test: %s", clade)
    logger.info("=" * 70)
    logger.info("Deployment k: %d", k)
    logger.info("Random baselines per k: %d", args.n_baselines)
    logger.info("Output: %s", output_root)

    # Run pipeline
    panel = run_cluster(
        embeddings_path=emb_path,
        output_dir=output_root,
        k=k,
        k_sweep_values=[5, 10, 15, 20, 25, 30],
        n_baselines=args.n_baselines,
        n_backups=2,
        seed=DEFAULT_SEED,
        protein_fasta=prot_path,
        compute_identity=True,
    )

    # Summarize
    logger.info("\n--- Deployment panel summary ---")
    logger.info("k = %d, n_medoids = %d", panel.k, len(panel.medoid_accessions))
    logger.info("Metrics:")
    logger.info("  silhouette = %.4f", panel.metrics.silhouette)
    logger.info("  mean nearest-medoid distance = %.6f",
                panel.metrics.mean_nearest_medoid_distance)
    logger.info("  inertia = %.6f", panel.metrics.inertia)
    if panel.metrics.beat_pct_silhouette is not None:
        logger.info("  PAM beats %.1f%% of random on silhouette",
                    100 * panel.metrics.beat_pct_silhouette)
        logger.info("  PAM beats %.1f%% of random on nearest-medoid distance",
                    100 * panel.metrics.beat_pct_nearest)

    if panel.pairwise_identity is not None:
        upper = panel.pairwise_identity[np.triu_indices(panel.k, k=1)]
        logger.info("Pairwise medoid identity:  median=%.1f%%, min=%.1f%%, max=%.1f%%",
                    np.median(upper), upper.min(), upper.max())

    # Render figures
    logger.info("\n--- Rendering figures ---")
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sweep_df = pd.read_csv(output_root / "sweep" / "metrics.csv")
    embeddings, accessions = load_embeddings(emb_path)
    sweep_df["n_sequences"] = len(accessions)

    plot_sweep_metrics(
        sweep_df=sweep_df, deployment_k=k, n_sequences=len(accessions),
        output_path=fig_dir / "sweep_metrics.png", dataset_name=clade,
    )

    distance_matrix = compute_cosine_distance_matrix(embeddings)
    assignments_df = pd.read_csv(
        output_root / "sweep" / "pam_results" / f"k{k:02d}_assignments.csv"
    )
    medoids_df = pd.read_csv(
        output_root / "sweep" / "pam_results" / f"k{k:02d}_medoids.csv"
    )
    medoid_indices = np.array(
        [accessions.index(a) for a in medoids_df["medoid_accession"]]
    )
    cluster_assignments = assignments_df["cluster_id"].to_numpy()

    protein_seqs = load_protein_fasta(prot_path)
    protein_lengths = np.array(
        [len(protein_seqs.get(acc, "")) for acc in accessions], dtype=np.int64
    )

    plot_pca_2x2(
        embeddings=embeddings, accessions=accessions,
        medoid_indices=medoid_indices,
        cluster_assignments=cluster_assignments,
        protein_lengths=protein_lengths,
        output_path=fig_dir / "pca_2x2.png", dataset_name=clade,
    )

    plot_pairwise_identity(
        identity_matrix=panel.pairwise_identity,
        labels=panel.medoid_accessions,
        output_path=fig_dir / "pairwise_identity.png", dataset_name=clade,
    )

    plot_dendrogram(
        distance_matrix=distance_matrix,
        accessions=accessions,
        medoid_indices=medoid_indices,
        cluster_assignments=cluster_assignments,
        output_path=fig_dir / "dendrogram.png", dataset_name=clade,
    )

    logger.info("\n" + "=" * 70)
    logger.info("Smoke test PASSED: %s", clade)
    logger.info("=" * 70)
    logger.info("Outputs in: %s", output_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())