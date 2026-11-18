"""
End-to-end smoke test on CRF08_BC data (reuses existing embeddings from ~/genseq/).

Pipeline tested:
    1. Load embeddings via envoy.cluster.load_embeddings
    2. Run PAM k-sweep (k=[5, 10, 15, 20, 25, 30])
    3. Deploy at k=10 (CRF08_BC's pre-registered threshold)
    4. Compute N+2 backups
    5. Compute MAFFT-aligned medoid identity
    6. Write all deployment outputs
    7. Render 4 per-dataset figures

Expected runtime: 1-2 minutes.
Expected outputs: 4 figures + full deployment panel in ./smoke_test_output/CRF08_BC/
"""

from __future__ import annotations

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


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Paths — adjust if your genseq directory is elsewhere
GENSEQ_ROOT = Path.home() / "genseq"
CRF08_EMB = GENSEQ_ROOT / "data/processed/embeddings/2026-11-05/CRF08_BC.esm2_650m.npy"
CRF08_PROT = GENSEQ_ROOT / "data/processed/dedup/2026-11-05/CRF08_BC/CRF08_BC.cdhit98.protein.fasta"

OUTPUT_ROOT = Path(__file__).parent.parent / "smoke_test_output"
CRF08_OUT = OUTPUT_ROOT / "CRF08_BC"


def main() -> int:
    logger.info("=" * 70)
    logger.info("ENVOY smoke test: CRF08_BC")
    logger.info("=" * 70)

    # Sanity-check inputs
    if not CRF08_EMB.exists():
        logger.error("Embeddings not found: %s", CRF08_EMB)
        logger.error("Adjust GENSEQ_ROOT in this script to point to your genseq directory.")
        return 1
    if not CRF08_PROT.exists():
        logger.error("Protein FASTA not found: %s", CRF08_PROT)
        return 1

    logger.info("Inputs:")
    logger.info("  embeddings: %s", CRF08_EMB)
    logger.info("  protein fasta: %s", CRF08_PROT)
    logger.info("  output dir: %s", CRF08_OUT)

    # Full clustering run — tests cluster module end-to-end
    logger.info("\n--- Running envoy.cluster.run ---")
    panel = run_cluster(
        embeddings_path=CRF08_EMB,
        output_dir=CRF08_OUT,
        k=10,  # CRF08_BC's pre-registered deployment k
        k_sweep_values=[5, 10, 15, 20, 25, 30],
        n_baselines=1000,  # reduced for smoke test speed (full run uses 3000)
        n_backups=2,
        seed=DEFAULT_SEED,
        protein_fasta=CRF08_PROT,
        compute_identity=True,
    )

    logger.info("\n--- Deployment panel summary ---")
    logger.info("k = %d", panel.k)
    logger.info("Medoids (%d):", len(panel.medoid_accessions))
    for acc in sorted(panel.medoid_accessions):
        logger.info("  %s", acc)
    logger.info("Metrics at deployment k=%d:", panel.k)
    logger.info("  silhouette = %.4f", panel.metrics.silhouette)
    logger.info(
        "  mean nearest-medoid distance = %.6f", panel.metrics.mean_nearest_medoid_distance
    )
    logger.info("  inertia = %.6f", panel.metrics.inertia)
    if panel.metrics.beat_pct_silhouette is not None:
        logger.info(
            "  PAM beats %.1f%% of random on silhouette",
            100 * panel.metrics.beat_pct_silhouette,
        )
        logger.info(
            "  PAM beats %.1f%% of random on nearest-medoid distance",
            100 * panel.metrics.beat_pct_nearest,
        )

    if panel.pairwise_identity is not None:
        upper = panel.pairwise_identity[np.triu_indices(panel.k, k=1)]
        logger.info("Pairwise medoid identity:")
        logger.info("  median = %.1f%%", np.median(upper))
        logger.info("  max    = %.1f%%", upper.max())
        logger.info("  min    = %.1f%%", upper.min())

    logger.info("Backups per cluster:")
    for cluster_id in sorted(panel.backups_per_cluster.keys()):
        backups = panel.backups_per_cluster[cluster_id]
        medoid = panel.medoid_accessions[cluster_id]
        logger.info(
            "  cluster %d (medoid %s): %s", cluster_id, medoid, backups if backups else "(none)"
        )

    # Render per-dataset figures — tests figures module
    logger.info("\n--- Rendering figures ---")
    fig_dir = CRF08_OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load sweep metrics CSV back in to feed plot_sweep_metrics
    sweep_df = pd.read_csv(CRF08_OUT / "sweep" / "metrics.csv")
    # Add n_sequences column (the plot_sweep_metrics function reads it)
    embeddings, accessions = load_embeddings(CRF08_EMB)
    sweep_df["n_sequences"] = len(accessions)

    plot_sweep_metrics(
        sweep_df=sweep_df,
        deployment_k=10,
        n_sequences=len(accessions),
        output_path=fig_dir / "sweep_metrics.png",
        dataset_name="CRF08_BC",
    )

    # For PCA: reload PAM result at k=10 to get medoid indices + assignments
    distance_matrix = compute_cosine_distance_matrix(embeddings)
    assignments_df = pd.read_csv(CRF08_OUT / "sweep" / "pam_results" / "k10_assignments.csv")
    medoids_df = pd.read_csv(CRF08_OUT / "sweep" / "pam_results" / "k10_medoids.csv")
    medoid_indices = np.array([accessions.index(a) for a in medoids_df["medoid_accession"]])
    cluster_assignments = assignments_df["cluster_id"].to_numpy()

    # Load protein lengths for the length-artifact check
    protein_seqs = load_protein_fasta(CRF08_PROT)
    protein_lengths = np.array(
        [len(protein_seqs.get(acc, "")) for acc in accessions], dtype=np.int64
    )

    plot_pca_2x2(
        embeddings=embeddings,
        accessions=accessions,
        medoid_indices=medoid_indices,
        cluster_assignments=cluster_assignments,
        protein_lengths=protein_lengths,
        output_path=fig_dir / "pca_2x2.png",
        dataset_name="CRF08_BC",
    )

    plot_pairwise_identity(
        identity_matrix=panel.pairwise_identity,
        labels=panel.medoid_accessions,
        output_path=fig_dir / "pairwise_identity.png",
        dataset_name="CRF08_BC",
    )

    plot_dendrogram(
        distance_matrix=distance_matrix,
        accessions=accessions,
        medoid_indices=medoid_indices,
        cluster_assignments=cluster_assignments,
        output_path=fig_dir / "dendrogram.png",
        dataset_name="CRF08_BC",
    )

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Smoke test PASSED")
    logger.info("=" * 70)
    logger.info("Outputs in: %s", CRF08_OUT)
    logger.info("  sweep/metrics.csv")
    logger.info("  sweep/pam_results/k*_medoids.csv + k*_assignments.csv")
    logger.info("  deployment/medoids.csv + backups.csv + assignments.csv + metrics.json")
    logger.info("  deployment/medoids.fasta")
    logger.info("  deployment/pairwise_identity.npy")
    logger.info("  figures/sweep_metrics.png")
    logger.info("  figures/pca_2x2.png")
    logger.info("  figures/pairwise_identity.png")
    logger.info("  figures/dendrogram.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())