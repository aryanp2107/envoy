"""
`envoy figures` — render the four per-dataset figures from a cluster output.

Reads an existing `envoy cluster` output directory and renders:
    sweep_metrics.png       — 4-panel k-sweep diagnostic
    pca_2x2.png             — PCA 2×2 with cluster coloring and length check
    pairwise_identity.png   — MAFFT-aligned medoid identity heatmap
    dendrogram.png          — UPGMA hierarchical tree, horizontal

Separate from `envoy cluster` so figures can be re-rendered without re-running
PAM (expensive for large clades with many baselines).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from envoy.accession import IDStrategy
from envoy.cli._common import (
    add_common_flags,
    configure_logging,
    handle_cli_error,
    resolve_path,
)


def register(subparsers) -> None:
    """Register `envoy figures` with the top-level dispatcher."""
    parser = subparsers.add_parser(
        "figures",
        help="Render per-dataset figures from a cluster output.",
        description=(
            "Reads an existing `envoy cluster` output directory and renders the "
            "four per-dataset figures: sweep metrics, PCA 2x2, pairwise identity "
            "heatmap, and UPGMA dendrogram. Separate from `envoy cluster` so "
            "figure styling can be iterated without re-running PAM."
        ),
    )
    parser.add_argument(
        "--cluster-dir", required=True, metavar="DIR",
        help="Cluster output directory (from `envoy cluster --output`). Must "
             "contain sweep/metrics.csv and deployment/ subdirectories.",
    )
    parser.add_argument(
        "--embeddings", "-e", required=True, metavar="NPY",
        help="ESM-2 embeddings .npy (same one passed to `envoy cluster`). "
             "Needed for PCA and dendrogram computation.",
    )
    parser.add_argument(
        "--protein", "-p", metavar="FASTA", default=None,
        help="Optional protein FASTA for protein-length annotation in PCA and "
             "pairwise identity. If omitted, the length-artifact panel still "
             "renders but without protein length data.",
    )
    parser.add_argument(
        "--output", "-o", metavar="DIR", default=None,
        help="Output directory for figures (default: <cluster-dir>/figures).",
    )
    parser.add_argument(
        "--k", type=int, default=None, metavar="K",
        help="Deployment k used in the cluster run (default: auto-detect from "
             "metrics.json). Specify explicitly if the deployment k is "
             "ambiguous.",
    )
    parser.add_argument(
        "--dataset-name", metavar="NAME", default=None,
        help="Name to use in figure titles (default: inferred from cluster-dir "
             "parent folder name).",
    )
    parser.add_argument(
        "--only", choices=["sweep", "pca", "identity", "dendrogram"],
        default=None, action="append",
        help="Render only specified figures (can be repeated). "
             "Example: --only sweep --only identity. Default: render all four.",
    )
    parser.add_argument(
        "--id-strategy", choices=["auto", "lanl", "genbank", "verbatim"],
        default="auto",
        help="FASTA header normalization strategy (default: auto-detect).",
    )

    add_common_flags(parser)
    parser.set_defaults(func=run)


def _infer_dataset_name(cluster_dir: Path) -> str:
    """Infer dataset name from the cluster output directory structure."""
    # Common case: <output_root>/<clade_name>/
    # Use the directory name directly.
    return cluster_dir.name


def _infer_deployment_k(cluster_dir: Path) -> int:
    """Auto-detect the deployment k from metrics.json."""
    import json
    metrics_path = cluster_dir / "deployment" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"deployment/metrics.json not found in {cluster_dir}. "
            f"Pass --k explicitly or re-run `envoy cluster` first."
        )
    with open(metrics_path) as fh:
        data = json.load(fh)
    return int(data["k"])


def run(args: argparse.Namespace) -> int:
    """Execute `envoy figures` with parsed arguments."""
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    try:
        cluster_dir = resolve_path(args.cluster_dir, must_exist=True)
        embeddings_path = resolve_path(args.embeddings, must_exist=True)

        protein_fasta = None
        if args.protein is not None:
            protein_fasta = resolve_path(args.protein, must_exist=True)

        output_dir = (
            resolve_path(args.output, must_exist=False) if args.output
            else cluster_dir / "figures"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_name = args.dataset_name or _infer_dataset_name(cluster_dir)
        k = args.k if args.k is not None else _infer_deployment_k(cluster_dir)

        # Determine which figures to render (default: all four)
        selected = set(args.only) if args.only else {
            "sweep", "pca", "identity", "dendrogram"
        }

        id_strategy = IDStrategy(args.id_strategy)

        # Import here so argparse errors surface before matplotlib loads
        import numpy as np
        import pandas as pd

        from envoy.cluster import (
            compute_cosine_distance_matrix,
            load_embeddings,
            load_protein_fasta,
        )
        from envoy.figures import (
            plot_dendrogram,
            plot_pairwise_identity,
            plot_pca_2x2,
            plot_sweep_metrics,
        )

        rendered: list[str] = []

        # --- Sweep metrics ---
        if "sweep" in selected:
            sweep_csv = cluster_dir / "sweep" / "metrics.csv"
            if not sweep_csv.exists():
                print(
                    f"envoy figures: sweep/metrics.csv not found in {cluster_dir}; "
                    f"skipping sweep figure.",
                    file=sys.stderr,
                )
            else:
                sweep_df = pd.read_csv(sweep_csv)
                # Need n_sequences for overcluster shading logic
                embeddings, accessions = load_embeddings(
                    embeddings_path, id_strategy=id_strategy,
                )
                sweep_df["n_sequences"] = len(accessions)
                out = output_dir / "sweep_metrics.png"
                plot_sweep_metrics(
                    sweep_df=sweep_df,
                    deployment_k=k,
                    n_sequences=len(accessions),
                    output_path=out,
                    dataset_name=dataset_name,
                )
                rendered.append("sweep_metrics.png")

        # --- PCA, identity, and dendrogram all need the PAM result at deployment k ---
        need_deployment = selected & {"pca", "identity", "dendrogram"}
        if need_deployment:
            assignments_path = (
                cluster_dir / "sweep" / "pam_results" / f"k{k:02d}_assignments.csv"
            )
            medoids_path = (
                cluster_dir / "sweep" / "pam_results" / f"k{k:02d}_medoids.csv"
            )
            if not assignments_path.exists() or not medoids_path.exists():
                raise FileNotFoundError(
                    f"PAM result files for k={k} not found in {cluster_dir}. "
                    f"Expected {assignments_path.name} and {medoids_path.name}."
                )

            embeddings, accessions = load_embeddings(
                embeddings_path, id_strategy=id_strategy,
            )
            assignments_df = pd.read_csv(assignments_path)
            medoids_df = pd.read_csv(medoids_path)
            medoid_indices = np.array(
                [accessions.index(a) for a in medoids_df["medoid_accession"]]
            )
            cluster_assignments = assignments_df["cluster_id"].to_numpy()

            # --- PCA ---
            if "pca" in selected:
                protein_lengths = None
                if protein_fasta is not None:
                    protein_seqs = load_protein_fasta(
                        protein_fasta, id_strategy=id_strategy,
                    )
                    protein_lengths = np.array(
                        [len(protein_seqs.get(acc, "")) for acc in accessions],
                        dtype=np.int64,
                    )
                out = output_dir / "pca_2x2.png"
                plot_pca_2x2(
                    embeddings=embeddings,
                    accessions=accessions,
                    medoid_indices=medoid_indices,
                    cluster_assignments=cluster_assignments,
                    protein_lengths=protein_lengths,
                    output_path=out,
                    dataset_name=dataset_name,
                )
                rendered.append("pca_2x2.png")

            # --- Pairwise identity ---
            if "identity" in selected:
                identity_path = (
                    cluster_dir / "deployment" / "pairwise_identity.npy"
                )
                if not identity_path.exists():
                    print(
                        f"envoy figures: pairwise_identity.npy not found; "
                        f"skipping identity figure. Re-run `envoy cluster` with "
                        f"--protein to compute it.",
                        file=sys.stderr,
                    )
                else:
                    identity_matrix = np.load(identity_path)
                    out = output_dir / "pairwise_identity.png"
                    plot_pairwise_identity(
                        identity_matrix=identity_matrix,
                        labels=list(medoids_df["medoid_accession"]),
                        output_path=out,
                        dataset_name=dataset_name,
                    )
                    rendered.append("pairwise_identity.png")

            # --- Dendrogram ---
            if "dendrogram" in selected:
                distance_matrix = compute_cosine_distance_matrix(embeddings)
                out = output_dir / "dendrogram.png"
                plot_dendrogram(
                    distance_matrix=distance_matrix,
                    accessions=accessions,
                    medoid_indices=medoid_indices,
                    cluster_assignments=cluster_assignments,
                    output_path=out,
                    dataset_name=dataset_name,
                )
                rendered.append("dendrogram.png")

        if not rendered:
            print(
                "envoy figures: no figures rendered (check --only flags and "
                "input files).",
                file=sys.stderr,
            )
            return 1

        print(
            f"envoy figures: {len(rendered)} figure(s) rendered in {output_dir}: "
            f"{', '.join(rendered)}",
            file=sys.stderr,
        )
        return 0

    except Exception as err:
        return handle_cli_error(err, verbose=args.verbose)