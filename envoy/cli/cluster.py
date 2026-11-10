"""
`envoy cluster` — PAM k-medoids clustering with random baselines and N+2 backups.

Thin wrapper around envoy.cluster.run(). Runs a k-sweep, selects a deployment
panel, computes N+2 backups per medoid, and (optionally) MAFFT-aligned medoid
identity. Original headers are propagated from the upstream embed sidecar into
all deployment outputs for study traceability.
"""

from __future__ import annotations

import argparse
import sys

from envoy.accession import IDStrategy
from envoy.cli._common import (
    add_common_flags,
    configure_logging,
    handle_cli_error,
    resolve_path,
)


def _parse_k_sweep(value: str) -> list[int]:
    """Parse a comma-separated list of k values from the CLI."""
    try:
        ks = [int(x.strip()) for x in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--k-sweep must be a comma-separated list of integers, got {value!r}"
        )
    if not ks:
        raise argparse.ArgumentTypeError("--k-sweep must contain at least one value")
    if any(k < 1 for k in ks):
        raise argparse.ArgumentTypeError("--k-sweep values must be >= 1")
    return sorted(set(ks))


def register(subparsers) -> None:
    """Register `envoy cluster` with the top-level dispatcher."""
    parser = subparsers.add_parser(
        "cluster",
        help="PAM k-medoids clustering with random baselines and N+2 backups.",
        description=(
            "Load ESM-2 embeddings and run PAM k-medoids clustering at a range "
            "of k values, with random baselines for statistical comparison. "
            "Select a deployment panel at the chosen k, compute N+2 backup "
            "sequences per medoid, and optionally compute MAFFT-aligned medoid "
            "identity. Outputs include sweep metrics, PAM results per k, and a "
            "full deployment panel with medoids, backups, assignments, and "
            "pairwise identity matrix."
        ),
    )

    # --- Input group ---
    inp = parser.add_argument_group("inputs")
    inp.add_argument(
        "--embeddings", "-e", required=True, metavar="NPY",
        help="ESM-2 embeddings .npy from `envoy embed`. The accompanying "
             ".accession_order.txt file is auto-discovered alongside. If "
             ".accessions.tsv is also present, original headers are propagated "
             "into the deployment outputs.",
    )
    inp.add_argument(
        "--protein", "-p", metavar="FASTA", default=None,
        help="Optional protein FASTA (typically <clade>.cdhit98.protein.fasta "
             "from dedup). When provided, enables medoids.fasta writing and "
             "MAFFT-aligned pairwise identity (requires mafft on PATH).",
    )
    inp.add_argument(
        "--output", "-o", required=True, metavar="DIR",
        help="Output directory (created if missing). Produces sweep/ and "
             "deployment/ subdirectories.",
    )

    # --- Clustering parameters ---
    clus = parser.add_argument_group("clustering parameters")
    clus.add_argument(
        "--k", type=int, default=20, metavar="K",
        help="Deployment k — the number of medoids to select for the final "
             "panel (default: 20, pre-registered from Phase B.2 validation).",
    )
    clus.add_argument(
        "--k-sweep", type=_parse_k_sweep, default=[5, 10, 15, 20, 25, 30],
        metavar="K,K,...",
        help="Comma-separated list of k values to sweep for diagnostics "
             "(default: 5,10,15,20,25,30). Deployment k is auto-added if "
             "missing.",
    )
    clus.add_argument(
        "--n-baselines", type=int, default=3000, metavar="N",
        help="Random baseline panels per k for statistical comparison "
             "(default: 3000; Phase B.2 protocol). Set to 1000 or lower for "
             "faster smoke tests; 500 is adequate for ±0.5%% beat-percentile "
             "precision.",
    )
    clus.add_argument(
        "--n-backups", type=int, default=2, metavar="N",
        help="Backup sequences per deployment medoid (default: 2, forming the "
             "N+2 redundancy scheme). Set to 0 to disable.",
    )
    clus.add_argument(
        "--seed", type=int, default=20261026, metavar="S",
        help="Random seed for baseline sampling (default: 20261026, Phase B.2 "
             "pre-registered value for reproducibility).",
    )

    # --- Optional analysis ---
    opt = parser.add_argument_group("optional analysis")
    opt.add_argument(
        "--no-identity", dest="compute_identity", action="store_false",
        default=True,
        help="Skip MAFFT-aligned pairwise identity computation. By default "
             "identity is computed if --protein is provided; this flag disables "
             "it even when --protein is set.",
    )
    opt.add_argument(
        "--id-strategy", choices=["auto", "lanl", "genbank", "verbatim"],
        default="auto",
        help="FASTA header normalization strategy (default: auto-detect).",
    )

    add_common_flags(parser)
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Execute `envoy cluster` with parsed arguments."""
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    try:
        if args.k < 1:
            raise ValueError(f"--k must be >= 1, got {args.k}")
        if args.n_baselines < 0:
            raise ValueError(
                f"--n-baselines must be >= 0, got {args.n_baselines}"
            )
        if args.n_backups < 0:
            raise ValueError(f"--n-backups must be >= 0, got {args.n_backups}")

        embeddings_path = resolve_path(args.embeddings, must_exist=True)
        output_dir = resolve_path(args.output, must_exist=False)

        protein_fasta = None
        if args.protein is not None:
            protein_fasta = resolve_path(args.protein, must_exist=True)
        elif args.compute_identity:
            # Silently fine — identity just won't run without protein
            pass

        id_strategy = IDStrategy(args.id_strategy)

        # Ensure deployment k is in the sweep
        k_sweep = args.k_sweep
        if args.k not in k_sweep:
            k_sweep = sorted(set(k_sweep + [args.k]))

        # Import here so argparse errors surface before expensive module loads
        from envoy.cluster import run as run_cluster

        panel = run_cluster(
            embeddings_path=embeddings_path,
            output_dir=output_dir,
            k=args.k,
            k_sweep_values=k_sweep,
            n_baselines=args.n_baselines,
            n_backups=args.n_backups,
            seed=args.seed,
            protein_fasta=protein_fasta,
            compute_identity=args.compute_identity,
            id_strategy=id_strategy,
        )

        # Summary line
        sil = panel.metrics.silhouette
        nearest = panel.metrics.mean_nearest_medoid_distance
        beat_sil = panel.metrics.beat_pct_silhouette
        beat_near = panel.metrics.beat_pct_nearest

        summary = (
            f"envoy cluster: k={args.k}, silhouette={sil:.4f}"
        )
        if beat_sil is not None:
            summary += f" (beats {100*beat_sil:.1f}% random)"
        summary += f", nearest-medoid={nearest:.6f}"
        if beat_near is not None:
            summary += f" (beats {100*beat_near:.1f}% random)"
        summary += f". Outputs in {output_dir}"
        print(summary, file=sys.stderr)

        return 0

    except Exception as err:
        return handle_cli_error(err, verbose=args.verbose)