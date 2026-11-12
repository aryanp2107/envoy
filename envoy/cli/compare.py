"""
`envoy compare` — (Optional) compare clustering results across multiple clades.

Reads two or more `envoy cluster` output directories and renders cross-dataset
comparison figures. NOT part of the core single-clade pipeline — use this only
when comparing clustering structure across clades you have already run through
the core chain (qc → dedup → embed → cluster → figures).

Currently renders:
    sweep_comparison.png — small-multiples silhouette vs k, one panel per clade

Additional comparison figures (PCA variance, pairwise identity, cluster size)
are planned for a future release.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from envoy.cli._common import (
    add_common_flags,
    configure_logging,
    handle_cli_error,
    resolve_path,
)


def _parse_dataset_spec(spec: str) -> tuple[str, str]:
    """Parse a NAME:PATH positional argument into (name, path).

    Raises argparse.ArgumentTypeError on malformed input so argparse
    surfaces a clean error message rather than a stack trace.
    """
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"Dataset spec must be NAME:PATH, got {spec!r}. "
            f"Example: CRF08_BC:/path/to/cluster_output"
        )
    name, _, path = spec.partition(":")
    name = name.strip()
    path = path.strip()
    if not name:
        raise argparse.ArgumentTypeError(
            f"Dataset name cannot be empty in {spec!r}."
        )
    if not path:
        raise argparse.ArgumentTypeError(
            f"Dataset path cannot be empty in {spec!r}."
        )
    return name, path


def register(subparsers) -> None:
    """Register `envoy compare` with the top-level dispatcher."""
    parser = subparsers.add_parser(
        "compare",
        help="(Optional) Compare clustering results across multiple clades.",
        description=(
            "Renders cross-dataset comparison figures from multiple `envoy "
            "cluster` outputs. Use this ONLY if you have run the core pipeline "
            "on two or more clades and want to compare their clustering "
            "structure. Not needed for single-clade analyses — the core "
            "pipeline (qc, dedup, embed, cluster, figures) is complete on its "
            "own."
        ),
        epilog=(
            "Example:\n"
            "  envoy compare --output comparison/ \\\n"
            "      CRF08_BC:~/envoy/output/CRF08_BC \\\n"
            "      A6:~/envoy/output/A6 \\\n"
            "      CRF01_AE:~/envoy/output/CRF01_AE\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        type=_parse_dataset_spec,
        metavar="NAME:PATH",
        help="Two or more NAME:PATH pairs. NAME labels the clade in figure "
             "titles; PATH is a cluster output directory from a prior `envoy "
             "cluster` run (must contain sweep/metrics.csv and "
             "deployment/metrics.json).",
    )
    parser.add_argument(
        "--output", "-o", required=True, metavar="DIR",
        help="Output directory for comparison figures. Created if absent.",
    )

    add_common_flags(parser)
    parser.set_defaults(func=run)


def _load_dataset(name: str, cluster_dir: Path):
    """Load the sweep CSV and deployment k for one clade.

    Returns a tuple of (sweep_df, deployment_k) with n_sequences injected
    into the sweep DataFrame if missing.
    """
    import pandas as pd

    sweep_csv = cluster_dir / "sweep" / "metrics.csv"
    metrics_json = cluster_dir / "deployment" / "metrics.json"

    if not sweep_csv.exists():
        raise FileNotFoundError(
            f"[{name}] sweep/metrics.csv not found at {sweep_csv}. "
            f"Re-run `envoy cluster` for this dataset first."
        )
    if not metrics_json.exists():
        raise FileNotFoundError(
            f"[{name}] deployment/metrics.json not found at {metrics_json}. "
            f"Re-run `envoy cluster` for this dataset first."
        )

    sweep_df = pd.read_csv(sweep_csv)

    with open(metrics_json) as fh:
        deployment = json.load(fh)

    if "k" not in deployment:
        raise KeyError(
            f"[{name}] deployment/metrics.json missing 'k' field."
        )
    deployment_k = int(deployment["k"])

    # Inject n_sequences if the sweep CSV doesn't carry it.
    # First try deployment/metrics.json, then fall back to counting rows
    # in deployment/assignments.csv. If neither works, let the figure
    # render without the N= annotation in panel titles.
    if "n_sequences" not in sweep_df.columns:
        n_seq = deployment.get("n_sequences")
        if n_seq is None:
            assignments_csv = cluster_dir / "deployment" / "assignments.csv"
            if assignments_csv.exists():
                n_seq = len(pd.read_csv(assignments_csv))
        if n_seq is not None:
            sweep_df["n_sequences"] = int(n_seq)
        # If still None, the sweep figure just omits "N=..." from the title.

    return sweep_df, deployment_k

def run(args: argparse.Namespace) -> int:
    """Execute `envoy compare` with parsed arguments."""
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    try:
        if len(args.datasets) < 2:
            print(
                "envoy compare: at least 2 datasets required "
                f"(got {len(args.datasets)}). Comparison of a single clade is "
                "not meaningful.",
                file=__import__("sys").stderr,
            )
            return 2

        # Check for duplicate names
        names = [n for n, _ in args.datasets]
        if len(set(names)) < len(names):
            seen = set()
            dupes = sorted({n for n in names if n in seen or seen.add(n)})
            print(
                f"envoy compare: duplicate dataset names: {', '.join(dupes)}. "
                f"Each NAME must be unique.",
                file=__import__("sys").stderr,
            )
            return 2

        output_dir = resolve_path(args.output, must_exist=False)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve and validate each cluster directory
        resolved: list[tuple[str, Path]] = []
        for name, raw_path in args.datasets:
            path = resolve_path(raw_path, must_exist=True)
            resolved.append((name, path))

        # Load sweep CSVs + deployment ks
        datasets: dict = {}
        deployment_ks: dict = {}
        for name, cluster_dir in resolved:
            sweep_df, k = _load_dataset(name, cluster_dir)
            datasets[name] = sweep_df
            deployment_ks[name] = k

        # Import figures here so argparse errors surface before matplotlib loads
        from envoy.figures import plot_sweep_comparison

        output_path = output_dir / "sweep_comparison.png"
        plot_sweep_comparison(
            datasets=datasets,
            deployment_ks=deployment_ks,
            output_path=output_path,
        )

        names_str = ", ".join(sorted(datasets.keys()))
        print(
            f"envoy compare: rendered sweep_comparison.png for "
            f"{len(datasets)} datasets ({names_str}) in {output_dir}",
            file=__import__("sys").stderr,
        )
        return 0

    except Exception as err:
        return handle_cli_error(err, verbose=args.verbose)
