"""
Main CLI entry point for envoy.

Dispatches subcommands to their respective modules. Each subcommand is a
thin wrapper around a library function, parsing its arguments from argparse.

Usage:
    envoy qc --input raw.fasta --consensus cons.fasta --clade X --output qc/
    envoy dedup --nt qc/X.qc_pass.fasta --protein qc/X.qc_pass.protein.fasta \\
                --qc-metadata qc/X.qc_pass.metadata.csv --raw-metadata lanl.tsv \\
                --clade X --output dedup/
    envoy embed --input dedup/X.cdhit98.protein.fasta --output embed/X.npy
    envoy cluster --embeddings embed/X.npy --output cluster/ --k 20 \\
                  --protein dedup/X.cdhit98.protein.fasta
    envoy figures --deployment cluster/ --output cluster/figures/ --dataset-name X
"""

from __future__ import annotations

import argparse
import sys

from envoy import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with all subcommands registered."""
    parser = argparse.ArgumentParser(
        prog="envoy",
        description=(
            "ENVOY: distance-network medoid selection for viral protein panels. "
            "Pipeline: qc → dedup → embed → cluster → figures."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"envoy {__version__}",
    )
    subparsers = parser.add_subparsers(
        dest="subcommand", metavar="SUBCOMMAND", required=True,
        help="Pipeline stage to run. See `envoy <subcommand> --help` for details.",
    )

    # Register each subcommand by importing its register() function.
    # Local imports keep CLI startup fast (expensive deps like torch only load
    # when `envoy embed` is actually invoked).
    from envoy.cli import qc as qc_cli
    from envoy.cli import dedup as dedup_cli
    from envoy.cli import embed as embed_cli
    from envoy.cli import cluster as cluster_cli
    from envoy.cli import figures as figures_cli

    qc_cli.register(subparsers)
    dedup_cli.register(subparsers)
    embed_cli.register(subparsers)
    cluster_cli.register(subparsers)
    figures_cli.register(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Main CLI entry point.

    Args:
        argv: command-line arguments (sys.argv[1:] if None). Passed for testability.

    Returns:
        Exit code: 0 on success, nonzero on error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    # Each subcommand's handler is set via set_defaults(func=...)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())