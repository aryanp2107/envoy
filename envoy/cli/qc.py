"""
`envoy qc` — run QC gates on a raw FASTA.

Thin wrapper around envoy.qc.run_gates(). Validates inputs, configures logging,
and delegates. All real logic lives in envoy/qc.py.
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


def register(subparsers) -> None:
    """Register `envoy qc` with the top-level dispatcher."""
    parser = subparsers.add_parser(
        "qc",
        help="Run QC gates on a raw FASTA.",
        description=(
            "Apply six QC gates (group M check, env coverage, translation, "
            "start-M, ambiguity, Hypermut) to a raw nucleotide FASTA. "
            "Outputs passing sequences as both nucleotide and protein FASTAs, "
            "plus a per-sequence metadata CSV with QC results and original headers."
        ),
    )
    parser.add_argument(
        "--input", "-i", required=True, metavar="FASTA",
        help="Input nucleotide FASTA (raw sequences to QC).",
    )
    parser.add_argument(
        "--consensus", "-c", required=True, metavar="FASTA",
        help="Clade consensus FASTA for Hypermut (gate 5). "
             "Single-file compendium (e.g., LANL lanl_consensus_env.fas) works; "
             "the matching clade entry is located by name.",
    )
    parser.add_argument(
        "--clade", required=True, metavar="NAME",
        help="Clade/dataset name (used for output filenames and consensus lookup).",
    )
    parser.add_argument(
        "--output", "-o", required=True, metavar="DIR",
        help="Output directory (created if missing).",
    )
    parser.add_argument(
        "--hypermut-p", type=float, default=0.05, metavar="P",
        help="Fisher's exact p-value threshold for Hypermut rejection "
             "(default: 0.05). Sequences with p < threshold are flagged as "
             "hypermutants and rejected.",
    )
    parser.add_argument(
        "--id-strategy", choices=["auto", "lanl", "genbank", "verbatim"],
        default="auto",
        help="FASTA header normalization strategy (default: auto-detect).",
    )

    add_common_flags(parser)
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Execute `envoy qc` with parsed arguments."""
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    try:
        input_fasta = resolve_path(args.input, must_exist=True)
        consensus_fasta = resolve_path(args.consensus, must_exist=True)
        output_dir = resolve_path(args.output, must_exist=False)

        id_strategy = IDStrategy(args.id_strategy)

        # Import here so argparse errors show before expensive module loads
        from envoy.qc import run_gates

        df = run_gates(
            input_fasta=input_fasta,
            consensus_fasta=consensus_fasta,
            clade=args.clade,
            output_dir=output_dir,
            hypermut_p_threshold=args.hypermut_p,
            id_strategy=id_strategy,
        )

        n_in = len(df)
        n_pass = int(df["passed"].sum())
        pass_rate = 100 * n_pass / n_in if n_in else 0
        print(
            f"envoy qc: {n_pass}/{n_in} passed ({pass_rate:.1f}%). "
            f"Outputs in {output_dir}",
            file=sys.stderr,
        )
        return 0

    except Exception as err:
        return handle_cli_error(err, verbose=args.verbose)