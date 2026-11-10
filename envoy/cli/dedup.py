"""
`envoy dedup` — CD-HIT-EST deduplication with smart-pick representative selection.

Thin wrapper around envoy.dedup.run(). Requires cd-hit-est on PATH.
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
    """Register `envoy dedup` with the top-level dispatcher."""
    parser = subparsers.add_parser(
        "dedup",
        help="CD-HIT-EST deduplication with smart-pick representative selection.",
        description=(
            "Collapse near-duplicate sequences using CD-HIT-EST at a configurable "
            "identity threshold (default 98%). Within each cluster, pick a "
            "representative using three ordered rules: populated patient codes, "
            "non-AMP-breakthrough sequences, longest sequence as tiebreaker. "
            "Requires cd-hit-est on PATH."
        ),
    )
    parser.add_argument(
        "--nt", required=True, metavar="FASTA",
        help="Nucleotide FASTA from `envoy qc` (<clade>.qc_pass.fasta).",
    )
    parser.add_argument(
        "--protein", required=True, metavar="FASTA",
        help="Matched protein FASTA from `envoy qc` "
             "(<clade>.qc_pass.protein.fasta).",
    )
    parser.add_argument(
        "--qc-metadata", required=True, metavar="CSV",
        help="Per-sequence QC metadata CSV from `envoy qc` "
             "(<clade>.qc_pass.metadata.csv). Enables original header "
             "preservation; if the file has an original_header column, "
             "it's propagated forward as <clade>.cdhit98.accessions.tsv.",
    )
    parser.add_argument(
        "--raw-metadata", required=True, metavar="TSV",
        help="Raw LANL metadata TSV (used for smart-pick: patient codes, "
             "AMP breakthrough flags).",
    )
    parser.add_argument(
        "--clade", required=True, metavar="NAME",
        help="Clade/dataset name (used for output filenames).",
    )
    parser.add_argument(
        "--output", "-o", required=True, metavar="DIR",
        help="Output directory (created if missing).",
    )
    parser.add_argument(
        "--identity", type=float, default=0.98, metavar="FRAC",
        help="CD-HIT identity threshold (default: 0.98). Range 0.5-1.0; "
             "0.98 is the Phase B.2 pre-registered value for HIV-1 Env.",
    )
    parser.add_argument(
        "--threads", type=int, default=4, metavar="N",
        help="CPU threads for CD-HIT (default: 4).",
    )
    parser.add_argument(
        "--id-strategy", choices=["auto", "lanl", "genbank", "verbatim"],
        default="auto",
        help="FASTA header normalization strategy (default: auto-detect).",
    )

    add_common_flags(parser)
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Execute `envoy dedup` with parsed arguments."""
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    try:
        # Validate identity bounds early (cleaner than CD-HIT's error later)
        if not 0.5 <= args.identity <= 1.0:
            raise ValueError(
                f"--identity must be in [0.5, 1.0], got {args.identity}"
            )
        if args.threads < 1:
            raise ValueError(f"--threads must be >= 1, got {args.threads}")

        nt_fasta = resolve_path(args.nt, must_exist=True)
        protein_fasta = resolve_path(args.protein, must_exist=True)
        qc_metadata = resolve_path(args.qc_metadata, must_exist=True)
        raw_metadata = resolve_path(args.raw_metadata, must_exist=True)
        output_dir = resolve_path(args.output, must_exist=False)

        id_strategy = IDStrategy(args.id_strategy)

        # Import here so argparse errors show before expensive module loads
        from envoy.dedup import run as run_dedup

        df = run_dedup(
            input_nt_fasta=nt_fasta,
            input_protein_fasta=protein_fasta,
            metadata_csv=qc_metadata,
            raw_metadata_tsv=raw_metadata,
            clade=args.clade,
            output_dir=output_dir,
            identity_threshold=args.identity,
            n_threads=args.threads,
            id_strategy=id_strategy,
            qc_metadata_csv=qc_metadata,
        )

        n_reps = len(df)
        print(
            f"envoy dedup: {n_reps} representatives selected at "
            f"{args.identity:.0%} identity. Outputs in {output_dir}",
            file=sys.stderr,
        )
        return 0

    except Exception as err:
        return handle_cli_error(err, verbose=args.verbose)