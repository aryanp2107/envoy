"""
`envoy embed` — ESM-2 protein language model embedding with device auto-detection.

Thin wrapper around envoy.embed.run(). Produces mean-pooled embeddings suitable
for downstream clustering. Device selection priority: CUDA → MPS → CPU.
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
    """Register `envoy embed` with the top-level dispatcher."""
    parser = subparsers.add_parser(
        "embed",
        help="ESM-2 embedding with device auto-detection.",
        description=(
            "Embed protein sequences with ESM-2 (650M by default, 150M fallback). "
            "Mean-pools over residue positions, excluding BOS/EOS/padding. "
            "Device auto-detected: CUDA > MPS > CPU. If an upstream dedup "
            "accessions sidecar is present alongside the input FASTA, original "
            "headers are propagated forward for study traceability."
        ),
    )
    parser.add_argument(
        "--input", "-i", required=True, metavar="FASTA",
        help="Protein FASTA (typically from `envoy dedup`: "
             "<clade>.cdhit98.protein.fasta).",
    )
    parser.add_argument(
        "--output", "-o", required=True, metavar="NPY",
        help="Output path for embedding matrix (.npy). Sidecar files will be "
             "written alongside with matching stems: .accession_order.txt, "
             ".embed_log.json, and .accessions.tsv (if headers available).",
    )
    parser.add_argument(
        "--model", choices=["esm2_650m", "esm2_150m"], default="esm2_650m",
        help="ESM-2 variant (default: esm2_650m, 1280-dim embeddings). "
             "esm2_150m is a 640-dim CPU-friendly fallback at reduced quality.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, metavar="N",
        help="Batch size (default: 4). Reduce if OOM; the function auto-falls "
             "back to batch_size=1 on OOM, so the main cost of a larger value "
             "is one wasted batch.",
    )
    parser.add_argument(
        "--max-length", type=int, default=1024, metavar="N",
        help="Max tokenized length (default: 1024). Must accommodate BOS + "
             "sequence + EOS; HIV Env (~860 aa) fits comfortably.",
    )
    parser.add_argument(
        "--device", choices=["cuda", "mps", "cpu"], default=None,
        help="Force a compute device (default: auto-detect best available).",
    )
    parser.add_argument(
        "--truncate", action="store_true",
        help="Allow sequences longer than --max-length to be truncated. "
             "Without this flag, long sequences raise an error (safer default).",
    )
    parser.add_argument(
        "--dedup-accessions-tsv", metavar="TSV", default=None,
        help="Optional explicit path to the upstream dedup accessions sidecar. "
             "When omitted, envoy auto-discovers the sidecar alongside --input "
             "(<stem>.accessions.tsv). Use this if your dedup output is in a "
             "non-standard location.",
    )
    parser.add_argument(
        "--id-strategy", choices=["auto", "lanl", "genbank", "verbatim"],
        default="auto",
        help="FASTA header normalization strategy (default: auto-detect).",
    )

    add_common_flags(parser)
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Execute `envoy embed` with parsed arguments."""
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    try:
        if args.batch_size < 1:
            raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
        if args.max_length < 10:
            raise ValueError(
                f"--max-length must be >= 10 to accommodate meaningful sequences, "
                f"got {args.max_length}"
            )

        input_fasta = resolve_path(args.input, must_exist=True)
        output_npy = resolve_path(args.output, must_exist=False)

        dedup_sidecar = None
        if args.dedup_accessions_tsv is not None:
            dedup_sidecar = resolve_path(args.dedup_accessions_tsv, must_exist=True)

        id_strategy = IDStrategy(args.id_strategy)

        # Import here so argparse errors surface before the slow torch load
        from envoy.embed import ESM2Model, run as run_embed

        model = ESM2Model(args.model)

        embeddings, accessions = run_embed(
            input_fasta=input_fasta,
            output_npy=output_npy,
            model=model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            truncate=args.truncate,
            id_strategy=id_strategy,
            dedup_accessions_tsv=dedup_sidecar,
        )

        print(
            f"envoy embed: {len(accessions)} sequences embedded "
            f"(shape {embeddings.shape}, model {args.model}). "
            f"Output: {output_npy}",
            file=sys.stderr,
        )
        return 0

    except Exception as err:
        return handle_cli_error(err, verbose=args.verbose)