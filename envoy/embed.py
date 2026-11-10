"""
ESM-2 protein language model embedding with device auto-detection.

Produces mean-pooled embeddings from the ESM-2 650M model (or 150M fallback)
suitable for downstream clustering. Device selection priority: CUDA → MPS → CPU.

Mean-pooling convention: pools over positions 1 to seq_len+1, excluding the BOS
token (position 0) and all padding tokens. This matches the Phase B.1 AMP
validation protocol and must not be changed without re-validating.

Original headers from upstream dedup (if available) are propagated forward as
<output_npy_stem>.accessions.tsv for downstream study traceability.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader

from envoy.accession import IDStrategy, normalize_accession


logger = logging.getLogger(__name__)

# Suppress HuggingFace's load report when using AutoModel on an MLM checkpoint.
# The "UNEXPECTED lm_head.*" and "MISSING pooler.*" messages are expected and
# harmless for encoder-only embedding use; see embed.py docstring for details.
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class ESM2Model(str, Enum):
    """Supported ESM-2 model variants."""
    ESM2_650M = "esm2_650m"
    ESM2_150M = "esm2_150m"


# HuggingFace model IDs for each variant
MODEL_HF_IDS = {
    ESM2Model.ESM2_650M: "facebook/esm2_t33_650M_UR50D",
    ESM2Model.ESM2_150M: "facebook/esm2_t30_150M_UR50D",
}

# Embedding dimension per model (final hidden state dim)
MODEL_DIMS = {
    ESM2Model.ESM2_650M: 1280,
    ESM2Model.ESM2_150M: 640,
}


@dataclass
class EmbedConfig:
    """Configuration for one embedding run."""
    model: ESM2Model = ESM2Model.ESM2_650M
    batch_size: int = 4
    max_length: int = 1024  # covers HIV Env (~860 aa) with headroom
    device: Optional[str] = None  # None = auto-detect; else "cuda" | "mps" | "cpu"
    truncate: bool = False  # if True, long sequences are truncated; if False, errors


def detect_device() -> str:
    """
    Return the best available compute device.

    Priority: CUDA > MPS > CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _warn_cpu_slow(model: ESM2Model, n_sequences: int) -> None:
    """Print a one-time warning if the user is running CPU inference."""
    per_seq = 30 if model == ESM2Model.ESM2_650M else 6  # rough seconds
    est_min = (n_sequences * per_seq) / 60
    logger.warning(
        "Running ESM-2 %s inference on CPU. Expected: ~%d seconds per sequence "
        "(~%.1f minutes for %d sequences). Consider --model esm2_150m for ~5x speedup "
        "at reduced embedding quality, or use a CUDA/MPS-capable machine.",
        model.value, per_seq, est_min, n_sequences,
    )


def _load_dedup_sidecar(
    protein_fasta_path: Path,
    explicit_sidecar_path: Optional[Path] = None,
) -> dict[str, str]:
    """
    Load the accession → original_header map from the upstream dedup sidecar.

    Discovery logic:
      1. If explicit_sidecar_path is given and exists, use it.
      2. Otherwise, infer from the protein FASTA path.
         Dedup writes <clade>.cdhit98.protein.fasta alongside
         <clade>.cdhit98.accessions.tsv, so we strip ".protein" from the stem
         and look for ".accessions.tsv".

    Returns empty dict if no sidecar is found or the file lacks the
    original_header column (backward compat with pre-v0.2 dedup outputs).
    """
    sidecar_path = None

    if explicit_sidecar_path is not None:
        p = Path(explicit_sidecar_path)
        if p.exists():
            sidecar_path = p
        else:
            logger.warning(
                "dedup_accessions_tsv=%s does not exist; skipping sidecar propagation.",
                explicit_sidecar_path,
            )
            return {}
    else:
        # Auto-discovery: strip ".protein.fasta" and look for ".accessions.tsv"
        stem = protein_fasta_path.name
        for suffix in (".protein.fasta", ".fasta"):
            if stem.endswith(suffix):
                base_name = stem[: -len(suffix)]
                candidate = protein_fasta_path.parent / f"{base_name}.accessions.tsv"
                if candidate.exists():
                    sidecar_path = candidate
                    break

    if sidecar_path is None:
        return {}

    df = pd.read_csv(sidecar_path, sep="\t")
    if "accession" not in df.columns or "original_header" not in df.columns:
        logger.warning(
            "Sidecar %s is missing required columns (accession, original_header); "
            "skipping sidecar propagation.",
            sidecar_path,
        )
        return {}

    logger.info("Loaded %d original headers from dedup sidecar %s.", len(df), sidecar_path)
    return dict(zip(df["accession"].astype(str), df["original_header"].astype(str)))


def load_fasta_sequences(
    fasta_path: str | Path,
    id_strategy: IDStrategy = IDStrategy.AUTO,
) -> tuple[list[str], list[str]]:
    """
    Load sequences and accessions from a protein FASTA.

    Returns:
        (accessions, sequences) — lists of equal length in input order.

    Protein sequences are expected. ESM-2 doesn't handle `*` characters (internal
    stops from the QC gate 2 tolerance for ≤2 stops); they're stripped here.
    Ambiguous characters (X, B, Z, J, U, O) are preserved — ESM-2's tokenizer
    handles them as valid residue tokens.
    """
    accessions: list[str] = []
    sequences: list[str] = []
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        acc = normalize_accession(rec.description, id_strategy)
        seq = str(rec.seq).replace("*", "").upper()
        if not seq:
            logger.warning("Empty sequence for %s after cleanup; skipping.", acc)
            continue
        accessions.append(acc)
        sequences.append(seq)
    if not accessions:
        raise ValueError(f"No valid protein sequences loaded from {fasta_path}")
    return accessions, sequences


def _mean_pool_embeddings(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean-pool over residue positions, excluding BOS and padding.

    ESM-2's tokenizer prepends <cls> (BOS) and appends <eos>. Padding is to
    max_length when batching variable-length sequences. This function pools
    over positions 1..seq_len+1 and divides by the number of residue tokens.

    Args:
        hidden_states: (batch_size, seq_len, hidden_dim) from model output
        attention_mask: (batch_size, seq_len) with 1 for real tokens, 0 for padding

    Returns:
        (batch_size, hidden_dim) mean-pooled embeddings
    """
    # Zero out BOS (position 0) by excluding it from the mask
    mask = attention_mask.clone()
    mask[:, 0] = 0  # exclude <cls>
    # Also exclude the last non-pad position (<eos>) — it's a special token
    # The attention mask covers positions through <eos>, so we find the
    # last 1 in each row and set it to 0
    for i in range(mask.shape[0]):
        eos_idx = mask[i].nonzero().max().item()
        mask[i, eos_idx] = 0

    mask_f = mask.unsqueeze(-1).float()
    pooled = (hidden_states * mask_f).sum(dim=1)
    counts = mask_f.sum(dim=1).clamp(min=1)
    return pooled / counts


def embed_sequences(
    sequences: list[str],
    config: Optional[EmbedConfig] = None,
) -> np.ndarray:
    """
    Embed a list of protein sequences using ESM-2.

    Args:
        sequences: list of protein sequence strings
        config: embedding configuration (device, model, batch size, etc.)

    Returns:
        (n_sequences, embedding_dim) mean-pooled embedding matrix as a NumPy array
    """
    from transformers import AutoModel, AutoTokenizer

    if config is None:
        config = EmbedConfig()

    device = config.device or detect_device()
    if device == "cpu":
        _warn_cpu_slow(config.model, len(sequences))

    logger.info(
        "Loading %s on device '%s' (first run downloads ~%s to ~/.cache/huggingface/)",
        config.model.value, device,
        "2.5 GB" if config.model == ESM2Model.ESM2_650M else "650 MB",
    )

    hf_id = MODEL_HF_IDS[config.model]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id)
    model.eval()
    model.to(device)

    # Validate sequence lengths against max_length
    max_seq_len = max(len(s) for s in sequences)
    if max_seq_len + 2 > config.max_length:  # +2 for BOS and EOS
        if config.truncate:
            logger.warning(
                "Longest sequence is %d aa; truncating to %d (max_length - 2 for BOS/EOS).",
                max_seq_len, config.max_length - 2,
            )
        else:
            raise ValueError(
                f"Longest sequence is {max_seq_len} aa but max_length is "
                f"{config.max_length} (needs 2 extra for BOS/EOS). "
                f"Increase max_length or set truncate=True."
            )

    n_total = len(sequences)
    embed_dim = MODEL_DIMS[config.model]
    embeddings = np.zeros((n_total, embed_dim), dtype=np.float32)

    batch_size = config.batch_size
    n_batches = (n_total + batch_size - 1) // batch_size
    logger.info(
        "Embedding %d sequences in %d batches of %d on %s.",
        n_total, n_batches, batch_size, device,
    )

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_total)
            batch = sequences[start:end]

            tokens = tokenizer(
                batch,
                padding="longest",
                truncation=config.truncate,
                max_length=config.max_length if config.truncate else None,
                return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}

            try:
                outputs = model(**tokens, output_hidden_states=False)
                hidden = outputs.last_hidden_state
                pooled = _mean_pool_embeddings(hidden, tokens["attention_mask"])
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    logger.warning(
                        "OOM at batch_size=%d on %s. Retrying this batch with batch_size=1.",
                        batch_size, device,
                    )
                    # Clear cache and retry one by one
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    elif device == "mps":
                        torch.mps.empty_cache()
                    pooled_rows = []
                    for single_seq in batch:
                        single_tokens = tokenizer(
                            [single_seq],
                            padding="longest",
                            truncation=config.truncate,
                            max_length=config.max_length if config.truncate else None,
                            return_tensors="pt",
                        )
                        single_tokens = {k: v.to(device) for k, v in single_tokens.items()}
                        single_out = model(**single_tokens, output_hidden_states=False)
                        single_pooled = _mean_pool_embeddings(
                            single_out.last_hidden_state,
                            single_tokens["attention_mask"],
                        )
                        pooled_rows.append(single_pooled.cpu().numpy())
                    embeddings[start:end] = np.vstack(pooled_rows)
                    continue
                else:
                    raise

            embeddings[start:end] = pooled.cpu().numpy()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
                logger.info(
                    "Processed batch %d/%d (%d/%d sequences)",
                    batch_idx + 1, n_batches, end, n_total,
                )

    logger.info(
        "Done. Embedding matrix shape: %s, mean: %.4f, std: %.4f",
        embeddings.shape, embeddings.mean(), embeddings.std(),
    )
    return embeddings


def run(
    input_fasta: str | Path,
    output_npy: str | Path,
    model: ESM2Model = ESM2Model.ESM2_650M,
    batch_size: int = 4,
    max_length: int = 1024,
    device: Optional[str] = None,
    truncate: bool = False,
    id_strategy: IDStrategy = IDStrategy.AUTO,
    dedup_accessions_tsv: Optional[str | Path] = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Full embed workflow: load protein FASTA, embed, write outputs.

    Outputs created:
        <output_npy>                           — (N, embed_dim) embedding matrix
        <output_npy>.accession_order.txt       — one accession per line, matching matrix rows
        <output_npy>.accessions.tsv            — accession ↔ original_header mapping,
                                                 one row per matrix row (only written if
                                                 a dedup sidecar is available upstream)
        <output_npy>.embed_log.json            — model, device, runtime metadata

    Args:
        input_fasta: path to a protein FASTA (from `envoy dedup` or user-supplied)
        output_npy: path for the .npy embedding matrix
        model: ESM-2 variant to use
        batch_size: batch size (reduce if OOM)
        max_length: max tokenized length (must accommodate BOS + sequence + EOS)
        device: force a device; None = auto-detect
        truncate: if True, allow sequences longer than max_length to be truncated
        id_strategy: FASTA header normalization strategy
        dedup_accessions_tsv: explicit path to the upstream dedup sidecar. If None,
            embed auto-discovers <stem>.accessions.tsv alongside input_fasta. If
            found and valid, propagates it forward as <output_npy>.accessions.tsv.
            Omitted outputs if no sidecar is available (backward compat).

    Returns:
        (embeddings, accessions) — the matrix and matching accessions in order
    """
    import json
    import time

    output_npy = Path(output_npy)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    input_fasta_path = Path(input_fasta)

    # Load the dedup sidecar (if available) BEFORE sequence loading, so we can
    # warn immediately if the explicit path is wrong rather than after embedding.
    original_headers = _load_dedup_sidecar(
        input_fasta_path,
        Path(dedup_accessions_tsv) if dedup_accessions_tsv is not None else None,
    )

    accessions, sequences = load_fasta_sequences(input_fasta, id_strategy)

    config = EmbedConfig(
        model=model, batch_size=batch_size,
        max_length=max_length, device=device, truncate=truncate,
    )
    actual_device = device or detect_device()

    t_start = time.time()
    embeddings = embed_sequences(sequences, config)
    elapsed = time.time() - t_start

    # Write embedding matrix
    np.save(output_npy, embeddings)

    # Write accession-order file (unchanged)
    accession_order_path = output_npy.with_suffix(".accession_order.txt")
    with open(accession_order_path, "w") as fh:
        for acc in accessions:
            fh.write(f"{acc}\n")

    # Write accessions sidecar (new) — only if we have original headers to propagate
    sidecar_path: Optional[Path] = None
    if original_headers:
        sidecar_path = output_npy.with_suffix(".accessions.tsv")
        sidecar_rows = [
            {"accession": acc, "original_header": original_headers.get(acc, acc)}
            for acc in accessions
        ]
        pd.DataFrame(sidecar_rows).to_csv(sidecar_path, sep="\t", index=False)
        logger.info("Wrote accessions sidecar: %s", sidecar_path)

    # Write log
    log = {
        "model": model.value,
        "model_hf_id": MODEL_HF_IDS[model],
        "embedding_dim": MODEL_DIMS[model],
        "n_sequences": len(accessions),
        "device": actual_device,
        "batch_size": batch_size,
        "max_length": max_length,
        "pooling": "mean_over_residues_excluding_bos_eos_pad",
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_sequence": round(elapsed / len(accessions), 3) if accessions else None,
        "matrix_shape": list(embeddings.shape),
        "matrix_dtype": str(embeddings.dtype),
        "matrix_mean": float(embeddings.mean()),
        "matrix_std": float(embeddings.std()),
        "original_headers_preserved": bool(original_headers),
    }
    log_path = output_npy.with_suffix(".embed_log.json")
    with open(log_path, "w") as fh:
        json.dump(log, fh, indent=2)

    logger.info(
        "Wrote: %s (shape %s), %s, %s",
        output_npy, embeddings.shape, accession_order_path, log_path,
    )
    return embeddings, accessions