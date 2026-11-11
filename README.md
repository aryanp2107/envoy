# ENVOY

https://aryanp2107.github.io/envoy/

**E**mbedding-based selection of a**N**thentic **V**iral **O**rtholog representati**Y**es.

ENVOY selects authentic representative HIV-1 Env sequences per clade using protein
language model embeddings (ESM-2) and PAM k-medoids clustering. Unlike consensus or
mosaic approaches that synthesize artificial sequences, ENVOY returns real sequence
tips that retain native epistasis, quaternary trimer conformation, and real glycan
shield — making them directly usable as immunogen candidates in prime-shape-polish
regimens.

## Status

Pre-release (v0.1.0).

## Install

Core (analysis pipeline, no GPU required):

```bash
pip install envoy-hiv
```

With embedding support (installs torch + fair-esm; needed to run `envoy embed` locally):

```bash
pip install "envoy-hiv[embed]"
```

Development install from a local checkout:

```bash
git clone https://github.com/aryanp2107/envoy
cd envoy
pip install -e ".[embed,dev]"
```

External dependency for the dedup stage: **CD-HIT** (`cd-hit-est` on PATH). Install
via `brew install cd-hit` on macOS or `conda install -c bioconda cd-hit` otherwise.

## Quick start

```bash
# Smoke test on bundled example data
envoy qc --example

# Full pipeline on your own clade
envoy qc        --input raw.fasta --metadata raw_meta.tsv --clade A6 --output results/qc/
envoy dedup     --input results/qc/A6.qc_pass.fasta --output results/dedup/A6/
envoy embed     --input results/dedup/A6.cdhit98.protein.fasta --output results/embeddings/A6.esm2_650m.npy
envoy cluster   --embeddings results/embeddings/A6.esm2_650m.npy --dedup-dir results/dedup/A6/ --output results/clusters/A6/
```

## Pipeline stages

| Command           | Purpose                                                              |
|-------------------|----------------------------------------------------------------------|
| `envoy qc`        | QC gates + Hypermut filtering on raw FASTAs                          |
| `envoy dedup`     | CD-HIT 98% deduplication with smart-pick representative selection    |
| `envoy embed`     | ESM-2 650M mean-pool embedding (GPU-accelerated if available)        |
| `envoy cluster`   | PAM k-medoids clustering + k-sweep + N+2 backup panel                |
| `envoy figures`   | Per-clade and cross-clade publication figures                        |
| `envoy divergence`| Identity-vs-embedding divergence analysis                            |

## Input conventions

ENVOY accepts FASTAs in any standard naming convention. Headers are normalized to
canonical accessions at ingestion so downstream stages (dedup, embed, cluster) can
merge cleanly with user-supplied metadata.

**Supported formats** (auto-detected):

| Convention | Example header | Normalized to |
|---|---|---|
| LANL     | `>B.US.1990.WEAU16.U21135`                   | `U21135` |
| GenBank  | `>U21135.1 Human immunodeficiency virus...`  | `U21135` |
| HIV-TRACE | `>B\|US\|1990\|WEAU16\|U21135`               | `U21135` |
| Custom   | `>patient_42_env` or `>seq-001_Lab_2024`     | preserved as-is |

**Normalization priority (AUTO mode)**:
1. LANL-style dotted headers → last field (the accession)
2. Pipe-delimited headers → last field, then re-normalize
3. GenBank accession patterns (`U21135.1`) → base accession (`U21135`)
4. Anything else → first whitespace-delimited token verbatim

**Override** with `--id-strategy`:
- `--id-strategy auto` (default): the priority above
- `--id-strategy lanl`: force LANL split-on-dot
- `--id-strategy genbank`: force GenBank version-stripping
- `--id-strategy verbatim`: use header as-is, no normalization

If normalization causes two different headers to collapse to the same accession,
ENVOY raises an error with the collision details. Use `--id-strategy verbatim`
in that case to preserve all IDs.

## The QC stage in detail

`envoy qc` applies six gates in sequence to filter raw Env sequences:

| Gate | What it checks | Behavior | Hard/Soft |
|---|---|---|---|
| 6 | M-group membership | Reject Group O/N/P sequences | Hard |
| 1 | Env coverage | Require ≥2000 nt (covers HXB2 env 6225-8795) | Hard |
| 2 | Translation | Longest ORF, 600-900 aa, ≤2 internal stops | Hard |
| 3 | Start methionine | Flag sequences without a leading M | Soft |
| 4 | Ambiguity | Record % ambiguous nucleotides; no rejection | Informational |
| 5 | Hypermutation | Rose & Korber Fisher's exact p < 0.05, GpR-biased direction | Hard |

The gate thresholds were optimized across 11 HIV-1 clades (A1, A6, B, C, D, G, F1,
CRF01_AE, CRF02_AG, CRF07_BC, CRF08_BC) to achieve 85-95% pass rates while reliably
removing APOBEC hypermutants and translation-broken sequences. Sequences passing all
hard gates are written to `<clade>.qc_pass.fasta` (nucleotide) and
`<clade>.qc_pass.protein.fasta` (translated). Every input sequence appears in
`<clade>.qc_pass.metadata.csv` with its gate results and failure reasons, regardless
of whether it passed, enabling downstream audit.

**Hypermut reference consensus**: Gate 5 compares each sequence against its clade's
LANL consensus env sequence. Fallbacks are used for clades without their own
consensus (A6 → CONSENSUS_A1; CRF07_BC → CONSENSUS_08_BC). The consensus FASTA path
is supplied via `--consensus-fasta`.

### QC command reference

```bash
envoy qc \
    --input sequences.fasta \
    --metadata metadata.tsv \
    --clade <CLADE> \
    --consensus-fasta path/to/lanl_consensus_env.fas \
    --output results/qc/ \
    [--id-strategy {auto,lanl,genbank,verbatim}] \
    [--hypermut-p-threshold 0.05]
```

**Required**:
- `--input`: nucleotide FASTA of raw Env sequences
- `--clade`: clade identifier (determines Hypermut reference consensus to use)
- `--consensus-fasta`: LANL consensus env alignment FASTA (any release)
- `--output`: directory for QC outputs

**Optional**:
- `--metadata`: tab-delimited metadata file (LANL-formatted or user-supplied).
  If present, propagates to downstream dedup smart-pick. Not required for QC
  itself.
- `--id-strategy`: header normalization strategy (see Input conventions)
- `--hypermut-p-threshold`: Fisher's exact p threshold for Gate 5 (default 0.05)

### QC outputs
<output_dir>/
├── <clade>.qc_pass.fasta Surviving nucleotide sequences
├── <clade>.qc_pass.protein.fasta Translated protein sequences
├── <clade>.qc_pass.metadata.csv Per-sequence gate results (ALL input rows)
└── <clade>.qc_summary.json Aggregate pass/fail statistics


Every input sequence has a row in `qc_pass.metadata.csv` with columns:
`accession, env_nt_length, env_coverage_confidence, protein_length,
n_internal_stops, has_start_met, hypermut_pvalue, ambiguity_pct, flags, passed`.
Sequences rejected at any hard gate have `passed=False` with the `flags` column
naming the failure reason.

## GPU support

`envoy embed` auto-detects the available compute device in priority order:
**CUDA → Apple MPS → CPU**.

- **CUDA (NVIDIA)**: ~0.01 s/sequence with ESM-2 650M. Full validated workflow.
- **MPS (Apple Silicon M1/M2/M3)**: ~0.1-0.5 s/sequence. Full ESM-2 650M on-device.
- **CPU fallback**: ~30-60 s/sequence with ESM-2 650M. Use `--model esm2_150m`
  for ~6-10 s/sequence at reduced embedding quality (the paper does not validate
  the 150M model against antigenic distance).

First `envoy embed` run downloads ESM-2 650M weights (~2.5 GB) to the HuggingFace
cache directory (`~/.cache/huggingface/`). Subsequent runs reuse the cache.


## License

MIT
