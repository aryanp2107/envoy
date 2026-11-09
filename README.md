# ENVOY

**E**mbedding-based selection of a**N**thentic **V**iral **O**rtholog representati**Y**es.

ENVOY selects authentic representative HIV-1 Env sequences per clade using protein
language model embeddings (ESM-2) and PAM k-medoids clustering. Unlike consensus or
mosaic approaches that synthesize artificial sequences, ENVOY returns real sequence
tips that retain native epistasis, quaternary trimer conformation, and real glycan
shield — making them directly usable as immunogen candidates in prime-shape-polish
regimens.

## Status

Pre-release (v0.1.0). Associated methods paper in preparation.

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

## Quick start

```bash
# Smoke test on bundled example data
envoy qc --example

# Full pipeline on your own clade (see docs for full flag reference)
envoy qc        --input raw.fasta --metadata raw.tsv --clade A6 --output results/qc/
envoy dedup     --input results/qc/A6.qc_pass.fasta --output results/dedup/A6/
envoy embed     --input results/dedup/A6.cdhit98.protein.fasta --output results/embeddings/A6.esm2_650m.npy
envoy cluster   --embeddings results/embeddings/A6.esm2_650m.npy --dedup-dir results/dedup/A6/ --output results/clusters/A6/
```

## Pipeline stages

| Command           | Purpose                                                              |
|-------------------|----------------------------------------------------------------------|
| `envoy qc`        | QC gates + Hypermut filtering on raw LANL FASTAs                     |
| `envoy dedup`     | CD-HIT 98% deduplication with smart-pick representative selection    |
| `envoy embed`     | ESM-2 650M mean-pool embedding (GPU-accelerated if available)        |
| `envoy cluster`   | PAM k-medoids clustering + k-sweep + N+2 backup panel                |
| `envoy figures`   | Per-clade and cross-clade publication figures                        |
| `envoy divergence`| Identity-vs-embedding divergence analysis                            |

## GPU support

`envoy embed` auto-detects the available compute device in priority order:
CUDA → Apple MPS → CPU. CPU inference works but is slow (~30-60s per sequence
with ESM-2 650M). For CPU-only users, `--model esm2_150m` gives ~6-10s per
sequence with reduced embedding quality.

## Citation

Methods paper forthcoming. For now, cite as:

> Patel, A. (2026). ENVOY: embedding-based selection of authentic HIV-1 Env
> representatives (v0.1.0) [Software]. https://github.com/aryanp2107/envoy

## License

MIT