# ENVOY configs

This directory contains frozen configuration files that govern the behavior
of the pipeline for a given release. These files exist so that every panel
in a release is generated under identical parameters — panels should be
comparable across clades within a release, and a release's results should
be exactly reproducible given the config files alone.

## Current configs

- `cross_clade_qc_v0.1.0.yaml` — QC gate configuration for the v0.1.0
  panel release. Applies uniformly to all 11 clades: A1, A6, B, C, D,
  F1, G, CRF01_AE, CRF02_AG, CRF07_BC, CRF08_BC.

## Config change policy

Configs are versioned and immutable. Once a config is locked for a
release:

- **Do not edit it in place.** A new release gets a new config file
  with an incremented version (e.g., `cross_clade_qc_v0.2.0.yaml`).
- **Clades within a release use the same config.** Mixing clades
  processed under different configs in a single release breaks the
  cross-clade comparability guarantee.
- **Document the reason for a version bump** in the new config's
  comments and in the release notes.

## Why this matters

ENVOY's cross-clade claims — that clustering structure differences
between clades (e.g., A6 single-founder vs. CRF01_AE multi-epidemic)
are biological rather than methodological artifacts — depend on every
clade being processed identically. If CRF08_BC was QC'd under
permissive gates (600 aa floor) and Clade C under tighter gates (700
aa floor), a reviewer could reasonably ask whether the structural
contrast reflects biology or just parameter drift. Config locking
forecloses that question.
