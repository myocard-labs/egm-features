# Changelog

All notable changes to `myocard-egm-features` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] — 2026-06-28

### Added

- `py.typed` marker (PEP 561) so downstream consumers and their CI can type-check against
  egm-features' public API; shipped as a wheel artifact.

## [0.1.0] — 2026-06-24

First release: a lean NumPy/SciPy library of per-trace morphology, spectral, and
complexity features for intracardiac EGMs, with a theory doc written before the code.

### Added

- **Three feature modules, 11 functions** — `time_domain` (`peak_to_peak`,
  `zero_crossings`, `activation_position` via dV/dt, `sec_peak_count`), `frequency`
  (`spectral_centroid`, `spectral_entropy`, `dominant_frequency`), and `complexity`
  (`sample_entropy`, `shannon_entropy`, `lempel_ziv_complexity` (LZ76), and
  `higuchi_fractal_dimension`).
- **`bundle.extract_all(signals, fs_hz)`** — returns a pandas DataFrame (N rows × 11
  feature columns), plus per-module `extract_time_domain` / `extract_frequency` /
  `extract_complexity` helpers for partial bundles.
- **Theory doc** (`docs/theory.md`) — math, parameter rationale, and a worked numeric
  example for every feature, written *before* the implementation.
- **Tests** — synthetic-signal anchors (pure sine, white noise, biphasic pulse, periodic
  binary) validating each feature against its theory-doc example.

### Dependencies

Lean and myocard-free: `numpy>=1.26`, `scipy>=1.11`, `pandas>=2.0`, `antropy>=0.1.6`.
No torch, no h5py, no internal myocard- dependencies.

[0.1.1]: https://github.com/myocard-labs/egm-features/releases/tag/v0.1.1
[0.1.0]: https://github.com/myocard-labs/egm-features/releases/tag/v0.1.0
