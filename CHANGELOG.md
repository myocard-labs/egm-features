# Changelog

All notable changes to `myocard-egm-features` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-08-06

Adds the **catch22** feature set alongside the original eleven, and makes feature
selection explicit so a caller pays only for what it asks for.

### Added

- **The catch22 set** (`catch22` module) — the 22 canonical features of Lubba et al.
  2019 plus the two `catch24` additions, as thin wrappers over the reference C
  implementation. Three narrowing entry points: `catch22_all`, `catch22_features`
  (a named subset), `catch22_feature` (one). Shipped as an **optional extra**,
  `myocard-egm-features[catch22]`.
- **`docs/theory.md` §4** — the math, a worked numeric anchor, and an EGM reading for
  every one of the 24, plus per-family notes on what combinations reveal. Written
  before the code, per the theory-first rule; the anchors double as the test spec.
- **A feature-provider seam** (`providers`) — `FeatureProvider`, `NativeProvider`,
  `Catch22Provider`. One interface over "which features exist, can they run here, and
  what are their values", so consumers need not know which family a name belongs to.
- **A feature-set registry** (`sets`) — `FEATURE_REGISTRY`, the named sets
  (`egm_features`, `catch22`, `catch24`, `egm_features+catch22`), `resolve`,
  `group_by_provider`, `check_available`. Pure metadata; works without the extra
  installed, because deciding whether to install a toolchain should not require one.
- **`bundle.extract_all(features=...)`** and **`bundle.extract_catch22()`** — select by
  set name or explicit list. Only the requested features are computed.
- **An aggregated `RuntimeWarning`** when a batch produces `NaN`, naming the trace count
  and worst-affected features. One per batch, not per trace.
- **A base-install CI job** — the eleven native features are tested with no `pycatch22`
  and no C toolchain, with a guard that fails the build if the extra sneaks in.

### Changed

- **`fs_hz` is now optional on `extract_all`**, required only when a requested feature
  needs it (the three §2 spectral features, listed in `providers.REQUIRES_FS_HZ`). The
  error names which feature wanted it. Omitting it on the default selection now raises
  `ValueError` rather than `TypeError`.
- **The project-standard policy constants moved** from `bundle` to `providers`, where
  the code that applies them lives. `bundle` re-exports them, so `bundle.ACTIVATION_METHOD`
  and friends are unchanged.
- `numpy` capped below 2.5: its stubs use PEP-695 syntax that mypy cannot parse at this
  project's `python_version`. Dev tooling pinned exactly (`ruff==0.15.17`, `mypy==2.1.0`).
- `__version__` derives from installed metadata instead of a hardcoded constant.

### Fixed

- **Documentation correction:** `sample_entropy` was described as the extraction
  bottleneck. Measured at `T = 192` it is 14% of the cost, and `shannon_entropy` is 3.5×
  more expensive. The original holds at `T = 512`; both documents now say the profile
  must be re-measured when `T` changes.
- `.gitignore` output-dir patterns root-anchored, so an unanchored `data/` can no longer
  swallow a same-named source package.

### Unchanged, deliberately

`extract_all`'s default output — the same eleven columns, in the same order, with the
same dtypes. Verified against the v0.1.1 implementation with
`pandas.testing.assert_frame_equal` rather than asserted by eye.

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

[0.2.0]: https://github.com/myocard-labs/egm-features/releases/tag/v0.2.0
[0.1.1]: https://github.com/myocard-labs/egm-features/releases/tag/v0.1.1
[0.1.0]: https://github.com/myocard-labs/egm-features/releases/tag/v0.1.0
