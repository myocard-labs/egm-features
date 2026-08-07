# myocard-egm-features

> Per-trace morphology, spectral, and complexity feature extractors for
> intracardiac bipolar EGM signals.

Part of the [myocard-labs](https://github.com/myocard-labs) cardiac
signal-processing toolkit.

---

## Why

`myocard-egm-features` extracts scalar features from already-
preprocessed 1-D bipolar EGM traces. Each public function takes a
NumPy array of shape `(T,)` and returns a single number; a bundle
helper runs every feature over an `(N, T)` batch and returns a tidy
`pandas.DataFrame` suitable for joining with per-trace metadata.

Eleven features in three families:

- **Time-domain** — `peak_to_peak`, `zero_crossings`,
  `activation_position`, `sec_peak_count`.
- **Frequency** — `spectral_centroid`, `spectral_entropy`,
  `dominant_frequency`.
- **Complexity** — `sample_entropy`, `shannon_entropy`,
  `lempel_ziv_complexity`, `higuchi_fractal_dimension`.

The feature set follows Sánchez et al. 2021 (Frontiers in Physiology):
the seven features they identified as discriminating fibrotic from
non-fibrotic atrial substrate plus four amplitude / timing / spectral
additions used by adjacent literature. See [`docs/theory.md`](docs/theory.md)
for the math behind each feature and the parameter-choice rationale.

What this repo does NOT do: bandpass, calibrate, R-wave-anchor,
resample, or otherwise preprocess the input — those belong upstream.
It also does not do HDF5 I/O, classification, or simulation. Hand it
a clean, bandpassed, calibrated trace and a sample rate, and it
returns the features. The library has **no internal myocard- deps** —
pure `numpy` + `scipy` + `pandas` + `antropy`. Anyone who wants to
extract features from an EGM trace in a notebook can
`pip install myocard-egm-features` and skip everything else in the
project.

Two feature families ship: the **eleven** clinically-grounded
morphology, spectral, and complexity features this library started
with, and the **catch22** set (22, or 24 with `catch24`'s mean and
standard deviation). They are complementary rather than overlapping —
catch22 is computed on the z-scored trace and so cannot see amplitude
at all, which is the axis voltage mapping is built on.

---

## Install

From PyPI (when published):

```bash
pip install myocard-egm-features            # the eleven native features
pip install "myocard-egm-features[catch22]" # + the catch22 set
```

The `catch22` extra is optional because it is the one dependency that
is not pure Python. `pycatch22` publishes no wheels, so pip builds it
from source and **a C compiler must be available** (`build-essential`
on Debian/Ubuntu). Keeping it out of the base install means the eleven
native features stay installable anywhere; a base install that asks
for a catch22 feature fails with instructions rather than a confusing
`ModuleNotFoundError`.

From source during pre-1.0 iteration:

```bash
pip install git+https://github.com/myocard-labs/egm-features.git
```

Editable install for development:

```bash
git clone https://github.com/myocard-labs/egm-features.git
cd egm-features
pip install -e ".[dev]"
pre-commit install
```

Runtime deps: `numpy>=1.26,<2.5`, `scipy>=1.11`, `pandas>=2.0`,
`antropy>=0.1.6`, plus `pycatch22>=0.4.5` with the `catch22` extra.
No PyTorch, no h5py.

---

## Programmatic usage

Three ways to call the library, from most-batch to most-bespoke.

### Full bundle (the 90% case)

```python
import numpy as np
from myocard_egm_features import bundle

# (N, T) batch of N=100 bandpassed, calibrated traces at fs=1 kHz.
signals = np.random.standard_normal((100, 512))

df = bundle.extract_all(signals, fs_hz=1000.0)
df.shape       # (100, 11)
df.columns     # ['peak_to_peak', 'zero_crossings', ..., 'higuchi_fractal_dimension']
```

`bundle.extract_all` applies the project's standard policy values
(`method="dvdt_max"`, `threshold_frac=0.3`, `binarize_method="median"`).

### Choosing which features to compute

The default is the eleven above. `features=` narrows or widens it —
by set name or by an explicit list — and **only what you ask for is
computed**, which is the main performance lever:

```python
# A named set. No fs_hz needed: no catch22 feature takes one.
df = bundle.extract_all(signals, features="catch22")

# An explicit study set, spanning both families.
df = bundle.extract_all(signals, features=["peak_to_peak", "trev", "entropy_pairs"])
```

Columns come back in a canonical order regardless of the order you
asked for them, so two callers requesting the same features get
identically-ordered frames.

`fs_hz` is required only when a requested feature actually needs it —
the three spectral features of `docs/theory.md` §2. Omitting it when
something does need it raises an error naming which one.

### Per-module batch helpers

When you only need one feature group — e.g. the cheap time-domain
features over a large batch, skipping the slow complexity step:

```python
from myocard_egm_features import bundle

df_time = bundle.extract_time_domain(signals)                # (N, 4)
df_freq = bundle.extract_frequency(signals, fs_hz=1000.0)    # (N, 3)
df_complex = bundle.extract_complexity(signals)              # (N, 4)
```

`extract_frequency` computes the periodogram once per trace and
reuses it across the three frequency features.

### Individual feature functions

When you want one feature, on one trace, ad hoc:

```python
from myocard_egm_features import time_domain, frequency, complexity

x = signals[0]  # one trace, shape (T,)

p2p     = time_domain.peak_to_peak(x)
ap      = time_domain.activation_position(x, method="dvdt_max")
freqs, psd = frequency.periodogram(x, fs_hz=1000.0)
centroid = frequency.spectral_centroid(x, fs_hz=1000.0, freqs=freqs, psd=psd)
se      = complexity.sample_entropy(x)
lz      = complexity.lempel_ziv_complexity(x, binarize_method="median")
```

Direct callers must supply policy parameters explicitly (no defaults
on `method` / `threshold_frac` / `binarize_method`) — the bundle is
the one place where project policy lives. See
[`docs/usage.md`](docs/usage.md) for the full walkthrough including
the math-vs-policy split and performance notes.

---

## Module map

| Module | What's in it |
|---|---|
| `myocard_egm_features.time_domain` | `peak_to_peak`, `zero_crossings`, `activation_position` (dV/dt or abs-peak), `sec_peak_count`. |
| `myocard_egm_features.frequency` | `periodogram` (public PSD helper), `spectral_centroid`, `spectral_entropy`, `dominant_frequency`. PSD-reuse path via optional `freqs`/`psd` kwargs. |
| `myocard_egm_features.complexity` | `sample_entropy` (Richman 2000), `shannon_entropy` (histogram-based), `lempel_ziv_complexity` (LZ76 + median binarize), `higuchi_fractal_dimension` (Higuchi 1988). |
| `myocard_egm_features.catch22` | The catch22 set (`docs/theory.md` §4): `catch22_all`, `catch22_features` (a named subset), `catch22_feature` (one). Needs the `catch22` extra. |
| `myocard_egm_features.providers` | The `FeatureProvider` seam — `NativeProvider`, `Catch22Provider` — plus the project-standard policy constants (`ACTIVATION_METHOD`, `LZ_BINARIZE_METHOD`, …), which `bundle` re-exports under their original names. |
| `myocard_egm_features.sets` | The feature-set registry: what exists, who computes it, and the named groupings. `resolve`, `group_by_provider`, `check_available`. |
| `myocard_egm_features.bundle` | `extract_all` (with `features=` selection) + `extract_catch22` + the per-module `extract_<module>` helpers. |

---

## Tests

```bash
pytest                  # full suite
pytest --cov            # with coverage
ruff check .            # lint
ruff format --check .   # format check
mypy                    # type check
```

CI runs the same checks on Python 3.10, 3.11, and 3.12 — see
`.github/workflows/ci.yml`. The complexity tests are the slow ones
(`sample_entropy` is ~O(T²)); the full suite runs in under 10 seconds
end-to-end against the synthetic-signal anchors in
[`docs/theory.md`](docs/theory.md).

---

## Project status

Pre-1.0; expect breaking changes across minor versions until the API
stabilizes. The current release is `v0.1.0`. No internal myocard-
runtime dependencies — the only deps are upstream
(`numpy`, `scipy`, `pandas`, `antropy`).

- For end-user usage examples + the math-vs-policy split walkthrough,
  see [`docs/usage.md`](docs/usage.md).
- For the math behind every feature (derivations + worked examples
  for code reviewers without prior entropy / fractal-dimension
  background), see [`docs/theory.md`](docs/theory.md).
- For the design rationale (why pure functions, why pandas, why the
  math-vs-policy split), see [`project/architecture.md`](project/architecture.md).
- For the v0.2.0+ plan (typed-contract bundle output, Wittkampf-
  smoothed dV/dt, more features), see [`project/roadmap.md`](project/roadmap.md).
- For the broader refactor context, see
  `intracardiac-platform/project/project_plan.md`.

---

## Citation

If you use this software in academic work, please cite both the
Sánchez 2021 feature set and this implementation:

```bibtex
@article{sanchez2021ranking,
  author  = {S{\'a}nchez, Jorge and Trenor, Beatriz and Saiz, Javier
             and Loewe, Axel},
  title   = {Ranking the Influence of Tissue Conductivities on
             Forward-Calculated ECGs},
  journal = {Frontiers in Physiology},
  volume  = {12},
  year    = {2021},
}

@software{klein_myocard_egm_features_2026,
  author  = {Klein, Daniel},
  title   = {myocard-egm-features: per-trace morphology, spectral,
             and complexity feature extractors for intracardiac
             bipolar EGM signals},
  year    = {2026},
  url     = {https://github.com/myocard-labs/egm-features},
}
```

The full mathematical reference list — Richman & Moorman 2000
(sample entropy), Lempel & Ziv 1976 (LZ complexity), Higuchi 1988
(fractal dimension), Pincus 1991 (parameter guidance) — lives in
[`docs/theory.md`](docs/theory.md) §4.

---

## License

MIT — see [LICENSE](LICENSE). Third-party software-license
acknowledgements are in [NOTICE](NOTICE).
