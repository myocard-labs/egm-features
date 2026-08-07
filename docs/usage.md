# Usage — myocard-egm-features

How to call the public API. For the math behind each feature see
[`theory.md`](theory.md); for the design rationale (why pure
functions, why a pandas DataFrame, why the math-vs-policy split) see
[`../project/architecture.md`](../project/architecture.md).

## What this library is — and is not

`myocard-egm-features` extracts **scalar features** from
**already-preprocessed** 1D bipolar EGM traces. Each public function
takes a NumPy array of shape `(T,)` and returns a single number.

This library is *not* a preprocessing stack. It will not bandpass,
calibrate, downsample, R-wave-anchor, or otherwise alter the input —
those steps belong upstream (`egm-signal` for DSP, the per-producer
pipeline for calibration and anchoring). Hand `egm-features` a clean,
bandpassed, mV-calibrated trace and a sample rate, and it returns the
features.

## Installation

```bash
pip install myocard-egm-features            # the eleven native features
pip install "myocard-egm-features[catch22]" # + the catch22 set
```

Editable / development install from a clone:

```bash
git clone https://github.com/myocard-labs/egm-features
cd egm-features
pip install -e ".[dev,catch22]"
```

Dependencies are lean: `numpy>=1.26,<2.5`, `scipy>=1.11`,
`pandas>=2.0`, `antropy>=0.1.6`. No PyTorch, no HDF5, no internal
`myocard-` deps.

### The catch22 extra needs a C compiler

`pycatch22` publishes no wheels, so pip builds it from source and a
toolchain must be present — `build-essential` on Debian/Ubuntu. That
is why it is optional rather than a base dependency: the eleven native
features stay installable anywhere.

A base install is fully supported and CI tests it as its own job. Ask
for a catch22 feature without the extra and you get instructions, not
a puzzle:

```
catch22 features require the optional 'catch22' extra, which is not installed.
    pip install "myocard-egm-features[catch22]"
pycatch22 publishes no wheels, so pip builds it from source: a C compiler
must be available (build-essential on Debian/Ubuntu).
The eleven native features in docs/theory.md §1-§3 need no extra and are
always available.
```

You can always ask *what* a feature set contains without the extra
installed — deciding whether a toolchain is worth it should not itself
require the toolchain.

## Quickstart

The 90% use case: extract the eleven native features over a batch
of traces. See [Choosing which features to compute](#choosing-which-features-to-compute)
for catch22 and for narrowing the set.

```python
import numpy as np
from myocard_egm_features import bundle

# An (N, T) array of N=100 traces at fs=1 kHz, T=512 samples.
signals = np.random.standard_normal((100, 512))

df = bundle.extract_all(signals, fs_hz=1000.0)

print(df.shape)           # (100, 11)
print(df.columns.tolist())
# ['peak_to_peak', 'zero_crossings', 'activation_position', 'sec_peak_count',
#  'spectral_centroid', 'spectral_entropy', 'dominant_frequency',
#  'sample_entropy', 'shannon_entropy', 'lempel_ziv_complexity',
#  'higuchi_fractal_dimension']
```

`extract_all` returns a `pandas.DataFrame` with one row per trace,
columns in [`theory.md`](theory.md) §1 → §2 → §3 order. Join it with
per-trace metadata (sim_id, patient_id, label) from your bank record
in the usual pandas way.

## Choosing which features to compute

`extract_all`'s default is the eleven native features, unchanged since
v0.1.0. Pass `features=` to ask for something else — a set name or an
explicit list — and **only what you ask for is computed**:

```python
from myocard_egm_features import bundle, sets

bundle.extract_all(signals, fs_hz=1000.0)                    # the default 11
bundle.extract_all(signals, features="catch22")              # the canonical 22
bundle.extract_all(signals, features="egm_features+catch22") # all 33
bundle.extract_catch22(signals, catch24=True)                # 22 + mean, std_dev

# A study set: amplitude and fractionation from the native features,
# morphology and dynamics from catch22.
bundle.extract_all(signals, features=["peak_to_peak", "sec_peak_count",
                                      "trev", "entropy_pairs"])
```

The shipped set names:

| set | features | needs the extra |
|---|---|---|
| `egm_features` | the native 11 (§1–§3) | no |
| `catch22` | the canonical 22 (§4) | yes |
| `catch24` | 22 + `mean`, `std_dev` (§4.11) | yes |
| `egm_features+catch22` | 33 | yes |

These are **structural and canonical groupings only**. There is
deliberately no "the features that work well for my study" set: that
judgement is true for one analysis and false for the next, and this
library cannot know which situation you are in. Build your own from
names — `sets.resolve([...])` validates them, rejects typos, and
returns canonical order. `docs/theory.md` §4.1.3 is the reading for
deciding which catch22 features suit a given window length.

Columns always come back in canonical order (§1–§3, then §4)
regardless of request order, so two callers asking for the same
features get identically-ordered frames.

### fs_hz is required only when something needs it

Only the three spectral features of §2 take a sample rate. Any catch22
selection, or a time-domain/complexity one, can omit it:

```python
bundle.extract_all(signals, features="catch22")            # fine, no fs_hz
bundle.extract_all(signals, features=["peak_to_peak"])     # also fine
bundle.extract_all(signals, features=["dominant_frequency"])
# ValueError: fs_hz is required for ['dominant_frequency'] ...
```

Not *taking* `fs_hz` is not the same as being sample-rate independent.
`forecast_error` forecasts three samples ahead — 3 ms at 1 kHz, 6 ms at
500 Hz — and the autocorrelation timescales are measured in samples.
Those values are only comparable between datasets recorded at the same
rate.

## NaN on degenerate input, and the warning

Features are `NaN` where the math is undefined: on a constant trace,
19 of the 22 catch22 features are. Those values are **propagated, not
replaced** — substituting `0.0` would let a meaningless number enter a
distribution comparison as though it were real, whereas a `NaN` column
is visible and droppable.

Because silence is its own failure mode on a long run, a batch that
produced any `NaN` raises **one** aggregated `RuntimeWarning`:

```
3 of 1000 traces produced NaN features: mode_5 (3), mode_10 (3), ... .
NaN is propagated, not replaced, so these reach your DataFrame as-is.
Usual causes: a constant or near-constant trace, a dropped channel, or
non-finite samples.
```

One per batch, not per trace. The values themselves are untouched —
the warning is diagnostic only.

> A **short** trace is not detected. `pycatch22` returns numbers for a
> five-sample input rather than `NaN`, and nothing here will tell you
> they are meaningless. `docs/theory.md` §4.1.3 documents which
> features want longer windows.

## Three ways to use the library

### 1. Individual feature functions (per-trace)

When you want one feature, on one trace, ad hoc. Each function lives
in its module and takes a 1D `(T,)` array:

```python
from myocard_egm_features import time_domain, frequency, complexity

x = signals[0]  # one trace, shape (T,)

p2p = time_domain.peak_to_peak(x)
zc = time_domain.zero_crossings(x)

# Functions that need policy values require them explicitly — no defaults.
ap = time_domain.activation_position(x, method="dvdt_max")
spc = time_domain.sec_peak_count(x, threshold_frac=0.3)

# Frequency features need fs_hz.
centroid = frequency.spectral_centroid(x, fs_hz=1000.0)
entropy = frequency.spectral_entropy(x, fs_hz=1000.0)
peak_f = frequency.dominant_frequency(x, fs_hz=1000.0)

# Complexity features.
se = complexity.sample_entropy(x)                              # math defaults
lz = complexity.lempel_ziv_complexity(x, binarize_method="median")
```

Direct callers of the per-feature functions are required to supply
the policy values (`method`, `threshold_frac`, `binarize_method`)
explicitly — there are no defaults. This is intentional: it forces
you to think about which policy you want at every call site, and
prevents accidental drift across the codebase. The bundle helpers
(below) hardcode the project-standard policies so most callers don't
need to remember them.

### 2. Per-module batch helpers

When you want one feature *group* over a batch — e.g. only the time-
domain features, because the complexity features are slow:

```python
from myocard_egm_features import bundle

df_time = bundle.extract_time_domain(signals)               # (N, 4)
df_freq = bundle.extract_frequency(signals, fs_hz=1000.0)   # (N, 3)
df_complex = bundle.extract_complexity(signals)             # (N, 4)
```

Each helper takes `(N, T)` and returns an N-row DataFrame with the
columns from that feature group. `extract_frequency` is the only one
that requires `fs_hz` since the other two don't use it.

These helpers hardcode the project-standard policy values internally,
so you don't need to supply `method` / `threshold_frac` /
`binarize_method` — the bundle will pass them through. The exact
values used are surfaced as module-level constants in
`bundle.py`:

```python
bundle.ACTIVATION_METHOD            # "dvdt_max"
bundle.SEC_PEAK_THRESHOLD_FRAC      # 0.3
bundle.LZ_BINARIZE_METHOD           # "median"
bundle.SAMPLE_ENTROPY_M             # 2
bundle.SAMPLE_ENTROPY_R_FRAC        # 0.2
bundle.SHANNON_ENTROPY_N_BINS       # 10
bundle.HIGUCHI_K_MAX                # 10
```

If a downstream consumer ever needs to deviate from these, do it by
calling the per-feature functions directly (mode 1 above) — *not* by
mutating the constants. The constants are a documentation surface;
they should never change at runtime.

### 3. The full bundle

`bundle.extract_all(signals, fs_hz)` is what you want when you're
running every feature over a batch. It's a thin wrapper that calls
the three per-module helpers and concatenates the result column-wise:

```python
df_all = bundle.extract_all(signals, fs_hz=1000.0)   # (N, 11)
```

Equivalent to:

```python
df_all = pd.concat([
    bundle.extract_time_domain(signals),
    bundle.extract_frequency(signals, fs_hz=1000.0),
    bundle.extract_complexity(signals),
], axis=1)
```

### Single-trace caller

If you have just one trace and want to use the bundle helpers, wrap
it as `(1, T)` first:

```python
x = signals[0]                                       # (T,)
df_one = bundle.extract_all(x[np.newaxis], fs_hz=1000.0)  # (1, 11)
```

The bundle is batch-only by design — passing a 1D array raises
`ValueError` with the `[np.newaxis]` fix in the error message.

## Math constants vs policy values

A parameter on a feature function is one of two kinds:

- **Math constants** carry a *default* equal to the canonical value
  from the literature (`m=2` and `r_frac=0.2` for sample entropy
  from Pincus 1991, `n_bins=10` for Shannon entropy from Sturges'
  rule at T=512, `k_max=10` for Higuchi from the literature
  mid-range). Override if you have a numerical reason; otherwise
  ignore.
- **Policy values** have *no default*. They reflect a project-level
  choice rather than a math constant (`method="dvdt_max"` vs
  `"abs_peak"` for activation_position, `threshold_frac=0.3` for
  sec_peak_count, `binarize_method="median"` for
  lempel_ziv_complexity). Required at every direct-call site.

See [`../project/architecture.md`](../project/architecture.md)
"Defaults policy — math constants vs project-policy values" for the
rationale. The short version: the bundle is the one and only place
where project policy lives. Direct callers are forced to be explicit
so a different policy never leaks into the codebase by accident.

## Performance notes

Measured on N=1000 EGM-like traces at T=192 samples, 1 kHz
(a biphasic activation at a varying position plus noise):

| selection | per trace | per 10,000-trace bank |
|---|---|---|
| the native 11 (default) | 0.91 ms | 9.1 s |
| `catch22` (22) | 0.41 ms | 4.1 s |
| `catch24` (24) | 0.42 ms | 4.2 s |
| both, `egm_features+catch22` (33) | 1.50 ms | 15.0 s |
| a 3-feature study set | 0.06 ms | 0.6 s |

**Ask for what you need.** Every provider computes only the
requested features, so a three-feature selection costs about a
fifteenth of the default eleven rather than the same as all of
them. This is the main lever available:

```python
# Cheap: three features, no periodogram, no entropy.
df = bundle.extract_all(signals, features=["peak_to_peak", "trev", "entropy_pairs"])
```

**Which features actually cost anything**, per trace at T=192:
`shannon_entropy` 0.34 ms, the shared periodogram 0.14 ms,
`sample_entropy` 0.10 ms, `lempel_ziv_complexity` 0.06 ms, and the
remaining seven under 0.03 ms each.

> Earlier versions of this document said `sample_entropy` dominated.
> That was true at T=512, where its `O(T²)` term takes over — at
> T=192 it is about 14% of the total and `shannon_entropy` costs
> 3.5× more. If you work at longer trace lengths, expect the
> ordering to swing back.

The three frequency features share one periodogram, computed once
per trace and only when at least one of them is requested — so a
time-domain or complexity-only selection skips it entirely.

## Where to look next

- [`theory.md`](theory.md) — the math behind every feature, with
  worked examples on synthetic signals and parameter-choice
  justifications. Read this when reviewing a feature's output or
  designing a follow-up feature.
- [`../project/architecture.md`](../project/architecture.md) — design
  rationale: why pure functions, why the math-vs-policy split, why a
  pandas DataFrame.
- [`../project/roadmap.md`](../project/roadmap.md) — what's planned
  for v0.2.0+ (typed-contract output, Wittkampf-smoothed dV/dt,
  more features).
