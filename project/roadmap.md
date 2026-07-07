# egm-features — roadmap

What's planned for future releases. Internal doc — public users see
the README and `docs/usage.md`. Items scheduled into cross-cutting
Phase work in the meta repo's `project_plan.md` carry a
`→ tracked at intracardiac-platform Phase X` annotation; the rest are
component-internal — driven by what consumers actually need.

## v0.1.0 — current release (target for Block 5)

Scope (recap; see `project/architecture.md` for the design rationale):

- **Three feature modules** with 11 functions total:
  - `time_domain.py` — `peak_to_peak`, `zero_crossings`,
    `activation_position` (dV/dt method), `sec_peak_count`.
  - `frequency.py` — `spectral_centroid`, `spectral_entropy`,
    `dominant_frequency`; shared `_periodogram` helper.
  - `complexity.py` — `sample_entropy`, `shannon_entropy`,
    `lempel_ziv_complexity` (LZ76 with median binarization),
    `higuchi_fractal_dimension`.
- **Bundle helper** — `bundle.extract_all(signals, fs_hz)` returns a
  pandas DataFrame with N rows + 11 columns; per-module
  `extract_time_domain` / `extract_frequency` / `extract_complexity`
  helpers for partial bundles.
- **Tests** — synthetic-signal anchors per the worked examples in
  `docs/theory.md` (pure sine, white noise, biphasic pulse, periodic
  binary, etc.).
- **Theory doc** — `docs/theory.md` covering math + parameter
  rationale for every feature. Written *before* the implementation per
  [[feedback-theory-first-for-unfamiliar-domains]].

Dependencies (lean): `numpy>=1.26, scipy>=1.11, pandas>=2.0, antropy>=0.1.6`.
No torch, no h5py, no internal myocard- deps.

## v0.2.0+ — concrete next steps

These are sized for "could land in one focused PR each." Order is
suggestive; pick by what unblocks the next consumer need.

### Typed-contract bundle output (`egm_features_bank` Pydantic model)

The v0.1.0 `bundle.extract_all` returns a pandas DataFrame.
egm-contracts has an `egm_features_bank` schema roadmapped (see
egm-contracts `project/roadmap.md` v0.5.0+). When that schema ships,
add a parallel API:

```python
def extract_all_typed(signals, fs_hz) -> FeaturesBank: ...
```

where `FeaturesBank` is the codegen'd Pydantic model from
egm-contracts. Keeps the existing DataFrame API for notebook
ergonomics; the typed API is for HDF5 round-trip via egm-data and
for any consumer that wants schema-validated output.

> → Tracked at `intracardiac-platform/project/refactor_checklist.md`
> Step 4 follow-up; depends on the egm-contracts schema landing first.
> Cascade order: contracts → data → features.

### `activation_position` upgrade — Wittkampf-smoothed dV/dt

`docs/theory.md` §1.3 Limitations notes that the discrete
first-difference algorithm is noise-sensitive within the 30–250 Hz
pass band. The cheapest upgrade is to low-pass at ~200 Hz before
taking the difference (the Wittkampf-smoothed convention). Adds a
parameter:

```python
def activation_position(
    signal,
    /, *,
    method="dvdt_smoothed",        # new default
    smoothing_cutoff_hz=200.0,     # new param
)
```

with `method="dvdt_raw"` reserved for the v0.1.0 behavior.

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 1.5
> (sim-realism comparison work). Trigger: if the sim-realism feature
> distribution comparison shows excessive `activation_position`
> variance on IAFDB, that's the signal we need the smoothed variant.

### Additional features (driven by Phase 1.5 needs)

The v1_baseline 11 is the starting set. As the sim-realism work
proceeds we may want more — the synthetic-vs-IAFDB distribution
comparison may surface specific phenomena that none of the 11 capture.
Plausible additions:

- **Wavelet-based features.** Energy ratios across wavelet
  scales — captures multi-scale structure that PSD averages away.
- **Hilbert envelope features.** Instantaneous-amplitude envelope
  features — peak amplitude, envelope decay rate.
- **Activation-template correlation.** Match each trace against a
  known clean-activation template; correlation height + position is
  a feature.

Each addition follows the theory-first rule: add to `docs/theory.md`
first with the math + worked example + parameter rationale, then
implement.

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 1.5
> (additional Phase 1.5 work surfaces specific needs).

### Performance optimization (if needed)

`sample_entropy` is O(T²) in the trace length T. For T=512 the
absolute cost is small (~ms per trace); for an N=10000 trace bank
it's ~5–10 seconds total. Not currently a bottleneck.

`higuchi_fractal_dimension` is O(k_max · T) — also fine at our
scale.

If a future Phase 4 / Phase 7 (longer traces, larger banks) makes
extraction slow enough to be a real friction in iteration loops,
options:

- `numba` JIT for the inner loops of the wrappers we wrote ourselves
  (`shannon_entropy`).
- `joblib` parallel iteration in `bundle.extract_all` (currently a
  serial loop).
- C-accelerated antropy variants if they ship.

Not on the v0.2.0 critical path — wait for evidence of friction.

> → Component-internal. No phase home until friction shows up.

### Cross-project code placement audit candidates

None identified for v0.1.0. The egm-features library is freshly built
specifically for this scope; no drift to clean up.

Adjacent question for the **Refactor Step 8 code-placement audit**:
is there any code in `egm-classifier`'s standalone diagnostic
notebooks (from v1_baseline) that should move here? The chat-based
feature analysis from the v1_baseline diagnostic was the source
material for this library, but the *plotting + comparison code* that
went around it lived in the diagnostic chat itself. If any of that
plotting code survives somewhere recoverable, it might belong in
egm-studio (per [Consumer responsibilities](architecture.md#consumer-responsibilities))
rather than here.

> → Tracked at `intracardiac-platform/project/refactor_checklist.md`
> Step 8 (cross-project code-placement audit).

## Schema bumps to coordinate — see egm-contracts roadmap

egm-features doesn't ship a schema today (the DataFrame bundle output
is informal). When the typed-contract bundle output lands (above), the
schema-cascade story applies in the usual order
(egm-contracts → egm-data → egm-features). Cross-reference at
`egm-contracts/project/roadmap.md` "Schema bumps to coordinate."

## Won't-do (out of scope, but documented to save the question)

- **No HDF5 / CSV / JSON I/O inside this repo.** Bank reading is
  egm-data's job. The bundle returns a pandas DataFrame; the caller
  decides what to do with it.
- **No DSP primitives inside this repo.** Bandpass, calibration,
  resampling all live in `myocard-egm-signal`. egm-features operates
  on already-bandpassed, already-calibrated numpy arrays.
- **No torch dependency.** No model code, no nn.Module. Stays at the
  numpy + scipy + pandas + antropy level.
- **No CLI.** Library only — drive it from notebooks, from
  egm-studio's `egm-studio-render` recipes, or from
  synthetic-egm-pipeline's tuning loops. See
  [Consumer responsibilities](architecture.md#consumer-responsibilities).
- **No bank-aware aggregation.** Group-by / median-per-patient /
  distribution-difference statistics are not in scope — those are
  caller responsibilities (pandas + scipy.stats handle them well in
  notebook code; egm-studio recipes will wrap them for paper figures).
- **No multi-trace features** (cross-pair correlation, spatial
  fragmentation indices). Per-trace only. Multi-pair features would
  need an entirely different shape contract; defer until a real
  consumer asks.

## Open architectural questions for later

These don't need decisions for v0.1.0 but are worth thinking about as
the next consumer arrives:

- **Should `bundle.extract_all` grow a `features=[...]` selection
  parameter** to compute only a subset? Today it computes all 11.
  Useful for performance-sensitive callers that only need a few
  features. Decide when a consumer asks.
- **Should the per-module bundle helpers
  (`extract_time_domain`, etc.) be exposed in the top-level
  `__init__.py`** as `extract_time_domain` rather than
  `bundle.extract_time_domain`? Lower call-site verbosity but more
  surface area in the top-level namespace.
- **Should features that need `fs_hz` (frequency-domain ones) share a
  `FsAware` Protocol** with egm-signal's similar dependency? Currently
  each function takes `fs_hz` as a positional arg. A shared Protocol
  would unify the type signature but adds a cross-repo dependency for
  marginal benefit.
- **Should we add a typed `FeatureResult` dataclass** instead of
  returning bare floats? Would carry units (Hz vs mV vs dimensionless)
  and confidence intervals where applicable. Speculative until a
  consumer needs the metadata.
