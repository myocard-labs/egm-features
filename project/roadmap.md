# egm-features — roadmap

Future work only — shipped history lives in [`CHANGELOG.md`](../CHANGELOG.md). Internal
doc; public users read the README + `docs/usage.md`.

Work lands here as it's identified, sits in the **Backlog** until a phase-planning session
promotes it into a **Phase** cluster, then moves to the CHANGELOG once shipped. Phase
clusters mirror the science Project Phases in
`intracardiac-platform/project/project_plan.md` and stay living lists until each phase's
planning session finalizes them. Items scheduled into cross-cutting Phase work carry a
`→ tracked at intracardiac-platform Phase X` annotation; the rest are component-internal.

## Phase 1.5 — sim-realism

### `activation_position` upgrade — Wittkampf-smoothed dV/dt

`docs/theory.md` §1.3 Limitations notes that the discrete first-difference algorithm is
noise-sensitive within the 30–250 Hz pass band. The cheapest upgrade is to low-pass at
~200 Hz before taking the difference (the Wittkampf-smoothed convention). Adds a
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

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 1.5. Trigger: if the
> sim-realism feature-distribution comparison shows excessive `activation_position`
> variance on IAFDB, that's the signal we need the smoothed variant.

### Additional features (driven by Phase 1.5 needs)

The v1_baseline 11 is the starting set. As the sim-realism work proceeds we may want more
— the synthetic-vs-IAFDB distribution comparison may surface phenomena none of the 11
capture. Plausible additions: **wavelet-based** energy ratios across scales;
**Hilbert-envelope** features (peak amplitude, envelope decay rate); **activation-template
correlation** (match each trace against a known clean-activation template — correlation
height + position). Each follows the theory-first rule: add to `docs/theory.md` first
(math + worked example + parameter rationale), then implement.

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 1.5.

## Backlog (unscheduled — promoted into a phase at a planning session)

### Typed-contract bundle output (`egm_features_bank` Pydantic model)

`bundle.extract_all` returns a pandas DataFrame today. egm-contracts has an
`egm_features_bank` schema roadmapped; when it ships, add a parallel API:

```python
def extract_all_typed(signals, fs_hz) -> FeaturesBank: ...
```

where `FeaturesBank` is the codegen'd Pydantic model from egm-contracts. Keep the
DataFrame API for notebook ergonomics; the typed API is for HDF5 round-trip via egm-data
and any consumer that wants schema-validated output.

> → Depends on the egm-contracts schema landing first; cascade order
> contracts → data → features. See `egm-contracts/project/roadmap.md` "Schema bumps to
> coordinate" and `intracardiac-platform/project/refactor_checklist.md` Step 4 follow-up.

### Performance optimization (if extraction becomes a friction)

`sample_entropy` is O(T²) and `higuchi_fractal_dimension` is O(k_max·T) — both fine at
current scale (T=512, ~ms/trace; ~5–10 s for an N=10000 bank). If a future Phase 4 / 7
(longer traces, larger banks) makes extraction slow enough to be a real iteration
friction: `numba` JIT for our hand-written inner loops (`shannon_entropy`), `joblib`
parallel iteration in `bundle.extract_all` (currently a serial loop), or C-accelerated
antropy variants if they ship. Wait for evidence of friction.

> → Component-internal; no phase home until friction shows up.

### Open API questions

Worth thinking about as the next consumer arrives; none need a decision yet:

- A `features=[...]` selection parameter on `extract_all` to compute only a subset (today
  it computes all 11) — useful for performance-sensitive callers.
- Exposing the per-module helpers (`extract_time_domain`, etc.) at the top-level
  `__init__.py` vs. under `bundle.` — lower call-site verbosity vs. more namespace surface.
- A shared `FsAware` Protocol with egm-signal for the `fs_hz`-dependent functions —
  unifies the signature but adds a cross-repo dependency for marginal benefit.
- A typed `FeatureResult` dataclass carrying units (Hz / mV / dimensionless) + confidence
  intervals instead of bare floats — speculative until a consumer needs the metadata.

## Known issues

None open. (`activation_position`'s dV/dt noise sensitivity in the 30–250 Hz band is a
known *limitation*, not a bug — it's addressed by the Phase 1.5 Wittkampf-smoothing
upgrade above.)

## Won't-do (out of scope, but documented to save the question)

- **No HDF5 / CSV / JSON I/O inside this repo.** Bank reading is egm-data's job; the bundle
  returns a pandas DataFrame and the caller decides what to do with it.
- **No DSP primitives.** Bandpass, calibration, resampling all live in
  `myocard-egm-signal`; egm-features operates on already-bandpassed, already-calibrated
  numpy arrays.
- **No torch dependency.** No model code, no `nn.Module` — stays at the numpy + scipy +
  pandas + antropy level.
- **No CLI.** Library only — drive it from notebooks, egm-studio's `egm-studio-render`
  recipes, or synthetic-egm-pipeline's tuning loops. See
  [Consumer responsibilities](architecture.md#consumer-responsibilities).
- **No bank-aware aggregation.** Group-by / median-per-patient / distribution-difference
  statistics are caller responsibilities (pandas + scipy.stats handle them well in
  notebook code; egm-studio recipes wrap them for paper figures).
- **No multi-trace features** (cross-pair correlation, spatial fragmentation indices).
  Per-trace only — multi-pair features need an entirely different shape contract; defer
  until a real consumer asks.
