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

**This repo's assigned Phase 1.5 scope is one item: FEA1** — the catch22 feature functions
plus the shared feature-set registry. It is *active*, so it lives in
[`phase_1_5_plan.md`](phase_1_5_plan.md) (steps, estimate, design decisions), not here —
this roadmap stays future-only. FEA1 lands in `CHANGELOG.md` when it ships and the plan is
deleted at phase cleanup.

Nothing else in this repo is scheduled for Phase 1.5. The two items below were previously
filed here as Phase 1.5 work; the phase design doc's §4 pass reclassified both as
**watch-triggers** — deferred, pulled in only if the named trigger fires.

### Watch-triggered — not scheduled

> Per `intracardiac-platform/phases/phase_1_5/design.md` §4 ("Deferred — stay in each repo's
> `roadmap.md`; revisit only on trigger"). Neither is planned work; each is a standing
> response to a specific observation.

#### `activation_position` upgrade — Wittkampf-smoothed dV/dt

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

> **Trigger:** the STU5 sim↔IAFDB feature-distribution comparison shows excessive
> `activation_position` variance on IAFDB. Until then, not planned work.
> Listed in design §4's deferred watch-trigger set.

#### Additional features (if the comparison surfaces a gap)

The v1_baseline 11 plus the FEA1 catch22 set is the working comparison space. If the
synthetic-vs-IAFDB comparison surfaces a phenomenon none of them capture, plausible
additions are: **wavelet-based** energy ratios across scales; **Hilbert-envelope** features
(peak amplitude, envelope decay rate); **activation-template correlation** (match each trace
against a known clean-activation template — correlation height + position). Each follows the
theory-first rule: add to `docs/theory.md` first (math + worked example + parameter
rationale), then implement.

> **Trigger:** STU5 / the §8.2 feature-responsiveness screening identifies a phenomenon the
> current feature set misses. Note catch22 (FEA1) already widens the space considerably —
> autocorrelation timescales, symbolic dynamics, forecasting error, extreme-event timing —
> so this is a smaller gap than it was when first written.

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
>
> **Confirmed out of Phase 1.5** (2026-07-28): `egm_features_bank` is not in the phase's
> Wave-1 contract bump (design §5), so the prerequisite doesn't land this phase. Note
> `project/architecture.md` pencils this work in "at v0.2.0" — v0.2.0 is now FEA1's release,
> so this simply lands in a later version. Architecture.md is corrected at FEA1's S8.

### Performance optimization (if extraction becomes a friction)

`sample_entropy` is O(T²) and `higuchi_fractal_dimension` is O(k_max·T) — both fine at
current scale (T=512, ~ms/trace; ~5–10 s for an N=10000 bank). If a future Phase 4 / 7
(longer traces, larger banks) makes extraction slow enough to be a real iteration
friction: `numba` JIT for our hand-written inner loops (`shannon_entropy`), `joblib`
parallel iteration in `bundle.extract_all` (currently a serial loop), or C-accelerated
antropy variants if they ship. Wait for evidence of friction.

> → Component-internal; no phase home until friction shows up. Also named in design §4's
> deferred watch-trigger set ("feature/studio perf accel"), so the phase agrees: evidence
> first. FEA1 measures per-trace extraction cost at its S7 — that measurement is what would
> trip this trigger, if anything does.

### Open API questions

Worth thinking about as the next consumer arrives; none need a decision yet:

- Exposing the per-module helpers (`extract_time_domain`, etc.) at the top-level
  `__init__.py` vs. under `bundle.` — lower call-site verbosity vs. more namespace surface.
- A shared `FsAware` Protocol with egm-signal for the `fs_hz`-dependent functions —
  unifies the signature but adds a cross-repo dependency for marginal benefit.
- A typed `FeatureResult` dataclass carrying units (Hz / mV / dimensionless) + confidence
  intervals instead of bare floats — speculative until a consumer needs the metadata.

> **Answered, in progress:** the `features=[...]` selection parameter on `extract_all` was
> the first question on this list. FEA1 answers it — a feature-set registry spanning the 11
> and the catch22 set, with named presets and a `features=` selector (plan S5–S6). Removed
> from the open list; it moves to `CHANGELOG.md` when it ships.

## Known issues

None open. (`activation_position`'s dV/dt noise sensitivity in the 30–250 Hz band is a
known *limitation*, not a bug — the Wittkampf-smoothing upgrade above would address it, but
that upgrade is **watch-triggered, not scheduled**: it lands only if the STU5 comparison
shows the variance actually bites.)

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
