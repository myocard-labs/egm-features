# egm-features — architecture and design rationale

Internal design doc for people building / maintaining the library. The
public surface is documented in `docs/usage.md`; the math behind every
feature is in `docs/theory.md`. **This doc explains the why** — module
shape, API choices, consumer responsibilities.

## Where the package sits

```
┌───────────────────────┐
│ myocard-egm-contracts │
│  (format schemas)     │
└───────────────────────┘
            ▲
            │ (egm-features may produce
            │  egm_features_bank conforming
            │  output in v0.2.0+ — currently
            │  returns a pandas DataFrame)
            │
┌───────────┴──────────┐
│  myocard-egm-features│  ◄── pure library
│   (this repo)        │      no I/O, no CLI, no torch
└──────────────────────┘      deps: numpy + scipy + pandas + antropy
            ▲
            │ consumed by
            │
┌───────────┴───────────────────────────────────────┐
│                                                   │
│  egm-studio              synthetic-egm-pipeline   │
│  ─ figures/recipes/      ─ feature-comparison    │
│    sim_realism_*         CLI for tuning loops    │
│  ─ gui/views/features    (during synthetic v2    │
│                            sim-realism work)     │
│                                                   │
│  notebook diagnostics    (any researcher script   │
│  (Phase 1.5 ad-hoc)       importing the library)  │
└───────────────────────────────────────────────────┘
```

**egm-features is *primitives only*.** It takes a 1D bipolar-EGM trace
+ a sample rate and returns scalar features. It does not:

- Read or write HDF5 banks (`egm-data` owns I/O against
  contracts-defined formats).
- Bandpass, calibrate, or resample (`egm-signal` owns shared DSP).
- Aggregate features across a bank (`bundle.extract_all` iterates over
  an `(N, T)` numpy array but doesn't load it; the caller does that
  via egm-data).
- Render plots (`egm-studio` does that — both the GUI features view
  and the `egm-figures` CLI recipes).
- Run analysis workflows like "compare feature distributions across
  synthetic + IAFDB banks" (see [Consumer responsibilities](#consumer-responsibilities)
  below).
- Ship any ML model code (`egm-classifier` doesn't currently consume
  egm-features but might in the future for diagnostic features
  alongside training; not load-bearing today).

The library is intentionally small and stable. Almost all of its
"interesting" content is the choice of *which features and which
parameters*, documented in `docs/theory.md`.

## Folder layout

```
src/myocard_egm_features/
├── __init__.py        ← package overview + __version__
├── time_domain.py     ← peak_to_peak, zero_crossings,
│                        activation_position, sec_peak_count
├── frequency.py       ← spectral_centroid, spectral_entropy,
│                        dominant_frequency (+ shared periodogram helper)
├── complexity.py      ← sample_entropy, shannon_entropy,
│                        lempel_ziv_complexity, higuchi_fractal_dimension
└── bundle.py          ← extract_all + per-module extract_<module> helpers
```

Four files, no subpackages, no `_helpers/` directory. The features
within each module are independent of each other; future growth means
new functions added to existing files, not a deeper hierarchy.

If a future feature spans modules (e.g., a "complexity feature based
on the dominant frequency"), it goes in whichever module's `import`
graph it leans on more heavily.

## Design decisions

### Why every feature is a function, even the trivial ones

`peak_to_peak` is one line of numpy: `signal.max() - signal.min()`.
`zero_crossings` is two lines. So why wrap them at all?

Three reasons, in order of importance:

1. **Uniform API for `bundle.extract_all`.** With every feature a
   function, `extract_all` is a clean loop over a list of `(name,
   callable)` pairs. Without wrappers, it has special-case branches
   for the trivial features interleaved with library calls for the
   complex ones. The bundle is the primary consumer of the per-feature
   functions; keeping a uniform API there is worth the four-line
   wrapper overhead.
2. **Discoverable entry point at the call site.** When a downstream
   consumer (egm-studio recipe, notebook diagnostic) writes
   `egm_features.peak_to_peak(trace)`, the reader knows exactly which
   function to read for the implementation choices. With inline
   `trace.max() - trace.min()` the reader has to deduce the choice
   (does it strip NaNs? exclude filter edge artifacts? use abs?).
   Wrappers are documentation by call site.
3. **Future-proofing the math.** v0.1.0 `peak_to_peak` is just
   `max - min`. v0.2.0 might want NaN handling, or to exclude the
   first/last 5% of samples to avoid filter edge artifacts, or to
   compute on `|x|` instead. One change in the wrapper; every consumer
   updates automatically on the next pin bump.

Same reasoning as `myocard-egm-signal.bandpass()` wrapping
`scipy.signal.butter + filtfilt` even though it's "just calling
scipy" — namespace + policy + future-proofing.

### Why wrap antropy for the complexity features

Three of the four complexity features (`sample_entropy`,
`lempel_ziv_complexity`, `higuchi_fractal_dimension`) are thin wrappers
around `antropy` functions. The wrapper body is often three lines —
compute a derived parameter (e.g. `r = r_frac * std`), call antropy,
return the scalar. So why wrap?

**The wrappers ARE the project's standardized parameter policy.** For
sample entropy specifically:

- `m = 2` (not 3, not 1)
- `r = 0.2 · σ(signal)` (not 0.15, not absolute 0.5 mV)
- Chebyshev metric (not Euclidean)
- `std(ddof=0)` (not `ddof=1`)

If egm-studio called `antropy.sample_entropy(trace, order=2, ...)` and
synthetic-egm-pipeline called it with `order=3`, we'd be comparing
apples to oranges but the DataFrame column names would look identical.
Cross-consumer drift would be silent until someone noticed that
"sample entropy on synthetic is 0.4 and on IAFDB is 0.8 — wait, were
those computed the same way?"

The wrapper makes the parameter choice **a single decision recorded in
this repo**, reviewed against the theory doc, and applied consistently
everywhere downstream. If we ever want to change `m = 2 → m = 3`, we
change it here and every consumer updates on the next pin bump.

Without the wrappers, every consumer has to:

1. Memorize / look up the project-standard parameters for every
   feature.
2. Pass them correctly to antropy (which uses `order` for embedding
   dim, not `m`; subtle calling-convention gotchas).
3. Update them in lockstep when we decide to change.

Same reasoning as why this repo wraps the trivial numpy features
above. The wrappers are the project's *policy* layer on top of the
canonical algorithm implementations.

### Why per-trace shape (not polymorphic over `(T,)` and `(N, T)`)

Each per-feature function takes a 1D trace `(T,)` and returns a
scalar. Batch processing over an `(N, T)` array is `bundle.extract_all`'s
job — not the individual extractors.

The alternative ("polymorphic shape — each function accepts `(T,)` or
`(N, T)`") would push shape-dependent branches into every feature.
The current design keeps the per-feature functions trivially typed
(`NDArray[shape (T,)] -> float`) and concentrates the batch logic in
one place. Cleaner per-feature code; cleaner type signatures.

### Why a pandas DataFrame return type for the bundle

`bundle.extract_all(signals, fs_hz)` returns a `pandas.DataFrame` with
N rows + 11 columns (one per feature). Named columns; joinable with
per-trace metadata (sim_id, patient_id, etc.) from the caller's bank
read; standard tooling for downstream filtering and plotting.

**Future migration path: typed contracts.** The
`egm-contracts` roadmap (v0.5.0+) includes an
`egm_features_bank` schema. When that schema lands, `bundle` grows a
parallel API that returns a typed Pydantic model. The pandas DataFrame
API stays for notebook ergonomics; the typed-bank API is for HDF5
round-trip via egm-data. Decision deferred to v0.2.0 of egm-features
to avoid bundling two design questions in one release.

### Why no CLI

egm-features is a library. Every consumer (egm-studio recipes,
synthetic-egm-pipeline tuning loops, notebooks) drives it
programmatically. No CLI is shipped.

The "what's a typical command-line invocation of feature extraction"
question is answered by the consumer:

- For a paper-figure-quality cross-bank comparison, run
  `egm-figures sim-realism-comparison` (egm-studio CLI; lands in
  Refactor Step 6).
- For an ad-hoc tuning loop during synthetic v2, run a notebook or a
  small driver script that imports egm-features.
- For per-trace feature columns added to a bank as a sibling
  HDF5, that would be a synthetic-egm-pipeline or egm-data CLI; not
  in scope for egm-features.

### Why these 11 features specifically

Source: the v1_baseline diagnostic (intracardiac-platform/project/v1_baseline_investigation.md)
plus Sánchez 2021's seven features. The 11 are:

- **4 time-domain:** peak_to_peak, zero_crossings, activation_position, sec_peak_count.
- **3 frequency-domain:** spectral_centroid, spectral_entropy, dominant_frequency.
- **4 complexity:** sample_entropy, shannon_entropy, lempel_ziv_complexity, higuchi_fractal_dimension.

Sánchez 2021's seven (peak-to-peak, duration, sample entropy, Shannon
entropy, spectral entropy, Kolmogorov complexity, fractal dimension)
are a subset of ours — we added `zero_crossings`,
`activation_position`, `sec_peak_count`, and `dominant_frequency`
during the v1_baseline diagnostic to capture phenomena Sánchez's seven
missed (fragmentation, frequency-domain peak structure). The
`kolmogorov_complexity` Sánchez named is implemented as
`lempel_ziv_complexity` here — see `docs/theory.md` §3.3 for why we
renamed it.

**Future feature additions** go through a design pass before landing:
each new feature needs a theory.md section before its implementation
ships, per [[feedback-theory-first-for-unfamiliar-domains]].

## Consumer responsibilities

This is the critical "where does the analysis live" point that comes
up the first time someone asks "great library, but where do I actually
USE this?" egm-features is *primitives only*; analysis workflows live
elsewhere.

| Workflow | Lives in | Pattern |
|---|---|---|
| Paper-quality figures comparing synthetic vs IAFDB feature distributions | **`egm-studio`** `figures/recipes/sim_realism_comparison.py` (lands in Refactor Step 6) | Recipe-driven, reproducible, journal-quality output via `egm-figures` CLI |
| GUI "features view" — per-trace feature values overlaid on the waveform display | **`egm-studio`** `gui/views/features.py` (lands in Refactor Step 6) | Interactive bank inspection alongside other GUI tabs |
| Synthetic v2 tuning loops (change a sim param, recompute features, see if distributions converge toward IAFDB) | **`synthetic-egm-pipeline`** as a `synthegm-feature-compare` CLI or notebook (Phase 1.5 work) | Producer-side iteration tool |
| Ad-hoc notebook exploration during Phase 1.5 | **`intracardiac-platform/examples/`** or local-only notebooks | Researcher-driven, not necessarily version-controlled at the analysis level |

The line between egm-studio and synthetic-egm-pipeline consumption:

- If the **output is a figure** that goes into a paper or report →
  egm-studio's `figures/recipes/`.
- If the **output is a tuning decision** that changes a sim YAML →
  synthetic-egm-pipeline's CLI.
- Both consume `egm-features` as a runtime library. **Neither piece of
  workflow code lives in this repo.**

Recording this here so the architecture stays intentional — when a
future contributor is tempted to add a `egm-features compare-banks`
CLI, the answer is "no, that lives in egm-studio."

## Testing strategy

Each per-feature function gets at least one test with a synthetic
signal where the answer is known by hand. The synthetic anchors are
the worked examples documented in `docs/theory.md` — so the theory doc
doubles as test-case specification.

Per-feature anchors (full list in the theory doc):

- `peak_to_peak`: pure sine of amplitude A → 2A.
- `zero_crossings`: sine of frequency f → ~2fT_seconds crossings.
- `activation_position`: biphasic pulse centered at sample k → k/(T-1).
- `sec_peak_count`: single biphasic pulse → 0; pulse + half-height
  secondary → 1.
- `spectral_centroid`: pure sine at f₀ → ~f₀; two-sine multi-power
  example → 130 Hz.
- `spectral_entropy`: white noise → ~1.0 normalized; pure sine → ~0.
- `dominant_frequency`: pure sine → its frequency; two-sine
  multi-power → higher-power peak.
- `sample_entropy`: pure sine → ~0; white noise → high (~2-3 for
  m=2, r=0.2σ).
- `shannon_entropy`: constant signal → 0; uniform-amplitude → log(n_bins).
- `lempel_ziv_complexity`: periodic binary → ~0 normalized; random
  binary → ~1.
- `higuchi_fractal_dimension`: perturbed line → ~1; white noise → ~2.

The bundle.extract_all is tested separately for output shape +
column-name + per-row consistency with the individual extractors. No
performance / benchmark tests in CI (those are component-internal
profiling work, not gating).

## Version coordination

egm-features has no internal dependencies (no egm-contracts, no
egm-data, no egm-signal). It can release independently. Downstream
consumers (egm-studio, synthetic-egm-pipeline) pin its version.

Schema-bump coordination is one-directional here:

- **egm-features → consumers:** if we change a feature's
  parameters or rename a column, downstream code may need updates.
  Semver discipline: parameter changes that alter feature values are
  treated as a minor (not patch) bump. The DataFrame column names are
  part of the public API; renaming is a minor bump.
- **egm-contracts → egm-features:** the v0.2.0+ typed-contract path
  ties egm-features to a future `egm_features_bank` schema in
  egm-contracts. When that lands, egm-features bumps its
  egm-contracts pin like the other consumer-side repos do; the
  schema-cascade order applies (egm-contracts ships first → egm-data
  updates → egm-features updates → downstream consumers update).

See [[project-egm-features-scope]] for the cross-component context.
