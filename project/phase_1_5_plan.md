# egm-features — Phase 1.5 implementation plan

**Repo:** egm-features · **Phase:** 1.5
**Phase design doc:** `intracardiac-platform/phases/phase_1_5/design.md`
**Status:** planning · **Progress:** 0/8 steps done
**Repo estimate:** **9–18 h active** (Cx = **M / 3 pts**, cold-start by-analogy range — see
[Effort tracking](#effort-tracking))

---

## Scope — what this plan covers

Phase 1.5 assigns this repo exactly one §3 issue and no §4 backlog items.

| Phase item | What it needs from this repo | Steps |
|---|---|---|
| **FEA1** — catch22 feature functions (+ shared feature set) | All 22 catch22 features (plus the catch24 mean/SD pair) as named, typed, documented per-trace extractors, behind a **feature-provider seam** and shipped as an **optional extra**; a **feature-set registry** spanning the existing 11 *and* the new ones so a caller can request a named subset; batch extraction through `bundle`; `docs/theory.md` §4 written first. | S1–S8 |

**Not in scope, deliberately.** Design §4 demotes both of this repo's roadmap "Phase 1.5" items —
the Wittkampf-smoothed `activation_position` upgrade and the speculative extra features (wavelet /
Hilbert / template-correlation) — to **watch-triggers**: they land only if the STU5 comparison shows
`activation_position` variance on IAFDB, or surfaces a phenomenon none of the current features
capture. S8 re-files them in `roadmap.md` accordingly.

**Consumers waiting on this.** STU5 (sim↔IAFDB feature-distribution comparison — phase success
criterion #2) and STU4 (parameter estimator) both list FEA1 under *Depends on*. FEA1 itself depends
on nothing, and touches no schema, so it runs **in parallel with the Wave-1 migration** rather than
behind it.

## Design notes

Decisions taken before coding. Anything that outlives this phase moves to `project/architecture.md`
at S8; this list is the working record.

**D1 · Wrap `pycatch22`, don't reimplement.** *(Daniel, 2026-07-28; reaffirmed by project-lead.)*
Thin named wrappers over the official C implementation — the same pattern, and the same rationale,
as the existing antropy wrappers in `complexity.py` (see architecture.md "Why wrap antropy").
Reimplementing catch22 in numpy would trade a low, well-understood build risk for a **silent
correctness risk** in the one feature set whose entire job is detecting sim-vs-real differences;
pycatch22 is the canonical validated reference, and using it is the point of FB-6. Not revisited.

**D2 · Ship it as an optional extra, behind a provider seam.** *(project-lead, 2026-07-28 — revises
the earlier "hard dependency, fallback if it bites" position.)* Two hedges, both cheap now and
expensive to retrofit:

1. **`myocard-egm-features[catch22]`** — the base library stays clean numpy/scipy/pandas/antropy
   (matching this repo's charter line in `repo_charters.md`), and the compiler dependency travels
   only with the comparison-feature path.
2. **A `catch22` feature *provider*** (D3) — so if a toolchain-less target ever becomes real,
   swapping the implementation is a one-repo change, and wheels may exist by then.

The underlying concern is real but small: the project-lead built `pycatch22` from source on a clean
Linux box in **~5.5 s**, imported it, and ran all 24 features with no numpy dependency. `gcc` is
present in every environment we run — GitHub Actions `ubuntu-latest` ships build-essential, and
Daniel is on Ubuntu. The only place it would bite is publishing to non-Linux / toolchain-less
machines, which is deferred to post-Phase-2 anyway. **CI does not assume a wheel** — the runner
compiles it; the install docs state the build-tools requirement (S8).

**D3 · The provider seam.** New `providers.py` defining a small `FeatureProvider` Protocol —
`name`, `available()`, `feature_names`, `extract(signal) -> dict[str, float]` — with two
implementations: `NativeProvider` (the existing 11, always available) and `Catch22Provider`
(optional, lazily importing `pycatch22` and raising an actionable ImportError naming the extra).
Expressing the existing 11 as a provider too costs almost nothing and is what lets the registry
(S5) compose uniformly instead of special-casing. This is the seam the project-lead asked for: the
swap point is one class.

**D4 · Compute the full catch22 set, then select — never cherry-pick for performance.**
*(project-lead.)* `catch22_all` computes all 22/24 in a single C call, so the batch path always
computes the full set and slices our subset (skip self-affine, defer linear-autocorrelation) out of
the result. The 24 named per-feature wrappers (S4) still delegate to the individual
`pycatch22.<CODE>` functions — that is the correct and cheapest path for a genuine *single-feature*
call, and it is where the theory-doc anchors and docstrings live per this repo's "every feature is a
function" principle — but **no batch or selection path may assemble its output from them**. S6
enforces this.

**D5 · Maintain our own short-name mapping — do NOT use `pycatch22`'s `short_names`.** This is the
one real trap. `pycatch22.catch22_all(..., short_names=True)` returns a short-name list in which
**two labels are crossed relative to their hctsa codes**:

| pycatch22 short name | hctsa code it is paired with | What that code actually computes |
|---|---|---|
| `centroid_freq` | `SP_Summaries_welch_rect_area_5_1` | power fraction in the lowest 20% of frequencies |
| `low_freq_power` | `SP_Summaries_welch_rect_centroid` | median (centroid) frequency |

Confirmed empirically on 2026-07-28 with two known signals: a 3-cycle sine gives
`area_5_1 = 0.998`, `centroid = 0.037`; a 200-cycle sine gives `area_5_1 = 0.000`,
`centroid = 2.454`. So `area_5_1` is the low-frequency power and `centroid` is the median
frequency — the reverse of the labels pycatch22 hands back, and matching the Fulcher-lab table that
`catch22_techniques_explained.md` follows. Taking those labels at face value would silently compare
a *median frequency* on synthetic against a *power fraction* on IAFDB under one column name — a
wrong STU5 answer with no error. Therefore: **key everything off the hctsa code** (or the individual
`pycatch22.<CODE>` functions), keep our own `HCTSA_TO_NAME` mapping, and pin it with a test (S3).
Our names also diverge for `transition_variance` (pycatch22: `transition_matrix`) and `std_dev`
(pycatch22: `SD`), so we own the vocabulary regardless.

**D6 · `pycatch22` rejects numpy arrays.** Verified: passing an `NDArray` raises
`SystemError: … returned NULL without setting an exception`; the C bindings accept only a Python
`list` of floats. The provider converts with `.tolist()` at the boundary, so the library's public
`NDArray, shape (T,)` contract is unchanged. Cost is one 512-element list build per trace —
expected negligible next to `sample_entropy`'s O(T²), and S7 measures it rather than assuming.

**D7 · Degenerate input propagates `NaN`.** A constant 512-sample signal returns **NaN for 19 of
22** features. We propagate rather than substitute a sentinel: a silent `0.0` would be swallowed
into MMD / energy distances as a real coordinate and quietly bias STU4's realistic region, whereas a
NaN column is visible and droppable by the consumer. Documented in theory.md §4 and the docstrings;
S6 adds a bundle-level test that a degenerate trace yields NaN, not an exception.

**D8 · `extract_all`'s default output does not change.** `extract_all(signals, fs_hz=...)` with no
`features=` argument keeps returning the same **11 columns in the same order**. architecture.md
calls the DataFrame column names part of the public API; changing the default would break every
existing v1_baseline diagnostic and egm-studio call site for no gain — and with catch22 now an
optional extra (D2), a changed default would additionally break base installs. catch22 is opt-in via
`features=` or `extract_catch22`. This keeps the release a **minor** bump.

**D9 · catch24's `mean` / `std_dev` are registered but not in any default set.** They exist so a
caller can reproduce canonical catch24, but `egm-features` already carries amplitude information via
`peak_to_peak`, so the project's own sets don't need them (FB-6: "catch24 optional").

**D10 · The registry stops at name → provider → callable.** Selecting a set *from config*,
assembling the `(N, F)` matrix, and sharing one extractor across both sides of the comparison remain
**egm-studio's** `EgmFeaturesExtractor` per `parameter_estimator_design.md` §4.2. This repo ships the
functions and the vocabulary; it does not grow an orchestration layer.

**D11 · Version → 0.2.0.** New features + new optional dependency + new API surface, no breaking
change (D8). Note that architecture.md's "typed contracts migration path" also pencilled itself in at
v0.2.0 — that work depends on an `egm_features_bank` schema in egm-contracts which is **not** in
Phase 1.5's Wave-1 list, so it stays roadmapped and simply lands in a later version instead.

**D12 · `T = 192` samples, not 512 — and it costs us some catch22 stability.** CL-024 §4 settled the
trace length: `T ≡ 0 (mod 64)` for CLF3's MobileViT, so the 150–250 ms target rounds to **192 ms =
192 samples at 1 kHz**. Two consequences for FEA1:

1. **Every theory.md §4 worked anchor uses `T = 192 @ 1 kHz`**, and theory.md's Notation section
   (which still says "Default 512") is corrected at S1. Getting this wrong would put anchors in the
   doc that the tests then can't reproduce.
2. **The usable-now 14 was reasoned at `N ≈ 512`** (B.3 / `catch22_techniques_explained.md`); at 192
   the short-window caveat bites harder. I measured it rather than assuming — 400 traces per length
   from one generative process (biphasic Gaussian-derivative deflection at a random position + white
   noise, 1 kHz), comparing across-trace **SD** (CV is unusable here — several features are
   near-zero-mean, which inflates CV artificially; on SD, `mode_5`/`mode_10` actually *improve*):
   - **No NaN inflation** — 0.00 at both lengths on non-degenerate traces. D7's NaN concern is
     strictly about degenerate input, not short windows.
   - **6 of the 14 widen by >1.5×** at 192: `entropy_pairs` 3.2×, `transition_variance` 2.4×,
     `ami2` 2.1×, `high_fluctuation` 1.9×, `embedding_dist` 1.6×. Modest, and a wider spread is not
     automatically disqualifying — but it eats into the separation STU5 needs.
   - **`whiten_timescale` looks degenerate at both lengths**: across 400 traces it returned **one
     distinct value** at T=512 (0.1111 = 1/9, clearly quantised) and **four** at T=192. A
     near-constant coordinate contributes nothing to an MMD / energy distance. It is the first
     candidate to drop from the set.

   Caveat, stated plainly: this is a crude EGM *surrogate*, not simulated or real EGM, so it is
   **indicative only**. The §8.2 feature-responsiveness screening on actual banks is the instrument
   that decides the set — this just says what to look for. Raised to the project-lead as CL-047.

**Named sets shipped** (per `parameter_estimator_design.md` B.3) — membership **provisional** pending
the §8.2 screening (B.3 requires it; D12 gives it a shortlist to check first):

| Set name | Members | Needs the extra? |
|---|---|---|
| `egm_features` | the existing 11 | no |
| `catch22` | all 22 | yes |
| `catch24` | all 22 + `mean` + `std_dev` | yes |
| `catch22_usable_now` | the 14 — skips the self-affine pair (`rs_range`, `dfa`); defers the linear-autocorrelation family (`acf_timescale`, `acf_first_min`, `periodicity`, `ami_timescale`, `low_freq_power`, `centroid_freq`) to the multi-beat phase | yes |
| `egm_features+catch22` | 11 + 22 — the STU4 Decision-1 config | yes |

## Steps

Each step is one focused commit, ends green (`ruff format` / `ruff check` / `mypy` / `pytest`), and
states its verification. ☐ todo · 🔨 wip · ✅ done.

### S1 — `docs/theory.md` §4: catch22 ☐ (est. 2–4 h)

- **Change:** New §4 covering all 22 catch22 features plus the catch24 pair, organised by the eight
  Fulcher-lab families. Per feature: definition, the hctsa code it maps to, our short name, a worked
  numeric anchor, and a short-window reliability flag. Full derivations and EGM-relevance discussion
  are **cross-linked** to `intracardiac-platform/project/investigations/catch22_techniques_explained.md`
  rather than duplicated (Daniel, 2026-07-28). Also: update the table of contents; add the
  **z-scoring / amplitude-blindness** note to *Preprocessing assumptions* (catch22 z-scores
  internally, so these features cannot see peak-to-peak voltage — the reason they pair with our 11
  rather than replace them); renumber §4 References → §5. **Correct the Notation section's
  "Default 512" to `T = 192 @ 1 kHz`** and carry 192 through every §4 worked anchor (D12), and record
  the short-window reliability read per feature.
- **Verify:** every name the registry will ship has a §4 entry; every worked anchor is one S4/S6 will
  actually assert, computed at `T = 192`; no stale "512" left in the doc; links resolve.
- **Depends on:** none. *Theory-first — this lands before any catch22 code.*

### S2 — Optional extra + provider seam ☐ (est. 1.5–3 h)

- **Change:** `pyproject.toml` gains `[project.optional-dependencies] catch22 = ["pycatch22>=0.4.5"]`
  (base dependencies untouched) plus a mypy `ignore_missing_imports` override — no py.typed upstream.
  New `src/myocard_egm_features/providers.py`: the `FeatureProvider` Protocol, `NativeProvider`
  wrapping the existing 11, and `Catch22Provider` with a lazy import and an ImportError message that
  names the extra (`pip install "myocard-egm-features[catch22]"`) and the build-tools requirement.
- **Verify:** `NativeProvider.available()` is unconditionally true; `Catch22Provider.available()`
  reflects the import; the ImportError text is asserted (it is user-facing UX, not an internal
  detail).
- **Depends on:** S1.

### S3 — `catch22.py` batch primitive + name mapping ☐ (est. 1–2 h)

- **Change:** `HCTSA_TO_NAME` (D5) and `catch22_all(signal, *, catch24=False) -> dict[str, float]` —
  ndarray→list at the boundary (D6), keyed by **our** names via the hctsa codes, one C call for the
  full set (D4).
- **Verify:** a test asserting `HCTSA_TO_NAME` maps `SP_Summaries_welch_rect_area_5_1` →
  `low_freq_power` and `..._centroid` → `centroid_freq`, with the two-sine discriminating case from
  D5 as the evidence — this is the regression guard against pycatch22's crossed labels. Plus: 22/24
  key counts, a sine anchor, and constant-signal NaN behaviour (D7).
- **Depends on:** S2.

### S4 — The 24 named per-feature wrappers ☐ (est. 1–2 h)

- **Change:** One public function per feature in `catch22.py` (`mode_5`, `outlier_timing_pos`,
  `trev`, …), each delegating to the corresponding `pycatch22.<HCTSA_CODE>`, typed
  `NDArray -> float`, docstring carrying its theory.md §4 anchor in the house style. Single-feature
  convenience path only — per D4, nothing batch-shaped consumes these.
- **Verify:** each wrapper's output equals the matching entry from `catch22_all` on the same trace;
  the sine / noise anchors from §4 hold.
- **Depends on:** S3.

### S5 — Feature-set registry ☐ (est. 1–2 h)

- **Change:** New `src/myocard_egm_features/sets.py` — a `FEATURE_REGISTRY` mapping every feature
  name to its **provider** (D3) and the project-standard policy arguments the bundle already
  hardcodes, plus the five `FEATURE_SETS` presets from the table above and a
  `resolve(selection) -> list[str]` helper accepting a set name or an explicit name list.
- **Verify:** preset membership matches the table (notably `catch22_usable_now` has exactly 14 and
  excludes the 8 deferred names); unknown names raise with a helpful message; registry names are
  unique and every one has a theory.md entry; resolving a catch22 set without the extra installed
  raises the D2 ImportError, not a `KeyError`.
- **Depends on:** S4.

### S6 — Bundle integration: `extract_catch22` + `features=` ☐ (est. 1–2 h)

- **Change:** `bundle.extract_catch22(signals, *, catch24=False)` and a `features=` parameter on
  `extract_all`. When any catch22 feature is selected, call `catch22_all` **once per trace** and
  slice (D4). Column order follows theory.md section order (§1–§3 then §4); `features=None`
  preserves today's 11-column output exactly (D8).
- **Verify:** default output is byte-for-byte the current 11 columns; a selected subset returns
  exactly those columns in registry order; per-row values agree with the individual extractors; a
  degenerate trace yields NaN rather than raising (D7); a test asserts the batch path issues **one**
  `catch22_all` call per trace regardless of how many features were selected (the D4 guard).
- **Depends on:** S5.

### S7 — Base-install guard + CI + cost measurement ☐ (est. 1.5–3 h)

- **Change:** CI installs `.[dev,catch22]` for the main test matrix — **no wheel assumed**, the
  runner compiles from sdist — plus a **base-install job** that installs `.[dev]` only and runs a
  marked subset proving the package imports, the 11 features work, and requesting a catch22 set
  fails with the actionable message. Measure per-trace extraction cost for the 11, the 22, and the
  union on a realistic `(N=1000, T=512)` batch; record the numbers in `docs/usage.md` so STU4/STU5
  can size their sweeps, and confirm or refute D6's "negligible next to `sample_entropy`" claim with
  an actual figure.
- **Verify:** both CI jobs green; the base-install job genuinely lacks `pycatch22` (assert
  `Catch22Provider.available()` is false there); measured figures written down. No benchmark added
  as a CI gate — architecture.md keeps profiling component-internal.
- **Depends on:** S6.

### S8 — Docs + phase exit ☐ (est. 1–2 h)

- **Change:** `docs/usage.md` (new API + sets + the NaN policy + the measured costs),
  `README.md` (feature count, the `[catch22]` extra and its **build-tools requirement**),
  `project/architecture.md` (D1–D5 and D10 as durable rationale — the wrapper argument extended to
  pycatch22, the provider seam and why it exists, the crossed-short-names trap, the
  registry-vs-adapter boundary), `CHANGELOG.md` `[Unreleased]`, `roadmap.md` (drop the now-answered
  `features=[...]` open API question; re-file the Wittkampf upgrade + extra features as
  **watch-triggered**, per design §4), version → 0.2.0.
- **Verify:** the full pre-PR run in `intracardiac-platform/project/pr_checklist.md` passes.
- **Depends on:** all prior steps.

## Effort tracking

> Method: `intracardiac-platform/project/investigations/estimate_vs_actual_tracking.md`.
> Daniel speaks the markers (`start` / `switch` / `break` / `resume` / `stop`); this chat stamps the
> time from `date`. **Active = marked span − breaks.** Backstop: unmarked silence > **2 h** = away.
> Rolls up at cleanup into design §6 / §11 and `estimation_ledger.csv`.

**Complexity: M = 3 points — at the top of M.** Situated against the rubric's anchors: larger than
the S anchor (`noise_bank` `bank_id`, a single additive field), above the M anchor (CLF2 train-split
metrics), still short of the L anchor (SEP12's breaking `synthetic_bank` v2.0 migration, which
coordinates across a producer and a schema). Drivers: *change size* = new module + provider
abstraction + optional-dependency plumbing + additive API, but no restructure; *novelty* = low, we
wrap a reference implementation and the math is already written up by research; *surface* = one
repo, zero sibling deps, no wave coupling; *verification* = unit tests against known anchors plus a
second CI install path, no schema round-trip or GUI check. Two things push it to the M ceiling:
**documentation volume** (24 theory entries is most of S1) and the **two hedges** the project-lead
added. **Calibration note for cleanup:** if actuals land above ~15 h, re-score this **L** in the
ledger rather than treating it as an M overrun — the scope genuinely grew after the initial scoring.

**Estimate: 9–18 h active.** Cold start — `estimation_ledger.csv` is empty, so this is by analogy
with deliberately wide bounds, not `points × rate`. Task-type: `feature` (library + docs). This is up
from the pre-review 6–14 h; the delta is the optional extra, the provider seam, and the base-install
CI job (S2 + S7).

### Session log

| Timestamp (local) | Event | Focus (issue) | Note |
|---|---|---|---|
| | | | |

### Effort by issue

| Issue | Task-type | Estimate | Active | Elapsed | Sessions |
|---|---|---|---|---|---|
| FEA1 | feature (library + docs) | 9–18 h | | | |
| **Repo total** | | **9–18 h** | | | |

## Notes / decisions log

- **2026-07-28** — Plan drafted. Pre-coding probe of `pycatch22` 0.4.5 established the
  source-only distribution, D6 (rejects ndarray), D5 (**crossed `centroid_freq` / `low_freq_power`
  short names** — the significant find), and D7 (19/22 NaN on constant input). D5 in particular would
  have produced a silently wrong STU5 comparison had we trusted the library's own labels.
- **2026-07-29** — **CL-041 applied** (chore, ahead of S1): `ci.yml` ruff install pinned
  `>=0.6.0` → `==0.15.17` and `.pre-commit-config.yaml` `rev: v0.6.9` → `v0.15.17`, per CL-024 §1
  (which adopted this repo's CL-014 recommendation fleet-wide). Verified at the pinned version:
  `ruff check .` clean, `ruff format --check .` = 11 files already formatted. **No 0.16 reformat
  run** — the pin makes it moot. Not an FEA1 step; commits separately as `[Chore]`.
- **2026-07-29** — **CL-024 §4** resolved this repo's CL-015: synthetic + IAFDB must share a sample
  rate ("both-sides-or-neither"), now recorded in design §8.1. Same item set `T = 192 ms` — see
  **D12** for the FEA1 consequences (theory-doc anchors + a measured short-window stability check;
  raised back as CL-047).
- **2026-07-28** — Project-lead review of the source-only concern: **pycatch22 confirmed, do not
  reimplement**; measured a ~5.5 s clean build; added two hedges — the **`[catch22]` optional
  extra** and the **provider seam** (D2, D3) — and the **compute-full-set-then-select** rule (D4).
  Steps restructured 7 → 8 (S2 split out the extra + provider; S7 gained the base-install CI job);
  estimate 6–14 h → 9–18 h.
