# egm-features — Phase 1.5 implementation plan

**Repo:** egm-features · **Phase:** 1.5
**Phase design doc:** `intracardiac-platform/phases/phase_1_5/design.md`
**Status:** implementation done · **Progress:** 9/9 steps done (S0–S8 ✅)
**Wave:** FEA1 is **Wave 1**, position 9 of 10 — *next* after iafdb's B22/S0 — and the Wave-1 gate
requires this repo **tagged**, so S8 ends in a v0.2.0 release, not just a merge.
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

**D4 · Compute only the requested catch22 features.** *(Revised 2026-08-06, Daniel — supersedes the
project-lead's compute-all-then-filter rule.)* Every path computes exactly what was asked for, via
`pycatch22`'s **individual** per-feature entry points.

> **What the original rule said, and why it no longer holds.** The project-lead's D4 was "never
> cherry-pick for performance — `catch22_all` computes all 22/24 in one C call, so compute the full
> set and slice." The premise was that the bundle is cheaper than its parts. **Measured at
> `T = 192`, it is not:** the bundled call costs **0.373 ms** and calling all 22 individually costs
> **0.368 ms** — parity, because `catch22_all` is simply a loop over the same functions. There is no
> bulk discount to preserve, so cost scales with how many features you ask for:
>
> | features | individual | vs bundled |
> |---|---|---|
> | 1 | 0.003 ms | 0.01× |
> | 3 | 0.016 ms | 0.04× |
> | 6 | 0.086 ms | 0.23× |
> | **14** (a typical study subset) | 0.164 ms | **0.44×** |
> | 22 | 0.368 ms | 0.99× |
>
> End-to-end through `Catch22Provider`: all 24 = 0.424 ms, a 14-feature subset = 0.229 ms (1.9×
> cheaper), one feature = 0.012 ms (**35×** cheaper). This matters most exactly where Daniel
> expected — if the §8.2 screening finds only a handful of catch22 features carry signal for the
> parameter-estimation study, every sweep afterwards pays a fraction of the full cost.

**Equivalence is asserted, not assumed.** Values from the individual functions are identical to the
bundled call — checked across all six anchor signals including the degenerate ones where 19 of 22
return `NaN`, since a subtle divergence would be easiest to miss there. That test is what makes it
safe to skip upstream's own entry point.

`catch22_features(signal, names)` is therefore the only extraction path; `catch22_all` is a thin
convenience wrapper asking for everything. Raised to the project-lead as **CL-141**, since D4 was
their rule.

**D5 · Key off the hctsa code, and maintain our own code → name mapping.**
*(Corrected 2026-08-06 — see the retraction below.)* We use short names as column labels but resolve
them from the stable hctsa codes, not from `pycatch22`'s `short_names` list. Three reasons, none
dramatic: **two names we choose differently** (`transition_variance` rather than upstream's
`transition_matrix`, since the feature is the summed column variance *of* that matrix; and
`std_dev` rather than `SD`); **documentation order**, since `pycatch22` returns features in a
different order than theory.md §4 presents them; and **a stable key**, because an hctsa code changes
only when the feature does, whereas a convenience label could be renamed upstream and silently
re-point a column. A test asserts every code upstream returns is one we map, so a feature-set change
fails loudly.

> **⚠ Retraction (2026-08-06).** D5 previously asserted that `pycatch22`'s `short_names` **crossed**
> `centroid_freq` and `low_freq_power` relative to their hctsa codes, and called it "the one real
> trap". **That was wrong.** Upstream's list is correct in both `catch24` modes; verified by
> re-running the probe and by measurement (a 50 Hz sine gives `..._centroid = 0.3191` against an
> exact 0.3142 rad/sample, and upstream labels that `centroid_freq` — correctly). The error was a
> **transcription mistake reading my own probe output**, which then propagated into theory.md
> §4.1.2, this plan, the S1/S2 commit messages, and coordination-log entries CL-014 and CL-046.
> All have been corrected; the fleet correction is **CL-140**.
>
> **What was never wrong:** the mapping actually implemented. Our names agree with upstream on 22 of
> 24, differing only in the two deliberate choices above, and every value the library returns was
> and is correct. The bug was in the justification, not the code — which is precisely why the
> replacement test verifies the mapping **against physics** rather than against a claim.

**D6 · `pycatch22` rejects numpy arrays.** Verified: passing an `NDArray` raises
`SystemError: … returned NULL without setting an exception`; the C bindings accept only a Python
`list` of floats. The provider converts with `.tolist()` at the boundary, so the library's public
`NDArray, shape (T,)` contract is unchanged. Cost is one 512-element list build per trace —
expected negligible next to `sample_entropy`'s O(T²), and S7 measures it rather than assuming.

**D7a · `NaN` must be *warned about*, not just propagated.** *(Daniel, review of S1.)* Propagating
silently is its own failure mode: a 10,000-trace extraction can run for minutes, and discovering
afterwards that a channel produced all-`NaN` wastes the whole run. So the batch path emits **one
aggregated `RuntimeWarning` per call** — naming how many traces and which features were affected —
rather than one per trace, which would drown the output and slow the loop. The `NaN` values still
flow through unchanged; the warning is diagnostic only. Implemented in S3 (primitive) + S6 (batch
aggregation), tested in both. Documented in theory.md §4.1.4.

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

**D13 · The caller chooses which shipped features get computed.** *(Daniel, 2026-08-06, after
reviewing S2.)* `features=` on `FeatureProvider.extract`: a provider must not compute what nobody
asked for. `features=None` keeps meaning "all of them", so every existing call is unchanged.

This is a **performance contract**, and the right way to honour it differs per provider — the
native eleven are independent calls, so skipping is a real saving; catch22 gets all 22 from one C
call, so the cheapest way to serve a subset is compute-all-then-filter (D4 unchanged). Each
provider decides; callers just ask.

Measured at `T = 192`: all eleven native features cost **0.68 ms/trace**, of which
`shannon_entropy` is 0.34 and the shared periodogram 0.14. Requesting one cheap feature is now
**~154×** cheaper, and a time-domain-only selection skips the periodogram entirely.

**Corrects a documented assumption:** `roadmap.md` has said since v0.1.0 that `sample_entropy` is
the extraction bottleneck. True at `T = 512`, where its O(T²) dominates — but at 192 it is 14% of
the cost and `shannon_entropy` is 3.5× more expensive. Fixed at S8.

> **Withdrawn, same day.** A first pass also built a `UserDefinedProvider` accepting *user-supplied
> callables*, on a misreading of "user-defined feature set" — Daniel meant the user selects among
> the features we already ship, which is what `features=` does. The plugin mechanism was removed
> before it shipped, along with the reproducibility problem it created (a callable with no
> theory.md entry, no test anchor, and no recorded policy makes an STU5 result that cannot be
> regenerated from the repos). Escalated as CL-138, **withdrawn as CL-139**; no §3 or §9 change is
> needed. Recorded because the reasoning is worth not re-deriving: **every feature this library can
> compute has a documented definition and a test anchor, and that is a property worth keeping.**

**D14 · No study-specific feature sets in the library.** *(Daniel, 2026-08-06, reviewing S5.)*
S5 originally shipped a `catch22_usable_now` set — the 14 expected to behave at `T = 192`, per
theory.md §4.1.3. **Removed.** That is a curated judgement for one study at one window length, and
this repo's charter is arrays in, feature values out. A consumer working at `T = 512` or on
multi-beat windows would inherit a set named for their situation and wrong for it. It also breaks
the standing convention that **libraries ship no policy defaults** — those live in schemas or the
executable consumer's config — which I should have caught when writing it.

**Per-feature `min_length` was considered and rejected**, not merely deferred: we have no
defensible thresholds. `dfa` does not work at 192 samples and fail at 191; it degrades
continuously, and §4.1.3's read is a judgement, not a number. Encoding one would manufacture
precision the literature does not give us, and would look authoritative while being invented.

**What replaces it:** nothing in the library. `FEATURE_SETS` keeps only structural groupings
(`egm_features`, `egm_features+catch22`) and canonical ones (`catch22`, `catch24`); a study set is
one call — `resolve(["peak_to_peak", "trev", "entropy_pairs"])`. The reliability *knowledge* stays
in theory.md §4.1.3, which is where it belongs: the library documents what is known about each
feature, the consumer decides which to use. **egm-studio consequence:** B.3's "catch22 subset fixed
in code" is now fixed in *their* code. Raised as **CL-142**.

**Backlogged instead:** a coarse short-trace warning (roadmap.md) — `pycatch22` returns numbers for
a 5-sample input, and nothing detects it. Scoped as an absurdity guard, not per-feature minimums.

**Named sets shipped** — **structural and canonical groupings only**, per D14. No study-specific
curation; the §8.2 screening's outcome is the consumer's to encode, not ours:

| Set name | Members | Needs the extra? |
|---|---|---|
| `egm_features` | the existing 11 | no |
| `catch22` | all 22 | yes |
| `catch24` | all 22 + `mean` + `std_dev` | yes |
| `egm_features+catch22` | 11 + 22 — the STU4 Decision-1 config | yes |

## Steps

Each step is one focused commit, ends green (`ruff format` / `ruff check` / `mypy` / `pytest`), and
states its verification. ☐ todo · 🔨 wip · ✅ done.

### S0 — Clear the coordination-log inbox ✅ (est. 0.5 h)

- **Change:** four one-liners the design doc asks be cleared in this session, plus CL-117 (cc'd):
  - **CL-118** — `numpy>=1.26` → `numpy>=1.26,<2.5`, with a comment naming the reason (numpy 2.5's
    PEP-695 stubs are unparseable at `python_version = "3.10"`) and the expiry condition.
  - **CL-085** — pin dev tools exactly: `ruff==0.15.17`, `mypy==2.1.0` (were `>=0.6.0` / `>=1.10`;
    egm-features was one of the last two repos still unbounded).
  - **CL-117** — `__version__` derived from `importlib.metadata`, not the hardcoded `"0.1.1"`.
  - **CL-099** — root-anchor the eight output-dir `.gitignore` patterns (`/data/`, `/banks/`, …).
- **Verify:** ✅ `git check-ignore` both directions — `src/myocard_egm_features/data/x.py` and
  `tests/data/z.py` are **not** ignored, all eight top-level output dirs still are; `__version__`
  resolves from installed metadata; `ruff check`/`format` clean; 76 tests pass.
- **Depends on:** none. Config only, no `src/` behaviour change → separate `[Chore]` commit.

> **mypy caveat (recorded, not a failure).** A clean `pip install -e ".[dev]"` here yields **12
> pre-existing errors** in `tests/` — 10 `type-arg` "missing type arguments for generic type
> `ndarray`" plus two shape-assignment complaints. They reproduce identically on a **pristine HEAD
> clone**, so S0 did not cause them, and they are the known sandbox artifact: this sandbox runs
> **Python 3.10**, where `numpy<2.5` resolves to **2.2.6** (2.3+ dropped 3.10), whose stubs lack the
> PEP-696 TypeVar defaults that make a bare `np.ndarray` legal. CI runs **3.12 → numpy 2.4.6**, which
> has them — the resolve CL-118 says is clean across the other six repos. numpy 2.4.6 **cannot be
> installed here at all** ("No matching distribution"), so this is unverifiable in-sandbox by
> construction; **CI is the authority**, and I'll read the lint/type job on push rather than claim
> local mypy cleanliness.

### S1 — `docs/theory.md` §4: catch22 ✅ (est. 2–4 h)

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
- **Verify:** ✅ all 24 short names **and** all 24 hctsa codes appear in §4; **64 worked anchors
  machine-checked against a fresh `pycatch22` run — 0 mismatches**; the 19-of-22 NaN claim confirmed;
  no broken in-page anchors; ruff clean; 76 tests pass. The anchors are now the spec S4/S6 assert.
- **Depends on:** none. *Theory-first — this lands before any catch22 code.*
- **Surfaced (not fixed here):** two §1–§3 parameter choices were justified at `T = 512` and are now
  approximate at 192 — `shannon_entropy`'s `n_bins = 10` (Sturges gives ≈9.6 at 192) and the
  frequency resolution `fs/T`, which coarsens 2 Hz → 5.2 Hz and costs `dominant_frequency` precision.
  **Deliberately left unchanged**: moving either would shift every affected feature value and break
  comparability with Phase-1 banks. Documented in the Notation section; **route to the project-lead if
  we want them re-derived at 192**, since it would be a feature-value change consumers see.

### S2 — Optional extra + provider seam ✅ (est. 1.5–3 h)

- **Change:** `pyproject.toml` gains `[project.optional-dependencies] catch22 = ["pycatch22>=0.4.5"]`
  (base dependencies untouched) plus a mypy `ignore_missing_imports` override — no py.typed upstream.
  New `src/myocard_egm_features/providers.py`: the `FeatureProvider` Protocol, `NativeProvider`
  wrapping the existing 11, and the dependency gate — `require_pycatch22()` / `pycatch22_available()`
  — carrying the ImportError message that names the extra and the build-tools requirement.
- **Scope change (deliberate):** `Catch22Provider` itself moves to **S3**, alongside the
  `catch22.py` module it wraps. Shipping the class here would mean a stub with no `feature_names`
  and no `extract`, since both need the S3 name map — an untestable half-class. The *dependency*
  concern (which is what S2 is really about) lands here in full and is tested now; the provider
  class lands when there is something for it to delegate to. No net change to the step count.
- **Also (unplanned, necessary):** the seven project-standard policy constants **moved** from
  `bundle.py` to `providers.py`, where the code that applies them now lives. `bundle` re-exports
  them under their original names, so `bundle.ACTIVATION_METHOD` and friends stay public API
  unchanged. This is what keeps the import graph acyclic: `bundle → providers` and
  `sets → providers`, never the reverse — without the move, S6's `bundle → sets → providers →
  bundle` would be a cycle. Architecture.md's "constants live in bundle.py" line is corrected at S8.
- **Verify:** ✅ `NativeProvider` satisfies the Protocol via `isinstance`; `available()` is
  unconditionally true; provider output **matches `bundle.extract_all` value-for-value** (the test
  that catches policy-constant drift); the ImportError text is asserted against a simulated base
  install, including that the original cause is chained; the 11 native features still work with
  `pycatch22` unimportable. **87 tests pass** (was 76), ruff clean, **`src/` mypy-clean**.
- **Caught in review:** two native features (`zero_crossings`, `sec_peak_count`) return `int`, not
  `float`. Coercing them in the provider would have flipped those bundle columns from `int64` to
  `float64` for every existing consumer — a breaking change disguised as a refactor. The Protocol
  now documents `float` as the PEP-484 numeric tower rather than a runtime guarantee, and a test
  pins both the `int` returns and the `int64` column dtypes.
- **Added after review (D13):** `features=` selection on the Protocol and on `NativeProvider` —
  computing only what was asked, and computing the shared periodogram only if a frequency feature is
  wanted. Verified: selection returns **documentation order, not caller order**; every selected value
  equals its full-extraction value; **the expensive call is not made when unrequested** (asserted on
  *invocation* rather than timing, so it cannot go flaky); the periodogram is computed exactly once
  for the three frequency features; an unknown name raises rather than silently narrowing the
  comparison. **92 tests pass** (was 87).
- **Depends on:** S1.

### S3 — `catch22.py` batch primitive + name mapping + `Catch22Provider` ✅ (est. 1–2 h)

- **Change:** `HCTSA_TO_NAME` (D5) and `catch22_all(signal, *, catch24=False) -> dict[str, float]` —
  ndarray→list at the boundary (D6), keyed by **our** names via the hctsa codes, one C call for the
  full set (D4).
  Also the per-call `NaN` detection D7a needs (return which features were `NaN`, don't warn here —
  the batch layer aggregates). **Plus `Catch22Provider`** in `providers.py`, deferred from S2:
  `feature_names` from `HCTSA_TO_NAME`, `available()` delegating to `pycatch22_available()`, and
  `extract` calling `catch22_all` and ignoring `fs_hz` per the uniform signature.
- **Verify:** a test asserting `HCTSA_TO_NAME` maps `SP_Summaries_welch_rect_area_5_1` →
  `low_freq_power` and `..._centroid` → `centroid_freq`, with the two-sine discriminating case from
  D5 as the evidence — the guard that the two easily-confused spectral features are mapped to the
  statistic they actually compute. Plus: 22/24
  key counts, a sine anchor, and constant-signal NaN behaviour (D7).
- **Also moved (necessary):** the dependency gate (`CATCH22_EXTRA_HINT`, `require_pycatch22`,
  `pycatch22_available`) moved from `providers.py` into `catch22.py`, where it conceptually belongs —
  otherwise `providers → catch22 → providers` would be a cycle. `providers` re-exports all three, so
  S2's public surface is unchanged and its tests still import from either module.
- **Verify:** ✅ mapping checked **against physics** (a 50 Hz sine gives `centroid_freq` = 0.3191 vs
  an exact 0.3142 rad/sample, `low_freq_power` = 0.9920, and a fast sine inverts both); our mapping
  differs from upstream's in exactly the two deliberate places, so a third difference fails the
  build; every hctsa code upstream returns is one we map; documentation order for both 22 and 24;
  ndarray accepted where raw pycatch22 raises `SystemError`; non-1D rejected; **twelve theory.md §4
  anchors** re-asserted; constant / `NaN` / `inf` input all give 19-of-22 `NaN` without raising; a
  5-sample trace gives numbers, not `NaN` (documented sharp edge); `Catch22Provider` makes **one**
  C call regardless of selection size (D4), ignores `fs_hz`, and reports its 24 names even when the
  extra is absent; and — after Daniel's review — **selection computes only the requested features**,
  asserted by spying on which `pycatch22` entry points were invoked, with equivalence to the bundled
  call checked across all six anchor signals. **130 tests pass** (was 92).
- **Correction made here:** D5's "crossed short names" claim was **wrong and is retracted** — see
  D5. The shipped mapping was always correct; the justification was not. Fleet notified as CL-140.
- **Depends on:** S2.

### S4 — Single-feature access ✅ (est. 1–2 h)

- **Scope change (Daniel, 2026-08-06):** planned as **24 named per-feature wrappers**
  (`catch22.mode_5(x)`, `catch22.trev(x)`, …) mirroring how §1–§3 expose every feature. Shipped
  instead as **one helper**, `catch22_feature(signal, name) -> float`.

  **Both original justifications had evaporated by the time S4 came up.** The repo's stated reason
  for per-feature wrappers (architecture.md, "Why wrap antropy") is that *the wrappers are the
  parameter policy* — `sample_entropy`'s `m` and `r` are ours to choose, and pinning them in one
  place is what stops two consumers computing differently-parameterised features under one column
  name. **catch22 has no parameters to pin** (§4.1.1), so 24 wrappers would carry no policy. And the
  second rationale — that individual calls were the cheap path for a single feature — was made
  redundant by the D4 revision, since `catch22_features` already computes exactly what is asked
  for. What remained was symmetry, 24 names on the public surface, and 24 docstrings duplicating
  theory.md §4.
- **Change:** `catch22_feature(signal, feature)` delegating to `catch22_features`, so a caller
  writes `catch22_feature(x, "trev")` rather than `catch22_features(x, ["trev"])["trev"]`. The
  extraction section of `catch22.py` gained a comment recording the asymmetry with §1–§3 and why it
  is deliberate, so the next reader does not "fix" it.
- **Verify:** ✅ the helper matches the full set **for all 24 features** (so the sugar cannot drift
  on one nobody exercises); returns a bare `float`; propagates `NaN`; rejects an unknown name; and
  **invokes exactly one `pycatch22` entry point**, asserted by spying rather than timing.
  **135 tests pass** (was 130); ruff exit 0; no new mypy findings.
- **Depends on:** S3.

### S5 — Feature-set registry ✅ (est. 1–2 h)

- **Change:** New `src/myocard_egm_features/sets.py`, pure metadata — nothing here touches a signal.
  `FEATURE_REGISTRY` maps all 35 names to their owning provider; `ALL_FEATURE_NAMES` fixes canonical
  order (native 11, then catch24's 24) so two callers asking for the same set get identically-ordered
  columns; `FEATURE_SETS` holds the five presets; `resolve(selection)` accepts a set name *or* a name
  list and returns canonical order, deduplicated; `group_by_provider(names)` splits a mixed selection
  so each provider is called **once** (which is what preserves both providers' selection strategies —
  the native one's shared periodogram, and not re-paying setup on either); `check_available(names)`
  raises the install message before extraction.
- **Refinement to the planned criterion.** The plan said resolving a catch22 set without the extra
  should raise the ImportError. It doesn't — **`resolve` is metadata and never raises ImportError**,
  because asking *what a set contains* is how someone decides whether installing a C toolchain is
  worth it, and it cannot itself require the toolchain. That would also contradict the Protocol rule
  that `feature_names` works when `available()` is `False`. The intent is met by
  **`check_available`**, called at the point of extraction (S6), which surfaces the provider's own
  actionable message rather than a `KeyError` or a silent `NaN` column.
- **One design note worth keeping:** `check_available` *provokes* the provider's own `ImportError`
  rather than restating install instructions. Duplicating them here would give the module a second
  copy to drift and would need editing for every provider added.
- **Verify:** ✅ the shipped set list is pinned **exactly** (adding a convenience set is the
  temptation D14 guards against, so it fails the build); `catch24` minus `catch22` is exactly
  `{mean, std_dev}`, and that pair
  appears in **no other set** (D9); every set member is a registered feature; a name claimed by two
  providers raises at **import**; `resolve` normalises order and duplicates and rejects unknown
  sets/names with messages naming what exists; `resolve` works with `pycatch22` unimportable;
  grouping covers every requested name and omits idle providers; `check_available` passes for a
  native-only set on a base install and raises the install message for a catch22 one.
  **156 tests pass** (was 135); ruff exit 0; mypy unchanged from the pre-S5 baseline.
- **Depends on:** S4.

### S6 — Bundle integration: `extract_catch22` + `features=` ✅ (est. 1–2 h)

- **Change:** `bundle.extract_catch22(signals, *, catch24=False)` and a `features=` parameter on
  `extract_all` accepting a set name or a name list. Providers are grouped **once per call**, not
  per trace — the grouping depends only on the names, so re-deriving it per trace would add work
  proportional to the batch for nothing. Availability is checked **before** extraction begins.
  Column order is canonical (§1–§3 then §4) regardless of request order; `features=None` preserves
  today's 11-column output exactly (D8).
- **Verify:** ✅ **default output proved identical to a pristine HEAD checkout** via
  `pandas.testing.assert_frame_equal` — values, dtypes, index, and columns — which is the D8
  guarantee stated as a check rather than an intention, and the one that would catch an accidental
  `int64 → float64` on the two count columns; default still agrees with the three per-module
  helpers concatenated; selection returns **canonical order, not caller order**, and its values
  equal the corresponding slice of a full extraction; a set name and a mixed cross-provider list
  both work; unknown names rejected; non-2D input still rejected; the default path works with
  `pycatch22` unimportable.
- **The availability check is asserted to happen *first*.** `test_missing_extra_fails_before_any_extraction`
  monkeypatches `_extract` to explode, so if availability were checked lazily the test fails —
  a 10,000-trace batch must die immediately, not after minutes of work.
- **D7a warning verified as designed:** exactly **one** `RuntimeWarning` for a 50-trace all-degenerate
  batch (per-trace would emit fifty and slow the loop it reports on); the message names the count
  (`1 of 4 traces` — the number that says "one bad channel" vs "broken export"), the worst features
  by frequency, and what happened to the values; `NaN` still reaches the DataFrame unchanged, since
  the warning is diagnostic and must not alter output; and **clean input warns about nothing**
  (asserted with `simplefilter("error")`).
- **Measured end-to-end:** default 11 = 0.94 ms/trace, two features = 0.046 ms/trace (**21× cheaper**),
  all 33 = 1.52 ms/trace. **171 tests pass** (was 156); ruff exit 0; `src/` mypy-clean, the 12 new
  findings all the `type-arg` numpy-stub artifact in the new test file, matching existing test style.
- **Caught in review (Daniel):** `extract_catch22` originally required `fs_hz`, which it can never
  use. The uniform-signature argument is a **Protocol** concern — the registry calls providers
  generically and cannot know which needs what — and does not apply to a public function that is
  catch22-only by construction. Dropped.
- **Same fix applied one level up.** `extract_all`'s `fs_hz` is now `float | None = None`, required
  only when a requested feature actually needs it. `providers.REQUIRES_FS_HZ` is the public,
  documented source of that fact (the three §2 spectral features), so the check is derived rather
  than hardcoded, and the error **names the offending feature** — with a selection API, "why does
  this need a sample rate?" is not obvious from a request of a dozen names. Backwards compatible:
  existing callers pass `fs_hz` and are unaffected, verified by a test that a supplied-but-unused
  `fs_hz` gives an identical frame. Behaviour change worth noting: forgetting `fs_hz` on the default
  selection now raises `ValueError` naming the three features instead of a bare `TypeError`.
- **Where an unused rate is passed internally, it is `NaN`,** not a plausible `1.0` — if a feature on
  that path ever became rate-dependent, the result is visibly wrong rather than silently computed
  against a made-up number.
- **178 tests pass** (was 171); default output re-verified identical to HEAD after both changes.
- **Depends on:** S5.

### S7 — Base-install guard + CI + cost measurement ✅ (est. 1.5–3 h)

- **Change:** CI installs `.[dev,catch22]` for the main test matrix — **no wheel assumed**, the
  runner compiles from sdist — plus a **base-install job** that installs `.[dev]` only and runs a
  marked subset proving the package imports, the 11 features work, and requesting a catch22 set
  fails with the actionable message. Measure per-trace extraction cost for the 11, the 22, and the
  union on a realistic `(N=1000, T=512)` batch; record the numbers in `docs/usage.md` so STU4/STU5
  can size their sweeps, and confirm or refute D6's "negligible next to `sample_entropy`" claim with
  an actual figure.
- **Verify:** ✅ the base-install path was **actually run**, not just configured: hiding `pycatch22`
  from the import system via a `sitecustomize` meta-path blocker gives a true absence
  (`find_spec` raises, `pycatch22_available()` is `False`), and the full suite then reports
  **129 passed, 49 skipped, 0 failed**. The local recipe, for repeating it without a venv rebuild:
  put a `sitecustomize.py` inserting a blocking `MetaPathFinder` on `PYTHONPATH`, then run pytest.
- **The guard step exists because the job would otherwise rot silently.** If anything ever pulled
  `pycatch22` in transitively, every catch22 test would run and pass, the base-install path would go
  unexercised, and the job would still report green. So it asserts absence explicitly and fails the
  build if the extra is present. `-rs` lists the skips so the log shows they happened.
- **Measured** (N=1000 EGM-like traces, T=192, 1 kHz): native 11 = **0.91 ms/trace** (9.1 s per 10k
  bank), catch22 = 0.41 ms, catch24 = 0.42 ms, all 33 = 1.50 ms (15.0 s), a 3-feature study set =
  **0.06 ms** — a fifteenth of the default. Recorded in `docs/usage.md`.
- **Second correction to an inherited claim.** Both `docs/usage.md` and `roadmap.md` asserted
  `sample_entropy` was the extraction bottleneck. At `T = 192` it is **14%** of the cost and
  `shannon_entropy` is **3.5× more expensive**. The original claim holds at `T = 512`, where the
  `O(T²)` term takes over, so both documents now say the profile must be **re-measured when `T`
  changes** rather than restated. The perf watch-trigger is explicitly **not tripped** by this
  measurement — selection is the lever, and it already exists.
- No benchmark added as a CI gate — architecture.md keeps profiling component-internal.
- **Depends on:** S6.

### S8 — Docs + phase exit ✅ (est. 1–2 h)

- **Change:** `docs/usage.md` (new API + sets + the NaN policy + the measured costs),
  `README.md` (feature count, the `[catch22]` extra and its **build-tools requirement**),
  `project/architecture.md` (D1–D5 and D10 as durable rationale — the wrapper argument extended to
  pycatch22, the provider seam and why it exists, the hctsa-code-as-key rule, the
  registry-vs-adapter boundary), `CHANGELOG.md` `[Unreleased]`, `roadmap.md` (drop the now-answered
  `features=[...]` open API question; re-file the Wittkampf upgrade + extra features as
  **watch-triggered**, per design §4), version → 0.2.0.
- **Verify:** ✅ full pre-PR run — ruff check and format exit 0, `src/` mypy-clean, **178 tests
  pass** with the extra and **129 pass / 49 skip** on a simulated base install, no IDE/venv/large
  files, version consistent across `pyproject.toml` (0.2.0), installed metadata, and the CHANGELOG.
  Every code example in `README.md` and `docs/usage.md` was **executed**, not eyeballed.
- **Docs corrected, not just extended.** Four claims in the existing docs had gone stale or were
  wrong, and each is fixed where it lives rather than contradicted elsewhere:
  1. `architecture.md` said the policy constants live in `bundle.py` — they moved to `providers.py`
     at S2, with the reason (the cycle it would otherwise create) recorded.
  2. `architecture.md` pencilled the typed-contract work in "at v0.2.0" — v0.2.0 is FEA1's release,
     so that note now says it lands later, matching `roadmap.md`.
  3. `usage.md` and `roadmap.md` said `sample_entropy` dominates cost — corrected at S7, with the
     instruction to re-measure when `T` changes rather than restate a fixed ordering.
  4. The package docstring, README module map, and `pyproject.toml` header all described a
     three-module, eleven-feature library.
- **New durable rationale in `architecture.md`,** promoted from the plan's design notes so it
  outlives this file: why catch22 gets **no** per-feature wrappers (the antropy pattern encodes
  parameter policy; catch22 has no parameters), why we key off hctsa codes, the provider seam and
  the per-provider selection contract, and why the registry ships no study-specific sets.
- **Backlogged at review (Daniel):** verify the catch22 extra installs on **Windows and macOS**.
  Everything verified so far is Linux — CI, Daniel's box, the project-lead's clean build — and
  `pycatch22` compiles from source everywhere. Windows needs MSVC Build Tools, a multi-GB installer
  often blocked on managed university machines, which is exactly the population most likely to want
  this library. The item says measure first, then prefer upstream wheels or conda-forge over a
  pure-Python reimplementation, since D1 rejected reimplementation on correctness grounds. It also
  notes `CATCH22_EXTRA_HINT`'s "build-essential on Debian/Ubuntu" text is Linux-only and needs
  fixing regardless of the verdict. Ties to D2's post-Phase-2 deferral and the project-level
  publishing-timing decision.
- **Depends on:** all prior steps.

## Phase exit

FEA1 is complete. What remains is not implementation:

1. **PR `development → release`**, using `intracardiac-platform/project/pr_checklist.md` (run above).
2. **Tag `v0.2.0`** — the Wave-1 gate requires egm-signal *and* egm-features tagged before Wave 2
   begins (design §7), so this repo's tag is a gating artifact, not just bookkeeping.
3. **Report up to the project-lead** so design §10's Wave-1 row can move.
4. **Delete this plan at phase cleanup** — per the template it is ephemeral, and its durable content
   has already been moved: design rationale to `architecture.md`, shipped work to `CHANGELOG.md`,
   deferred work to `roadmap.md`. Effort tracking was skipped this phase (design §6), so nothing
   rolls up to `estimation_ledger.csv`.

**Open coordination-log items raised by this repo, all non-blocking:** CL-140 (the pycatch22
naming retraction), CL-141 (D4 revised — no bulk discount), CL-142 (egm-studio owns its own catch22
subset). None gate the tag.

## Effort tracking

> **⚠ Effort tracking is SKIPPED for Phase 1.5** (Daniel, 2026-07-29 — design §6): no §6 roll-up, no
> `Actual`/`Elapsed`, nothing appended to `estimation_ledger.csv`; the ledger stays cold. Repos keep
> *rough* estimates in their own plans, which is all the per-step figures below are. The session log
> and the by-issue table are left in place but **not maintained this phase**. Original method, for
> when it's readopted:
> `intracardiac-platform/project/investigations/estimate_vs_actual_tracking.md`.
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
  source-only distribution, D6 (rejects ndarray), and D7 (19/22 NaN on constant input). A fourth
  claim from that probe — that upstream's `short_names` crossed two labels — was a misreading and
  is **retracted**; see D5.
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
