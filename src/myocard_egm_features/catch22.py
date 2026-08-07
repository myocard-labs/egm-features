"""The catch22 feature set — a thin wrapper over the reference C implementation.

The 22 features of Lubba et al. (2019), plus the two ``catch24`` additions
(``mean``, ``std_dev``). The math, the worked anchors, and the per-feature EGM
interpretation are in ``docs/theory.md`` §4; this module is the plumbing.

Three things it owns, none of which are arithmetic:

1. **The naming.** catch22 features have an ``hctsa`` code and a short name,
   and we key off the code, keeping our own code-to-name mapping — see
   :data:`HCTSA_TO_NAME` for the three reasons.
2. **The optional dependency.** ``pycatch22`` ships as an extra
   (``pip install "myocard-egm-features[catch22]"``); :func:`require_pycatch22`
   turns its absence into a message that says what to do about it.
3. **The array boundary.** ``pycatch22`` accepts only a Python ``list``, so the
   conversion happens here and the library's public ``NDArray`` contract is
   unaffected.

No parameters
-------------

None of these functions take tuning parameters, unlike ``complexity`` and
``time_domain``. catch22's constants are *part of the feature definition* — the
"5" in ``mode_5`` is its bin count, the "40" in ``ami_timescale`` its lag cap —
so exposing them would let a caller compute something that is no longer the
published statistic. See ``docs/theory.md`` §4.1.1.

Degenerate input
----------------

On a constant trace, 19 of the 22 are mathematically undefined and come back as
``NaN``; a trace containing ``NaN`` or ``inf`` does the same. Those values are
propagated rather than replaced (``docs/theory.md`` §4.1.4), because a silent
``0.0`` would enter a distribution distance as though it were a real
coordinate. Batch callers are expected to warn — see
``providers.nan_feature_names``.

Note that a *too short* trace does **not** produce ``NaN``: ``pycatch22``
returns numbers for a 5-sample input. Nothing detects that for you, which is
part of why §4.1.3 documents which features are unreliable at ``T = 192``.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CATCH22_EXTRA_HINT",
    "CATCH22_NAMES",
    "CATCH24_NAMES",
    "HCTSA_TO_NAME",
    "NAME_TO_HCTSA",
    "catch22_all",
    "catch22_feature",
    "catch22_features",
    "pycatch22_available",
    "require_pycatch22",
]


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

#: The 22, as ``hctsa code -> our short name``, in ``docs/theory.md`` §4
#: documentation order (§4.2 distribution shape through §4.10 other) — which is
#: **not** the order ``pycatch22`` returns them in.
#:
#: **Why we maintain this rather than consuming pycatch22's short_names.** Not
#: because that list is wrong — it is correct, and we agree with it on 22 of
#: the 24. Three narrower reasons: (1) two names we choose differently —
#: ``SB_TransitionMatrix_3ac_sumdiagcov`` is ``transition_variance`` here, not
#: ``transition_matrix``, because the feature is the summed column variance of
#: that matrix rather than the matrix; and ``DN_Spread_Std`` is ``std_dev``,
#: not ``SD``. (2) Documentation order — ``pycatch22`` returns features in a
#: different order, and a DataFrame should read the way ``docs/theory.md`` §4
#: reads. (3) The hctsa code is the stable key: it changes only when the
#: feature does, whereas a convenience label could be renamed upstream and
#: silently re-point a column at a different statistic.
#:
#: A test asserts every code pycatch22 returns is one we map, so an upstream
#: feature-set change fails loudly rather than narrowing the comparison space.
_CATCH22_ITEMS: tuple[tuple[str, str], ...] = (
    # §4.2 Distribution shape
    ("DN_HistogramMode_5", "mode_5"),
    ("DN_HistogramMode_10", "mode_10"),
    # §4.3 Extreme-event timing
    ("DN_OutlierInclude_p_001_mdrmd", "outlier_timing_pos"),
    ("DN_OutlierInclude_n_001_mdrmd", "outlier_timing_neg"),
    # §4.4 Linear autocorrelation structure
    ("CO_f1ecac", "acf_timescale"),
    ("CO_FirstMin_ac", "acf_first_min"),
    ("PD_PeriodicityWang_th0_01", "periodicity"),
    ("SP_Summaries_welch_rect_area_5_1", "low_freq_power"),
    ("SP_Summaries_welch_rect_centroid", "centroid_freq"),
    ("IN_AutoMutualInfoStats_40_gaussian_fmmi", "ami_timescale"),
    # §4.5 Nonlinear autocorrelation
    ("CO_trev_1_num", "trev"),
    ("CO_HistogramAMI_even_2_5", "ami2"),
    # §4.6 Simple forecasting
    ("FC_LocalSimple_mean3_stderr", "forecast_error"),
    # §4.7 Incremental differences
    ("MD_hrv_classic_pnn40", "high_fluctuation"),
    ("FC_LocalSimple_mean1_tauresrat", "whiten_timescale"),
    # §4.8 Symbolic
    ("SB_BinaryStats_mean_longstretch1", "stretch_high"),
    ("SB_BinaryStats_diff_longstretch0", "stretch_decreasing"),
    ("SB_MotifThree_quantile_hh", "entropy_pairs"),
    ("SB_TransitionMatrix_3ac_sumdiagcov", "transition_variance"),
    # §4.9 Self-affine scaling
    ("SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1", "rs_range"),
    ("SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1", "dfa"),
    # §4.10 Other
    ("CO_Embed2_Dist_tau_d_expfit_meandiff", "embedding_dist"),
)

#: The two ``catch24`` additions (``docs/theory.md`` §4.11) — the mean and
#: spread that z-scoring removes, and the only features here that see
#: amplitude at all.
_CATCH24_EXTRA_ITEMS: tuple[tuple[str, str], ...] = (
    ("DN_Mean", "mean"),
    ("DN_Spread_Std", "std_dev"),
)

#: Every hctsa code we know, mapped to the name we use as a column label.
HCTSA_TO_NAME: dict[str, str] = dict(_CATCH22_ITEMS + _CATCH24_EXTRA_ITEMS)

#: The 22, in documentation order.
CATCH22_NAMES: tuple[str, ...] = tuple(name for _, name in _CATCH22_ITEMS)

#: The 22 plus the catch24 pair, in documentation order.
CATCH24_NAMES: tuple[str, ...] = CATCH22_NAMES + tuple(name for _, name in _CATCH24_EXTRA_ITEMS)

#: Reverse of :data:`HCTSA_TO_NAME`. Used to reach the individual ``pycatch22``
#: function for one feature, which is how selection avoids computing the rest.
NAME_TO_HCTSA: dict[str, str] = {name: code for code, name in HCTSA_TO_NAME.items()}


# ---------------------------------------------------------------------------
# The optional dependency
# ---------------------------------------------------------------------------

#: Shown when catch22 features are requested without the extra installed.
#: Names both the fix and the C-toolchain requirement, because pycatch22
#: publishes no wheels and the pip failure that follows a missing compiler is
#: considerably less legible than this message.
CATCH22_EXTRA_HINT = (
    "catch22 features require the optional 'catch22' extra, which is not installed.\n"
    '    pip install "myocard-egm-features[catch22]"\n'
    "pycatch22 publishes no wheels, so pip builds it from source: a C compiler "
    "must be available (build-essential on Debian/Ubuntu).\n"
    "The eleven native features in docs/theory.md §1-§3 need no extra and are "
    "always available."
)


def require_pycatch22() -> ModuleType:
    """Import and return ``pycatch22``, or raise with an actionable message.

    Deliberately performs a real import rather than checking for the module
    spec: a half-built C extension is present on disk but fails to import, and
    the point of this function is to fail *legibly* in exactly that case.

    Raises
    ------
    ImportError
        If the extra is not installed or the extension will not import. The
        message is :data:`CATCH22_EXTRA_HINT`; the original error is chained.
    """
    try:
        import pycatch22
    except ImportError as exc:  # pragma: no cover - exercised in the base-install CI job
        raise ImportError(CATCH22_EXTRA_HINT) from exc

    # pycatch22 ships no stubs, so ``ignore_missing_imports`` types it as Any.
    # Binding it to a declared name keeps the return type honest — returning
    # the import directly would leak Any to every caller under strict mode.
    module: ModuleType = pycatch22
    return module


def pycatch22_available() -> bool:
    """Whether the optional catch22 extra can actually be used here.

    Never raises — this is the question a caller asks *before* deciding to
    request catch22 features.
    """
    try:
        require_pycatch22()
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Extraction
#
# Three entry points, narrowing: every feature (:func:`catch22_all`), a named
# subset (:func:`catch22_features`), one (:func:`catch22_feature`). All three
# compute exactly what was asked for and nothing else.
#
# Note the deliberate asymmetry with §1-§3, where each feature is its own
# public function (``time_domain.peak_to_peak``, ``complexity.sample_entropy``,
# …). That pattern exists because those wrappers *are* the project's parameter
# policy — ``sample_entropy``'s m and r are ours to choose, and pinning them in
# one place is what stops two consumers computing differently-parameterised
# features under one column name (``project/architecture.md``, "Why wrap
# antropy"). **catch22 has no parameters to pin** (§4.1.1), so 24 equivalent
# wrappers would carry no policy, duplicate ``docs/theory.md`` §4 in their
# docstrings, and add 24 names to the public surface for nothing but symmetry.
# Selection by name covers the same ground.
# ---------------------------------------------------------------------------


def catch22_features(
    signal: NDArray[np.floating],
    features: Sequence[str],
) -> dict[str, float]:
    """Compute **only** the named catch22 features for one trace.

    ``pycatch22`` exposes every feature as its own entry point as well as the
    bundled ``catch22_all``, and the bundle carries **no discount** — it is a
    loop over the same functions. Measured at ``T = 192``: all 22 bundled cost
    0.373 ms per trace, all 22 called individually cost 0.368 ms, and a subset
    costs in proportion to its size — one feature is 0.003 ms, roughly **100x**
    cheaper, and the 14-feature usable-now set is 0.44x the full cost.

    So selection here is a straight win and there is no crossover to reason
    about. This is the only extraction path in the module; :func:`catch22_all`
    is a convenience wrapper that asks for everything.

    The values are identical to the bundled call — asserted by a test across
    all six anchor signals, including the degenerate ones where 19 features
    return ``NaN``.

    Parameters
    ----------
    signal
        1D trace of shape ``(T,)``. Converted to a Python list **once** and
        shared across the individual calls, because the C bindings reject numpy
        arrays.
    features
        Names to compute. Order is irrelevant — the result is always in
        ``docs/theory.md`` §4 documentation order.

    Raises
    ------
    ImportError
        If the optional extra is not installed.
    ValueError
        If ``signal`` is not one-dimensional, or a name is not a catch22
        feature.
    """
    pycatch22 = require_pycatch22()

    array = np.asarray(signal, dtype=float)
    if array.ndim != 1:
        raise ValueError(
            f"catch22 features are per-trace; expected a 1D signal of shape (T,), "
            f"got ndim={array.ndim}, shape={array.shape}."
        )

    wanted = set(features)
    unknown = sorted(wanted - set(NAME_TO_HCTSA))
    if unknown:
        raise ValueError(f"not catch22 features: {unknown}. Available: {sorted(NAME_TO_HCTSA)}")

    # Converted once and shared across the calls below: pycatch22 rejects numpy
    # arrays, and rebuilding the list per feature would undo the saving.
    data = array.tolist()

    values: dict[str, float] = {}
    for name in CATCH24_NAMES:  # documentation order, whatever order was asked
        if name not in wanted:
            continue
        compute = getattr(pycatch22, NAME_TO_HCTSA[name])
        values[name] = float(compute(data))
    return values


def catch22_all(
    signal: NDArray[np.floating],
    *,
    catch24: bool = False,
) -> dict[str, float]:
    """Compute the whole catch22 (or catch24) set for one trace.

    Convenience wrapper over :func:`catch22_features`. If you want a subset,
    call that instead and pay only for what you asked for.

    Parameters
    ----------
    signal
        1D trace of shape ``(T,)``.
    catch24
        Also return ``mean`` and ``std_dev`` (``docs/theory.md`` §4.11).

    Returns
    -------
    dict[str, float]
        Keyed by **our** short names, in documentation order. Values may be
        ``NaN`` for degenerate input — see the module docstring.
    """
    return catch22_features(signal, CATCH24_NAMES if catch24 else CATCH22_NAMES)


def catch22_feature(
    signal: NDArray[np.floating],
    feature: str,
) -> float:
    """Compute one catch22 feature for one trace.

    The single-feature convenience over :func:`catch22_features`, so a caller
    wanting one value writes ``catch22_feature(x, "trev")`` rather than
    ``catch22_features(x, ["trev"])["trev"]``.

    Costs what one feature costs — 0.003 ms at ``T = 192``, against 0.373 ms
    for the whole set. If you want several, ask for them together: this reloads
    nothing between calls, but each call re-converts the trace to a list, so
    ``catch22_features`` is the cheaper way to get more than one.

    Parameters
    ----------
    signal
        1D trace of shape ``(T,)``.
    feature
        One of :data:`CATCH24_NAMES`. See ``docs/theory.md`` §4 for what each
        one measures and its worked anchor.

    Returns
    -------
    float
        The feature value; ``NaN`` for degenerate input (§4.1.4).

    Raises
    ------
    ImportError
        If the optional extra is not installed.
    ValueError
        If ``signal`` is not one-dimensional, or ``feature`` is not a catch22
        feature name.
    """
    return catch22_features(signal, [feature])[feature]
