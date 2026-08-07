"""Tests for the catch22 wrapper.

The anchors are the ones written into ``docs/theory.md`` §4 — the theory doc
doubles as the test specification, so a value changing here means the doc is
now wrong.

``test_spectral_names_verified_against_physics`` checks the two easily-confused
spectral features by measurement rather than by trusting any label, since
mixing them up would compare a median frequency against a power fraction under
one column name with nothing raised.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable

import numpy as np
import pytest

from myocard_egm_features import catch22
from myocard_egm_features.catch22 import (
    CATCH22_EXTRA_HINT,
    CATCH22_NAMES,
    CATCH24_NAMES,
    HCTSA_TO_NAME,
    catch22_all,
    catch22_feature,
    catch22_features,
    pycatch22_available,
    require_pycatch22,
)

FS_HZ = 1000.0
T = 192

pytestmark = pytest.mark.skipif(
    not pycatch22_available(), reason="optional catch22 extra not installed"
)


# --- the six anchor signals of docs/theory.md §4.1.5 ------------------------


def _sine50() -> np.ndarray:
    return np.asarray(np.sin(2 * np.pi * 50 * np.arange(T) / FS_HZ), dtype=float)


def _white() -> np.ndarray:
    return np.random.default_rng(0).standard_normal(T)


def _ramp() -> np.ndarray:
    return np.arange(T, dtype=float)


def _constant() -> np.ndarray:
    return np.ones(T)


def _sawtooth() -> np.ndarray:
    return (np.arange(T) % 64) / 64.0


def _biphasic() -> np.ndarray:
    i = np.arange(T)
    g = np.exp(-0.5 * ((i - 96.0) / 6.0) ** 2)
    b = -np.gradient(g)
    return np.asarray(b / np.abs(b).max(), dtype=float)


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def test_spectral_names_verified_against_physics() -> None:
    """Confirm the two easily-confused spectral features by measurement.

    ``centroid`` and ``area`` both sound like they could be either statistic,
    and mixing them up would compare a median frequency on one dataset against
    a power fraction on another under a single column name, with nothing
    raised. A 50 Hz sine at 1 kHz has median frequency 2*pi*50/1000 = 0.3142
    rad/sample and essentially all its power below 0.2*Nyquist, so the two are
    separable by inspection.
    """
    out = catch22_all(_sine50())
    assert out["centroid_freq"] == pytest.approx(2 * np.pi * 50 / FS_HZ, abs=0.01)
    assert out["low_freq_power"] > 0.9

    # The discriminating case: a fast sine inverts both.
    fast = np.sin(2 * np.pi * np.arange(T) * 80 / T)
    out_fast = catch22_all(fast)
    assert out_fast["centroid_freq"] > out["centroid_freq"]
    assert out_fast["low_freq_power"] < out["low_freq_power"]


def test_our_mapping_agrees_with_pycatch22_except_where_we_chose_otherwise() -> None:
    """We keep our own mapping, but not because upstream's is wrong.

    It agrees on 22 of 24. The two differences are deliberate naming choices,
    documented in docs/theory.md §4.1.2. If this ever reports a *third*
    difference, upstream renamed something and §4.1.2 needs re-reading.
    """
    raw = require_pycatch22().catch22_all(list(_sine50()), short_names=True, catch24=True)
    theirs = dict(zip(raw["names"], raw["short_names"], strict=True))
    differences = {
        code: (HCTSA_TO_NAME[code], theirs[code])
        for code in theirs
        if HCTSA_TO_NAME[code] != theirs[code]
    }
    assert differences == {
        "SB_TransitionMatrix_3ac_sumdiagcov": ("transition_variance", "transition_matrix"),
        "DN_Spread_Std": ("std_dev", "SD"),
    }


def test_name_tables_are_consistent() -> None:
    assert len(CATCH22_NAMES) == 22
    assert len(CATCH24_NAMES) == 24
    assert len(set(CATCH24_NAMES)) == 24, "names must be unique"
    assert CATCH24_NAMES[:22] == CATCH22_NAMES
    assert CATCH24_NAMES[22:] == ("mean", "std_dev")
    assert set(HCTSA_TO_NAME.values()) == set(CATCH24_NAMES)


def test_every_code_pycatch22_returns_is_mapped() -> None:
    """Guards against an upstream feature-set change passing silently."""
    raw = require_pycatch22().catch22_all(list(_sine50()), catch24=True)
    assert set(raw["names"]) == set(HCTSA_TO_NAME)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_returns_documentation_order() -> None:
    assert tuple(catch22_all(_sine50())) == CATCH22_NAMES
    assert tuple(catch22_all(_sine50(), catch24=True)) == CATCH24_NAMES


def test_accepts_a_numpy_array() -> None:
    """pycatch22 itself rejects ndarray; the conversion is ours to do."""
    with pytest.raises(SystemError):
        require_pycatch22().DN_HistogramMode_5(_sine50())
    assert np.isfinite(catch22_all(_sine50())["mode_5"])


def test_rejects_a_non_1d_signal() -> None:
    with pytest.raises(ValueError, match="1D signal"):
        catch22_all(np.zeros((3, T)))


@pytest.mark.parametrize(
    ("signal_fn", "feature", "expected"),
    [
        # docs/theory.md §4.4.2 — first ACF minimum is half the period.
        (_sine50, "acf_first_min", 10.0),
        # §4.4.4 — a flat spectrum puts ~20% of power in the lowest 20%.
        (_white, "low_freq_power", 0.2584),
        # §4.5.1 — slow rise, sharp drop gives a large negative trev.
        (_sawtooth, "trev", -0.4119),
        # §4.6.1 — the 3-point mean forecaster is worse than useless on noise.
        (_white, "forecast_error", 1.1331),
        # §4.7.1 — one deflection on a quiet baseline: most steps are still.
        (_biphasic, "high_fluctuation", 0.2094),
        # §4.8.1 — longest run above the mean is half the period for a sine.
        (_sine50, "stretch_high", 10.0),
        # §4.8.2 — the long smooth decay of the Gaussian tail.
        (_biphasic, "stretch_decreasing", 72.0),
        # §4.8.3 — white noise sits near the ln(9) ceiling.
        (_white, "entropy_pairs", 2.1803),
        # §4.8.4 — a ramp is the maximally deterministic case, 1/6.
        (_ramp, "transition_variance", 0.1667),
        # §4.10.1 — noise is near-exponential in the embedding.
        (_white, "embedding_dist", 0.0821),
        # §4.4.3 — recovers the 20-sample period within Wang's tolerance.
        (_sine50, "periodicity", 19.0),
        # §4.4.5 — the naming anchor, in rad/sample.
        (_sine50, "centroid_freq", 0.3191),
    ],
)
def test_theory_doc_anchors(
    signal_fn: Callable[[], np.ndarray], feature: str, expected: float
) -> None:
    assert catch22_all(signal_fn())[feature] == pytest.approx(expected, abs=5e-5)


def test_catch24_anchors() -> None:
    """§4.11 — the two features that see amplitude, and are defined on a constant."""
    out = catch22_all(_ramp(), catch24=True)
    assert out["mean"] == pytest.approx(95.5)  # (0 + 191) / 2
    assert out["std_dev"] == pytest.approx(55.5698, abs=5e-5)

    degenerate = catch22_all(_constant(), catch24=True)
    assert degenerate["mean"] == pytest.approx(1.0)
    assert degenerate["std_dev"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------


def test_constant_trace_yields_nan_not_an_exception() -> None:
    """docs/theory.md §4.1.4 — 19 of 22 are undefined, and we propagate."""
    out = catch22_all(_constant())
    nan = [name for name, value in out.items() if value != value]
    assert len(nan) == 19
    assert len(out) == 22


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_input_also_yields_nan(bad: float) -> None:
    signal = np.zeros(T)
    signal[1] = bad
    out = catch22_all(signal)
    assert sum(1 for v in out.values() if v != v) == 19


def test_a_too_short_trace_is_not_flagged() -> None:
    """Documented sharp edge: brevity produces numbers, not NaN.

    pycatch22 happily returns values for a 5-sample input. Nothing in this
    library can tell you they are meaningless — which is why §4.1.3 documents
    reliability by window length instead of relying on a runtime check.
    """
    out = catch22_all(np.array([1.0, 2.0, 1.5, 0.5, 2.5]))
    assert all(v == v for v in out.values()), "no NaN — brevity is not detected"


# ---------------------------------------------------------------------------
# The missing extra
# ---------------------------------------------------------------------------


def test_extraction_without_the_extra_raises_the_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _no_pycatch22(name: str, *args: object, **kwargs: object) -> object:
        if name == "pycatch22":
            raise ImportError("No module named 'pycatch22'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _no_pycatch22)
    with pytest.raises(ImportError, match="catch22"):
        catch22.catch22_all(_sine50())
    assert CATCH22_EXTRA_HINT.startswith("catch22 features require")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_selection_equals_the_bundled_call_on_every_anchor() -> None:
    """The equivalence guarantee that lets us skip the bundled entry point.

    We compute features individually rather than through pycatch22's
    ``catch22_all``, because the bundle carries no discount. That is only safe
    if the two agree exactly — including on the degenerate signals where 19 of
    22 come back NaN, where a subtle difference would be easiest to miss.
    """
    for signal_fn in (_sine50, _white, _ramp, _constant, _sawtooth, _biphasic):
        signal = signal_fn()
        ours = catch22_all(signal)
        raw = require_pycatch22().catch22_all(list(signal))
        bundled = dict(zip(raw["names"], raw["values"], strict=True))
        for code, name in HCTSA_TO_NAME.items():
            if name not in ours:
                continue
            a, b = ours[name], bundled[code]
            assert (a == b) or (a != a and b != b), f"{signal_fn.__name__}/{name}"


def test_selection_returns_documentation_order_regardless_of_input_order() -> None:
    out = catch22_features(_sine50(), ["dfa", "mode_5", "trev"])
    assert tuple(out) == ("mode_5", "trev", "dfa")


def test_selection_matches_the_full_set_value_for_value() -> None:
    full = catch22_all(_biphasic(), catch24=True)
    for name in CATCH24_NAMES:
        one = catch22_features(_biphasic(), [name])[name]
        assert (one == full[name]) or (one != one and full[name] != full[name]), name


def test_selection_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="not catch22 features"):
        catch22_features(_sine50(), ["trevv"])


def test_empty_selection_computes_nothing() -> None:
    assert catch22_features(_sine50(), []) == {}


# ---------------------------------------------------------------------------
# The single-feature helper
# ---------------------------------------------------------------------------


def test_single_feature_helper_matches_the_set() -> None:
    """catch22_feature is sugar, not a second implementation.

    Checked over every one of the 24 so the sugar cannot drift from the set on
    some feature nobody happens to exercise.
    """
    signal = _biphasic()
    full = catch22_all(signal, catch24=True)
    for name in CATCH24_NAMES:
        one = catch22_feature(signal, name)
        assert (one == full[name]) or (one != one and full[name] != full[name]), name


def test_single_feature_helper_returns_a_bare_float() -> None:
    value = catch22_feature(_sine50(), "acf_first_min")
    assert isinstance(value, float)
    assert value == pytest.approx(10.0)  # §4.4.2, half the 20-sample period


def test_single_feature_helper_propagates_nan() -> None:
    assert catch22_feature(_constant(), "trev") != catch22_feature(_constant(), "trev")


def test_single_feature_helper_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="not catch22 features"):
        catch22_feature(_sine50(), "trevv")


def test_single_feature_helper_computes_only_that_feature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One name in must mean one pycatch22 entry point called."""
    called: list[str] = []
    real = require_pycatch22()

    class _Spy:
        def __getattr__(self, code: str) -> object:
            called.append(code)
            return getattr(real, code)

    monkeypatch.setattr(catch22, "require_pycatch22", lambda: _Spy())
    catch22.catch22_feature(_sine50(), "trev")
    assert called == ["CO_trev_1_num"]
