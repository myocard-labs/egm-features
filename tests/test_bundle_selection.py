"""Tests for bundle selection, catch22 batching, and the NaN warning.

Kept separate from ``test_bundle.py``, which pins the v0.1.0 behaviour — the
distinction matters, because the single most important property here is that
the default output did **not** change.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from myocard_egm_features import bundle
from myocard_egm_features.providers import REQUIRES_FS_HZ, Catch22Provider
from myocard_egm_features.sets import FEATURE_SETS

FS_HZ = 1000.0
T = 192

catch22_only = pytest.mark.skipif(
    not Catch22Provider().available(), reason="optional catch22 extra not installed"
)


@pytest.fixture
def batch() -> np.ndarray:
    return np.asarray(np.random.default_rng(0).standard_normal((6, T)), dtype=float)


# ---------------------------------------------------------------------------
# The default must not move
# ---------------------------------------------------------------------------


def test_default_output_is_the_original_eleven(batch: np.ndarray) -> None:
    """D8: extract_all's default is the v0.1.0 contract and stays put.

    Column names, order, and dtypes are all part of what consumers depend on —
    a silently-changed dtype would break a downstream join as surely as a
    missing column.
    """
    df = bundle.extract_all(batch, fs_hz=FS_HZ)
    assert list(df.columns) == [
        "peak_to_peak",
        "zero_crossings",
        "activation_position",
        "sec_peak_count",
        "spectral_centroid",
        "spectral_entropy",
        "dominant_frequency",
        "sample_entropy",
        "shannon_entropy",
        "lempel_ziv_complexity",
        "higuchi_fractal_dimension",
    ]
    assert df["zero_crossings"].dtype == np.int64
    assert df["sec_peak_count"].dtype == np.int64
    assert len(df) == len(batch)


def test_default_matches_the_per_module_helpers(batch: np.ndarray) -> None:
    """The helpers are public API and must stay consistent with the whole."""
    whole = bundle.extract_all(batch, fs_hz=FS_HZ)
    parts = pd.concat(
        [
            bundle.extract_time_domain(batch),
            bundle.extract_frequency(batch, fs_hz=FS_HZ),
            bundle.extract_complexity(batch),
        ],
        axis=1,
    )
    pd.testing.assert_frame_equal(whole, parts)


def test_default_needs_no_optional_extra(
    monkeypatch: pytest.MonkeyPatch, batch: np.ndarray
) -> None:
    import builtins

    real_import = builtins.__import__

    def _blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "pycatch22":
            raise ImportError("No module named 'pycatch22'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _blocked)
    assert len(bundle.extract_all(batch, fs_hz=FS_HZ).columns) == 11


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_selection_by_name_list(batch: np.ndarray) -> None:
    df = bundle.extract_all(batch, fs_hz=FS_HZ, features=["sec_peak_count", "peak_to_peak"])
    # Canonical order, not the order the caller listed them in.
    assert list(df.columns) == ["peak_to_peak", "sec_peak_count"]


def test_selection_values_match_the_full_extraction(batch: np.ndarray) -> None:
    """Selecting changes which columns are computed, never their values."""
    full = bundle.extract_all(batch, fs_hz=FS_HZ)
    subset = bundle.extract_all(batch, fs_hz=FS_HZ, features=["peak_to_peak", "sample_entropy"])
    pd.testing.assert_frame_equal(subset, full[["peak_to_peak", "sample_entropy"]])


def test_selection_rejects_an_unknown_name(batch: np.ndarray) -> None:
    with pytest.raises(ValueError, match="not features this library computes"):
        bundle.extract_all(batch, fs_hz=FS_HZ, features=["peak_to_pea"])


def test_non_2d_input_still_rejected() -> None:
    with pytest.raises(ValueError, match="2D array"):
        bundle.extract_all(np.zeros(T), fs_hz=FS_HZ)


@catch22_only
def test_selection_by_set_name(batch: np.ndarray) -> None:
    df = bundle.extract_all(batch, fs_hz=FS_HZ, features="catch22")
    assert list(df.columns) == list(FEATURE_SETS["catch22"])


@catch22_only
def test_mixed_selection_spans_providers(batch: np.ndarray) -> None:
    """The realistic case: catch22 is amplitude-blind, so a set pairs both."""
    df = bundle.extract_all(batch, fs_hz=FS_HZ, features=["trev", "peak_to_peak", "entropy_pairs"])
    assert list(df.columns) == ["peak_to_peak", "trev", "entropy_pairs"]
    assert not df.isna().to_numpy().any()


@catch22_only
def test_extract_catch22_helper(batch: np.ndarray) -> None:
    assert list(bundle.extract_catch22(batch).columns) == list(FEATURE_SETS["catch22"])
    assert list(bundle.extract_catch22(batch, catch24=True).columns) == list(
        FEATURE_SETS["catch24"]
    )


def test_missing_extra_fails_before_any_extraction(
    monkeypatch: pytest.MonkeyPatch, batch: np.ndarray
) -> None:
    """A long batch must fail immediately, not after minutes of work.

    Asserted by making the native provider explode: if availability were
    checked lazily, extraction would start and hit that first.
    """
    import builtins

    real_import = builtins.__import__

    def _blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "pycatch22":
            raise ImportError("No module named 'pycatch22'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _blocked)

    def _explode(*_a: object, **_k: object) -> None:
        raise AssertionError("extraction started before the availability check")

    monkeypatch.setattr(bundle, "_extract", _explode)

    with pytest.raises(ImportError, match="myocard-egm-features\\[catch22\\]"):
        bundle.extract_all(batch, fs_hz=FS_HZ, features=["peak_to_peak", "trev"])


# ---------------------------------------------------------------------------
# The NaN warning (D7a)
# ---------------------------------------------------------------------------


@catch22_only
def test_one_warning_per_batch_not_per_trace() -> None:
    """The whole design point: a bad channel must not emit thousands of warnings."""
    degenerate = np.ones((50, T))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle.extract_all(degenerate, fs_hz=FS_HZ, features="catch22")
    assert len(caught) == 1
    assert caught[0].category is RuntimeWarning


@catch22_only
def test_warning_names_the_count_and_the_features() -> None:
    batch = np.vstack([np.ones(T), np.random.default_rng(0).standard_normal((3, T))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle.extract_all(batch, fs_hz=FS_HZ, features="catch22")
    message = str(caught[0].message)
    assert "1 of 4 traces" in message  # scale: one bad channel vs a broken export
    assert "mode_5" in message  # which feature to go and look at
    assert "propagated, not replaced" in message  # what happened to the values


@catch22_only
def test_nan_values_still_reach_the_frame() -> None:
    """The warning is diagnostic; it must not change what is returned."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df = bundle.extract_all(np.ones((2, T)), fs_hz=FS_HZ, features="catch22")
    assert df.isna().to_numpy().sum() == 2 * 19  # §4.1.4: 19 of 22 undefined


def test_clean_input_warns_about_nothing(batch: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bundle.extract_all(batch, fs_hz=FS_HZ)


# ---------------------------------------------------------------------------
# fs_hz is required only when something actually needs it
# ---------------------------------------------------------------------------


def test_requires_fs_hz_lists_only_the_spectral_three() -> None:
    """Pinned, because the fs_hz contract is derived from this set.

    Note what is *absent*: every catch22 feature. They are defined in samples
    and lags, so they do not take the argument — which is not the same as
    being rate-independent (docs/theory.md, Preprocessing assumptions).
    """
    assert {
        "spectral_centroid",
        "spectral_entropy",
        "dominant_frequency",
    } == REQUIRES_FS_HZ


@pytest.mark.parametrize(
    "features",
    [
        ["peak_to_peak", "sec_peak_count"],  # time-domain only
        ["sample_entropy", "higuchi_fractal_dimension"],  # complexity only
    ],
)
def test_fs_hz_may_be_omitted_when_nothing_needs_it(batch: np.ndarray, features: list[str]) -> None:
    assert list(bundle.extract_all(batch, features=features).columns) == features


@catch22_only
def test_catch22_selection_needs_no_fs_hz(batch: np.ndarray) -> None:
    assert len(bundle.extract_all(batch, features="catch22").columns) == 22


def test_omitting_fs_hz_names_the_feature_that_wanted_it(batch: np.ndarray) -> None:
    """The error has to answer "why?", not just "no".

    With a selection API a caller may have asked for a dozen features; making
    them work out which one needs a sample rate is the unhelpful version.
    """
    with pytest.raises(ValueError) as excinfo:
        bundle.extract_all(batch, features=["peak_to_peak", "dominant_frequency"])
    message = str(excinfo.value)
    assert "dominant_frequency" in message
    assert "peak_to_peak" not in message  # only the offender, not the whole request
    assert "sample rate" in message


def test_default_selection_still_needs_fs_hz(batch: np.ndarray) -> None:
    """The eleven include the spectral three, so the default is unchanged."""
    with pytest.raises(ValueError, match="fs_hz is required"):
        bundle.extract_all(batch)


def test_supplying_fs_hz_when_unused_is_harmless(batch: np.ndarray) -> None:
    """Existing call sites pass fs_hz unconditionally and must keep working."""
    with_it = bundle.extract_all(batch, fs_hz=FS_HZ, features=["peak_to_peak"])
    without = bundle.extract_all(batch, features=["peak_to_peak"])
    pd.testing.assert_frame_equal(with_it, without)
