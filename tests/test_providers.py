"""Tests for the feature-provider seam.

Covers three things: that the Protocol is satisfied, that the native provider
agrees with the bundle it shares policy constants with, and that the optional
catch22 dependency fails legibly rather than with a bare ImportError.
"""

from __future__ import annotations

import builtins
import re

import numpy as np
import pytest

from myocard_egm_features import bundle, providers
from myocard_egm_features.providers import (
    CATCH22_EXTRA_HINT,
    FeatureProvider,
    NativeProvider,
    pycatch22_available,
    require_pycatch22,
)

FS_HZ = 1000.0
T = 192


@pytest.fixture
def trace() -> np.ndarray:
    """An idealised clean activation — theory.md §4.1.5 `biphasic`."""
    i = np.arange(T)
    g = np.exp(-0.5 * ((i - 96.0) / 6.0) ** 2)
    b = -np.gradient(g)
    return np.asarray(b / np.abs(b).max(), dtype=float)


# ---------------------------------------------------------------------------
# The Protocol
# ---------------------------------------------------------------------------


def test_native_provider_satisfies_the_protocol() -> None:
    assert isinstance(NativeProvider(), FeatureProvider)


def test_provider_identity_and_names_are_stable() -> None:
    p = NativeProvider()
    assert p.name == "egm_features"
    assert len(p.feature_names) == 11
    assert len(set(p.feature_names)) == 11, "feature names must be unique"
    # Stable across calls — a registry caches one instance.
    assert p.feature_names == NativeProvider().feature_names


def test_native_provider_is_always_available() -> None:
    assert NativeProvider().available() is True


# ---------------------------------------------------------------------------
# The native provider must agree with the bundle
# ---------------------------------------------------------------------------


def test_extract_returns_every_declared_name_in_order(trace: np.ndarray) -> None:
    p = NativeProvider()
    out = p.extract(trace, fs_hz=FS_HZ)
    assert tuple(out) == p.feature_names
    # "float" is the PEP 484 numeric tower, not a runtime guarantee — see the
    # note on FeatureProvider.extract. bool is excluded deliberately: it is an
    # int subclass and would silently satisfy a looser check.
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in out.values())


def test_count_features_stay_integers(trace: np.ndarray) -> None:
    """Pins the dtype contract the bundle has exposed since v0.1.0.

    ``zero_crossings`` and ``sec_peak_count`` are counts. If a provider
    coerced them to float, every consumer's DataFrame would silently change
    those columns from int64 to float64 — a breaking change disguised as a
    refactor, and exactly what D8 (no change to the default output) forbids.
    """
    out = NativeProvider().extract(trace, fs_hz=FS_HZ)
    assert isinstance(out["zero_crossings"], int)
    assert isinstance(out["sec_peak_count"], int)

    df = bundle.extract_all(trace[np.newaxis], fs_hz=FS_HZ)
    assert df["zero_crossings"].dtype == np.int64
    assert df["sec_peak_count"].dtype == np.int64


def test_provider_and_bundle_agree(trace: np.ndarray) -> None:
    """The whole point of sharing the policy constants.

    If these ever diverge, two consumers are computing differently-parameterised
    features under identical column names — the silent drift the constants exist
    to prevent.
    """
    from_provider = NativeProvider().extract(trace, fs_hz=FS_HZ)
    from_bundle = bundle.extract_all(trace[np.newaxis], fs_hz=FS_HZ).iloc[0]

    assert list(from_bundle.index) == list(from_provider)
    for name, value in from_provider.items():
        assert value == pytest.approx(from_bundle[name], nan_ok=True), name


def test_policy_constants_are_re_exported_by_bundle() -> None:
    """bundle.ACTIVATION_METHOD etc. are public API and must keep working."""
    for const in (
        "ACTIVATION_METHOD",
        "SEC_PEAK_THRESHOLD_FRAC",
        "LZ_BINARIZE_METHOD",
        "SAMPLE_ENTROPY_M",
        "SAMPLE_ENTROPY_R_FRAC",
        "SHANNON_ENTROPY_N_BINS",
        "HIGUCHI_K_MAX",
    ):
        assert getattr(bundle, const) is getattr(providers, const), const


# ---------------------------------------------------------------------------
# The optional dependency
# ---------------------------------------------------------------------------


def test_availability_probe_never_raises() -> None:
    """Callers ask this *before* deciding to request catch22 features."""
    assert isinstance(pycatch22_available(), bool)


def test_require_returns_the_module_when_installed() -> None:
    if not pycatch22_available():
        pytest.skip("catch22 extra not installed")
    assert hasattr(require_pycatch22(), "catch22_all")


def test_missing_extra_raises_an_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a base install and assert the message earns its place.

    A bare ``ModuleNotFoundError: pycatch22`` tells a user nothing about the
    extra or the compiler requirement, which is the whole reason this wrapper
    exists.
    """
    real_import = builtins.__import__

    def _no_pycatch22(name: str, *args: object, **kwargs: object) -> object:
        if name == "pycatch22":
            raise ImportError("No module named 'pycatch22'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _no_pycatch22)

    assert pycatch22_available() is False

    with pytest.raises(ImportError) as excinfo:
        require_pycatch22()

    message = str(excinfo.value)
    assert message == CATCH22_EXTRA_HINT
    # The three things a stuck user needs.
    assert 'pip install "myocard-egm-features[catch22]"' in message
    assert re.search(r"C compiler|build-essential", message)
    assert "no wheels" in message
    # And the original cause is preserved for anyone debugging a broken build.
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_native_features_still_work_without_the_extra(
    monkeypatch: pytest.MonkeyPatch, trace: np.ndarray
) -> None:
    """A base install must be fully functional for the eleven native features."""
    real_import = builtins.__import__

    def _no_pycatch22(name: str, *args: object, **kwargs: object) -> object:
        if name == "pycatch22":
            raise ImportError("No module named 'pycatch22'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _no_pycatch22)

    out = NativeProvider().extract(trace, fs_hz=FS_HZ)
    assert len(out) == 11
    assert not any(np.isnan(v) for v in out.values())


# ---------------------------------------------------------------------------
# Selection — the performance contract
# ---------------------------------------------------------------------------


def test_selection_returns_only_what_was_asked(trace: np.ndarray) -> None:
    out = NativeProvider().extract(trace, fs_hz=FS_HZ, features=["sec_peak_count", "peak_to_peak"])
    # Documentation order, not the order the caller happened to list them in.
    assert tuple(out) == ("peak_to_peak", "sec_peak_count")


def test_selection_matches_the_full_extraction(trace: np.ndarray) -> None:
    """Selecting must not change a value, only which values are computed."""
    p = NativeProvider()
    full = p.extract(trace, fs_hz=FS_HZ)
    for name in p.feature_names:
        assert p.extract(trace, fs_hz=FS_HZ, features=[name])[name] == pytest.approx(
            full[name], nan_ok=True
        ), name


def test_unrequested_features_are_not_computed(
    monkeypatch: pytest.MonkeyPatch, trace: np.ndarray
) -> None:
    """The actual point of selection: the expensive call must not happen.

    ``shannon_entropy`` is the most expensive of the eleven at T=192 (0.34 ms
    per trace, half the total). Asserting on cost would be flaky, so this
    asserts on *invocation* instead.
    """
    from myocard_egm_features import complexity, frequency

    calls: list[str] = []

    def _record_shannon(*_a: object, **_k: object) -> float:
        calls.append("shannon")
        return 0.0

    def _record_periodogram(*_a: object, **_k: object) -> tuple[None, None]:
        calls.append("periodogram")
        return (None, None)

    monkeypatch.setattr(complexity, "shannon_entropy", _record_shannon)
    monkeypatch.setattr(frequency, "periodogram", _record_periodogram)

    NativeProvider().extract(trace, fs_hz=FS_HZ, features=["peak_to_peak"])
    assert calls == [], "computed features nobody asked for"

    NativeProvider().extract(trace, fs_hz=FS_HZ, features=["shannon_entropy"])
    assert calls == ["shannon"]


def test_periodogram_is_shared_and_computed_once(
    monkeypatch: pytest.MonkeyPatch, trace: np.ndarray
) -> None:
    from myocard_egm_features import frequency

    real = frequency.periodogram
    calls: list[int] = []

    def counted(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(frequency, "periodogram", counted)
    NativeProvider().extract(
        trace, fs_hz=FS_HZ, features=["spectral_centroid", "spectral_entropy", "dominant_frequency"]
    )
    assert len(calls) == 1, "the three frequency features must share one spectrum"


def test_unknown_feature_name_raises(trace: np.ndarray) -> None:
    """A typo must fail, not silently narrow the comparison."""
    with pytest.raises(ValueError, match="does not provide"):
        NativeProvider().extract(trace, fs_hz=FS_HZ, features=["peak_to_pea"])
