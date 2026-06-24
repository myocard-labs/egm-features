"""Tests for myocard_egm_features.frequency.

Each feature's tests use synthetic signals (pure sinusoid, two-component
sinusoid sum, white noise, DC) with hand-derivable expected values from
the worked examples in ``docs/theory.md`` §2. A code reviewer who has
read the theory doc should be able to verify each test by checking the
worked-example math.

Sample rate is fixed at 1 kHz with **1000-sample** traces throughout, so
``df = fs / N = 1 Hz`` per bin and ``N_bins = N // 2 + 1 = 501``. The
sinusoid frequencies used in the anchors (50 Hz, 150 Hz) are integers,
so they land exactly on bin centers — spectral leakage from the default
rectangular window is therefore negligible and the theoretical
"power-in-one-bin" math applies. With an off-grid frequency (e.g.
50 Hz at N=512) the sinc-shaped leakage spreads energy across many
bins and the theory anchors would not hold — see ``docs/theory.md``
§2.1 "Implementation notes".
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from myocard_egm_features import frequency

# ---------------------------------------------------------------------------
# Shared synthetic-signal helpers (used across multiple feature tests).
# ---------------------------------------------------------------------------

FS_HZ = 1000.0
N_SAMPLES = 1000  # df = 1 Hz; 50/150 Hz land exactly on bin centers (k=50, 150).


def _time_grid() -> np.ndarray:
    """1 kHz sample grid of length N_SAMPLES (0.512 s)."""
    return np.arange(N_SAMPLES, dtype=np.float64) / FS_HZ


def _pure_sine(freq_hz: float, amplitude: float = 1.0) -> np.ndarray:
    """A pure sinusoid at ``freq_hz`` over the standard time grid."""
    return amplitude * np.sin(2 * np.pi * freq_hz * _time_grid())


def _two_sines(f1_hz: float, a1: float, f2_hz: float, a2: float) -> np.ndarray:
    """Sum of two sinusoids; used for power-weighting and entropy anchors."""
    t = _time_grid()
    return a1 * np.sin(2 * np.pi * f1_hz * t) + a2 * np.sin(2 * np.pi * f2_hz * t)


def _white_noise(seed: int = 0, amplitude: float = 1.0) -> np.ndarray:
    """Reproducible Gaussian white noise; entropy anchor for ``~1``."""
    rng = np.random.default_rng(seed)
    return amplitude * rng.standard_normal(N_SAMPLES)


# ---------------------------------------------------------------------------
# spectral_centroid — theory.md §2.2
# ---------------------------------------------------------------------------


def test_spectral_centroid_pure_sine_returns_sine_frequency() -> None:
    """Pure sine at f0 → centroid ≈ f0 (theory §2.2 worked example 1).

    With f0=50 Hz on an exact bin center (df=1 Hz, N=1000), all power
    sits in one bin and the weighted mean equals that bin's frequency
    to within float precision.
    """
    x = _pure_sine(freq_hz=50.0)
    centroid = frequency.spectral_centroid(x, fs_hz=FS_HZ)
    assert centroid == pytest.approx(50.0, abs=0.5)


def test_spectral_centroid_two_sines_power_weighted() -> None:
    """Two sines (50 Hz @ A=1, 150 Hz @ A=2): centroid = 130 Hz exactly.

    Per theory §2.2 worked example 2: power ∝ amplitude², so
    centroid = (50·1 + 150·4) / (1 + 4) = 130 Hz. With both frequencies
    on exact bin centers and only those two bins carrying power, the
    centroid equals 130 Hz to within float precision.
    """
    x = _two_sines(f1_hz=50.0, a1=1.0, f2_hz=150.0, a2=2.0)
    centroid = frequency.spectral_centroid(x, fs_hz=FS_HZ)
    assert centroid == pytest.approx(130.0, abs=0.5)


def test_spectral_centroid_all_zero_signal_is_zero() -> None:
    """All-zero signal has no power; centroid returns 0.0 (degenerate)."""
    x = np.zeros(N_SAMPLES, dtype=np.float64)
    assert frequency.spectral_centroid(x, fs_hz=FS_HZ) == pytest.approx(0.0)


def test_spectral_centroid_dc_signal_is_zero_hz() -> None:
    """A constant non-zero signal concentrates all power at DC (0 Hz)."""
    x = np.full(N_SAMPLES, 0.5, dtype=np.float64)
    assert frequency.spectral_centroid(x, fs_hz=FS_HZ) == pytest.approx(0.0, abs=0.5)


def test_spectral_centroid_fs_hz_is_required() -> None:
    """fs_hz is a required keyword-only argument; omitting it is a TypeError."""
    x = _pure_sine(freq_hz=50.0)
    with pytest.raises(TypeError):
        frequency.spectral_centroid(x)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# spectral_entropy — theory.md §2.3
# ---------------------------------------------------------------------------


def test_spectral_entropy_pure_sine_is_near_zero() -> None:
    """Pure sine concentrates all power in ~one bin → entropy ≈ 0.

    With f0=50 Hz on an exact bin center, ~all power sits in one bin
    and Shannon entropy is essentially zero. The small slack
    accommodates float-precision power in adjacent bins.
    """
    x = _pure_sine(freq_hz=50.0)
    h = frequency.spectral_entropy(x, fs_hz=FS_HZ)
    assert h == pytest.approx(0.0, abs=0.01)


def test_spectral_entropy_white_noise_is_near_one() -> None:
    """Gaussian white noise spreads power evenly across bins → entropy ≈ 1.

    Finite-sample variance keeps the result slightly below 1.0 in
    practice (the per-bin power is χ²-distributed, not uniform). The
    ≥ 0.9 threshold is comfortable for N=512 samples.
    """
    x = _white_noise(seed=42)
    h = frequency.spectral_entropy(x, fs_hz=FS_HZ)
    assert 0.9 <= h <= 1.0


def test_spectral_entropy_two_sines_unequal_power() -> None:
    """Two sines (50 Hz @ A=1, 150 Hz @ A=2): entropy ≈ 0.08.

    Per theory §2.3 worked example: with two unequal-power bins
    (p ≈ 0.2 and p ≈ 0.8), H = -(0.2·ln 0.2 + 0.8·ln 0.8) ≈ 0.500
    nats. Normalised by ln(N_bins) = ln(501) ≈ 6.217 gives ≈ 0.0805.
    Tight tolerance since both frequencies are on bin centers.
    """
    x = _two_sines(f1_hz=50.0, a1=1.0, f2_hz=150.0, a2=2.0)
    h = frequency.spectral_entropy(x, fs_hz=FS_HZ)
    assert h == pytest.approx(0.0805, abs=0.01)


def test_spectral_entropy_pure_sine_below_white_noise() -> None:
    """Sanity ordering: pure-sine entropy < white-noise entropy.

    Independent of the absolute tolerances above — even if the
    leakage estimate is off, the ordering must hold.
    """
    h_sine = frequency.spectral_entropy(_pure_sine(50.0), fs_hz=FS_HZ)
    h_noise = frequency.spectral_entropy(_white_noise(seed=0), fs_hz=FS_HZ)
    assert h_sine < h_noise


def test_spectral_entropy_all_zero_signal_is_zero() -> None:
    """All-zero signal: total_power = 0 → returns 0.0 without raising."""
    x = np.zeros(N_SAMPLES, dtype=np.float64)
    assert frequency.spectral_entropy(x, fs_hz=FS_HZ) == pytest.approx(0.0)


def test_spectral_entropy_in_unit_interval() -> None:
    """Sanity: normalised entropy must live in [0, 1] for any input."""
    for x in (
        _pure_sine(50.0),
        _white_noise(seed=1),
        _two_sines(50.0, 1.0, 150.0, 2.0),
    ):
        h = frequency.spectral_entropy(x, fs_hz=FS_HZ)
        assert 0.0 <= h <= 1.0


def test_spectral_entropy_fs_hz_is_required() -> None:
    """fs_hz is a required keyword-only argument; omitting it is a TypeError."""
    x = _white_noise(seed=0)
    with pytest.raises(TypeError):
        frequency.spectral_entropy(x)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# dominant_frequency — theory.md §2.4
# ---------------------------------------------------------------------------


def test_dominant_frequency_pure_sine_returns_sine_frequency() -> None:
    """Pure sine at f0 → PSD-argmax frequency = f0 (on-grid frequency)."""
    x = _pure_sine(freq_hz=50.0)
    f_peak = frequency.dominant_frequency(x, fs_hz=FS_HZ)
    assert f_peak == pytest.approx(50.0)


def test_dominant_frequency_tracks_higher_power_not_higher_frequency() -> None:
    """Two sines (50 Hz @ A=1, 150 Hz @ A=2) → returns 150 Hz.

    The 150 Hz peak has 4× the power of the 50 Hz peak, so it wins
    argmax. Calls out the highest-power-vs-highest-frequency distinction
    documented in theory §2.4.
    """
    x = _two_sines(f1_hz=50.0, a1=1.0, f2_hz=150.0, a2=2.0)
    f_peak = frequency.dominant_frequency(x, fs_hz=FS_HZ)
    assert f_peak == pytest.approx(150.0)


def test_dominant_frequency_tracks_higher_power_when_lower_frequency_wins() -> None:
    """Two sines (50 Hz @ A=2, 150 Hz @ A=1) → returns 50 Hz.

    Mirror of the previous test: the lower-frequency component now has
    more power, so argmax should pick 50 Hz (not 150 Hz).
    """
    x = _two_sines(f1_hz=50.0, a1=2.0, f2_hz=150.0, a2=1.0)
    f_peak = frequency.dominant_frequency(x, fs_hz=FS_HZ)
    assert f_peak == pytest.approx(50.0)


def test_dominant_frequency_all_zero_signal_is_zero_hz() -> None:
    """All-zero signal: PSD is all zeros; argmax falls at DC bin (0 Hz)."""
    x = np.zeros(N_SAMPLES, dtype=np.float64)
    assert frequency.dominant_frequency(x, fs_hz=FS_HZ) == pytest.approx(0.0)


def test_dominant_frequency_fs_hz_is_required() -> None:
    """fs_hz is a required keyword-only argument; omitting it is a TypeError."""
    x = _pure_sine(freq_hz=50.0)
    with pytest.raises(TypeError):
        frequency.dominant_frequency(x)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# periodogram + PSD-reuse path — bundle.extract_all optimization
# ---------------------------------------------------------------------------


def test_periodogram_shape_and_nyquist_limit() -> None:
    """periodogram returns ``(freqs, psd)`` with length ``N // 2 + 1``."""
    x = _pure_sine(freq_hz=50.0)
    freqs, psd = frequency.periodogram(x, fs_hz=FS_HZ)
    expected_n_bins = N_SAMPLES // 2 + 1
    assert freqs.shape == (expected_n_bins,)
    assert psd.shape == (expected_n_bins,)
    # DC bin and Nyquist bin frame the grid.
    assert freqs[0] == pytest.approx(0.0)
    assert freqs[-1] == pytest.approx(FS_HZ / 2.0)


@pytest.mark.parametrize(
    "feature",
    [
        frequency.spectral_centroid,
        frequency.spectral_entropy,
        frequency.dominant_frequency,
    ],
    ids=["spectral_centroid", "spectral_entropy", "dominant_frequency"],
)
def test_precomputed_psd_matches_recomputed(feature: Callable[..., float]) -> None:
    """Passing pre-computed freqs+psd yields identical result to recomputing.

    Pins the bundle.extract_all optimization: callers can compute the
    periodogram once via :func:`periodogram` and pass the pair into each
    feature function without changing the answer.
    """
    x = _two_sines(f1_hz=50.0, a1=1.0, f2_hz=150.0, a2=2.0)
    freqs, psd = frequency.periodogram(x, fs_hz=FS_HZ)
    result_recomputed = feature(x, fs_hz=FS_HZ)
    result_precomputed = feature(x, fs_hz=FS_HZ, freqs=freqs, psd=psd)
    assert result_precomputed == result_recomputed


@pytest.mark.parametrize(
    "feature",
    [
        frequency.spectral_centroid,
        frequency.spectral_entropy,
        frequency.dominant_frequency,
    ],
    ids=["spectral_centroid", "spectral_entropy", "dominant_frequency"],
)
def test_precomputed_psd_requires_both_freqs_and_psd(
    feature: Callable[..., float],
) -> None:
    """Passing only ``freqs`` (or only ``psd``) raises ValueError.

    Guard against bugs where a caller updates one and forgets the
    other — silently mismatched freqs/psd would produce wrong-but-
    plausible numbers.
    """
    x = _pure_sine(freq_hz=50.0)
    freqs, psd = frequency.periodogram(x, fs_hz=FS_HZ)
    with pytest.raises(ValueError, match="freqs and psd"):
        feature(x, fs_hz=FS_HZ, freqs=freqs)
    with pytest.raises(ValueError, match="freqs and psd"):
        feature(x, fs_hz=FS_HZ, psd=psd)
