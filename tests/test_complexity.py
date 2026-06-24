"""Tests for myocard_egm_features.complexity.

Each feature's tests use synthetic signals (pure sine, white noise,
constant, straight line, periodic binary) with hand-derivable expected
values from the worked examples in ``docs/theory.md`` §3. A code
reviewer who has read the theory doc should be able to verify each
test by checking the worked-example math.

Random-seeded tests use ``numpy.random.default_rng(seed)`` so they're
reproducible. White-noise anchors are loose (range checks) since they
depend on the specific draw, but the ranges are wide enough to survive
seed changes — they pin behavioral expectations, not exact values.
"""

from __future__ import annotations

import numpy as np
import pytest

from myocard_egm_features import complexity

# ---------------------------------------------------------------------------
# Shared synthetic-signal helpers (used across multiple feature tests).
# ---------------------------------------------------------------------------

N_SAMPLES = 512


def _pure_sine(freq_hz: float = 50.0, fs_hz: float = 1000.0) -> np.ndarray:
    """Pure sinusoid; the canonical "perfectly predictable" anchor."""
    t = np.arange(N_SAMPLES, dtype=np.float64) / fs_hz
    return np.sin(2 * np.pi * freq_hz * t)


def _white_noise(seed: int = 0) -> np.ndarray:
    """Reproducible standard-normal noise; the "maximally unpredictable" anchor."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(N_SAMPLES)


def _straight_line() -> np.ndarray:
    """Ramp ``x[i] = i``; fractal-dimension anchor for ``D = 1``."""
    return np.arange(N_SAMPLES, dtype=np.float64)


# ---------------------------------------------------------------------------
# sample_entropy — theory.md §3.1
# ---------------------------------------------------------------------------


def test_sample_entropy_pure_sine_is_near_zero() -> None:
    """Pure sine is perfectly periodic → SampEn near 0 (theory §3.1)."""
    x = _pure_sine()
    se = complexity.sample_entropy(x)
    # antropy's empirical value for a clean 50 Hz sine at fs=1kHz is ~0.19;
    # well below the chaos/noise regime (~2-3). Loose upper bound here.
    assert se < 0.5


def test_sample_entropy_white_noise_is_high() -> None:
    """White noise has SampEn in the ~2-3 range for m=2, r_frac=0.2 (theory §3.1)."""
    x = _white_noise(seed=0)
    se = complexity.sample_entropy(x)
    assert 1.5 <= se <= 3.0


def test_sample_entropy_noise_greater_than_sine() -> None:
    """Sanity ordering: SampEn(noise) > SampEn(sine).

    Holds independent of the absolute tolerances above.
    """
    se_sine = complexity.sample_entropy(_pure_sine())
    se_noise = complexity.sample_entropy(_white_noise(seed=0))
    assert se_noise > se_sine


def test_sample_entropy_custom_r_frac_changes_result() -> None:
    """A different r_frac maps onto a different tolerance and a different result.

    Pins that r_frac is actually plumbed through to antropy (rather than
    silently dropped) — increasing r_frac broadens the tolerance and
    typically lowers SampEn since more pattern matches qualify.
    """
    x = _white_noise(seed=1)
    se_default = complexity.sample_entropy(x, r_frac=0.2)
    se_wide = complexity.sample_entropy(x, r_frac=0.5)
    # Wider tolerance → more matches → smaller -log ratio → lower SampEn.
    assert se_wide < se_default


# ---------------------------------------------------------------------------
# shannon_entropy — theory.md §3.2
# ---------------------------------------------------------------------------


def test_shannon_entropy_constant_signal_is_zero() -> None:
    """Constant signal: all samples in one bin → entropy = 0 (theory §3.2 example 1)."""
    x = np.full(N_SAMPLES, 0.5, dtype=np.float64)
    assert complexity.shannon_entropy(x) == pytest.approx(0.0)


def test_shannon_entropy_uniform_amplitude_approaches_log_n_bins() -> None:
    """Uniform-amplitude signal → entropy ≈ log(n_bins) (theory §3.2 example 2).

    For ``n_bins=10`` the upper bound is ``log(10) ≈ 2.303`` nats. With
    1000 samples from ``uniform(0, 1)`` the empirical histogram is
    nearly uniform → entropy is close to but slightly below the max.
    """
    rng = np.random.default_rng(seed=42)
    x = rng.uniform(0.0, 1.0, size=2000)
    h = complexity.shannon_entropy(x, n_bins=10)
    log_n_bins = float(np.log(10))
    assert h == pytest.approx(log_n_bins, abs=0.1)


def test_shannon_entropy_gaussian_noise_is_below_uniform() -> None:
    """Gaussian noise: ~2.0 nats for n_bins=10, below the uniform max (theory §3.2 example 3)."""
    x = _white_noise(seed=2)
    h = complexity.shannon_entropy(x, n_bins=10)
    # Loose bracket; the specific draw shifts this a bit.
    assert 1.5 <= h <= np.log(10)


def test_shannon_entropy_in_nats_not_bits() -> None:
    """Anchor: for a perfectly-uniform 2-bin signal, entropy = ln(2) ≈ 0.693 nats.

    Pins the units (nats, not bits — bits would give 1.0 exactly).
    """
    # Construct a signal that hits exactly 2 amplitude bins (low half and high half).
    x = np.concatenate([np.zeros(256), np.ones(256)])
    h = complexity.shannon_entropy(x, n_bins=2)
    assert h == pytest.approx(np.log(2), abs=1e-6)


# ---------------------------------------------------------------------------
# lempel_ziv_complexity — theory.md §3.3
# ---------------------------------------------------------------------------


def test_lempel_ziv_complexity_alternating_bits_is_low() -> None:
    """Alternating-bit signal → LZ < 0.1 (theory §3.3 worked example 1).

    Constructs a signal that median-binarizes to exactly "010101..." —
    the walkthrough's anchor sequence — by alternating amplitudes
    sample-by-sample. The theory derivation shows the LZ count stops
    growing once the dictionary saturates (after ~3 substrings), so the
    normalized complexity ``c · log_2(T) / T → 0`` for large ``T``.
    """
    # Alternating ±1: above median = 1, below = 0 → binary "01010101..."
    x = np.where(np.arange(N_SAMPLES) % 2 == 0, -1.0, 1.0)
    c = complexity.lempel_ziv_complexity(x, binarize_method="median")
    assert c < 0.1


def test_lempel_ziv_complexity_sine_is_below_noise() -> None:
    """A pure sine has lower LZ complexity than white noise.

    A real 50 Hz sine at fs=1 kHz binarizes to *chunks* of half-cycle
    width (~10 zeros, ~10 ones) — not single-bit alternation — so its
    LZ value sits well below random noise but above the contrived
    alternating-bit anchor. This test pins the ordering rather than
    a specific value.
    """
    c_sine = complexity.lempel_ziv_complexity(_pure_sine(), binarize_method="median")
    rng = np.random.default_rng(seed=10)
    x_noise = rng.choice([-1.0, 1.0], size=N_SAMPLES)
    c_noise = complexity.lempel_ziv_complexity(x_noise, binarize_method="median")
    assert c_sine < c_noise


def test_lempel_ziv_complexity_random_binary_is_high() -> None:
    """Random binary sequence → LZ ≈ 1 (theory §3.3 worked example 2).

    The normalization ``c · log_2(T) / T`` targets ``1`` as the
    asymptote for random sequences; finite-length sequences can land
    slightly above 1.
    """
    rng = np.random.default_rng(seed=3)
    # Two-level signal: random ±1. After median-binarize this is a uniform
    # random binary string, the worked-example anchor.
    x = rng.choice([-1.0, 1.0], size=N_SAMPLES)
    c = complexity.lempel_ziv_complexity(x, binarize_method="median")
    assert c > 0.7


def test_lempel_ziv_complexity_zero_method_matches_median_for_zero_mean_signal() -> None:
    """For a zero-mean bandpassed signal, median ≈ 0 → 'median' and 'zero' agree.

    Pins the theory §3.3 claim that the two binarization methods are
    nearly equivalent for the kind of preprocessed traces this library
    actually consumes.
    """
    rng = np.random.default_rng(seed=4)
    x = rng.standard_normal(N_SAMPLES)  # zero-mean, so median ≈ 0
    c_median = complexity.lempel_ziv_complexity(x, binarize_method="median")
    c_zero = complexity.lempel_ziv_complexity(x, binarize_method="zero")
    # Not identical (median is empirical, not exactly 0), but within tight band.
    assert abs(c_median - c_zero) < 0.05


def test_lempel_ziv_complexity_unknown_binarize_method_raises() -> None:
    """An unrecognized binarize_method string raises a clear ValueError."""
    x = _pure_sine()
    with pytest.raises(ValueError, match="Unknown binarize_method"):
        complexity.lempel_ziv_complexity(x, binarize_method="not_a_real_method")


def test_lempel_ziv_complexity_binarize_method_is_required() -> None:
    """binarize_method is a required keyword-only argument; omitting is a TypeError."""
    x = _pure_sine()
    with pytest.raises(TypeError):
        complexity.lempel_ziv_complexity(x)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# higuchi_fractal_dimension — theory.md §3.4
# ---------------------------------------------------------------------------


def test_higuchi_fractal_dimension_straight_line_is_one() -> None:
    """Straight line → D = 1 exactly (theory §3.4 worked example 1).

    The derivation in §3.4 shows ``L(k) ∝ k^{-1}`` for a linear ramp,
    so the slope of ``log L(k) vs log k`` is exactly ``-1`` → D = 1.
    """
    x = _straight_line()
    d = complexity.higuchi_fractal_dimension(x)
    assert d == pytest.approx(1.0, abs=0.01)


def test_higuchi_fractal_dimension_white_noise_is_near_two() -> None:
    """White Gaussian noise → D ≈ 2 (theory §3.4 worked example 2)."""
    x = _white_noise(seed=5)
    d = complexity.higuchi_fractal_dimension(x)
    assert 1.9 <= d <= 2.05


def test_higuchi_fractal_dimension_orders_smooth_below_rough() -> None:
    """Sanity ordering: D(smooth biphasic) < D(white noise).

    A smooth biphasic pulse is much closer to a line (D=1) than to white
    noise (D=2). Anchors the geometric intuition.
    """
    # Smooth biphasic-ish pulse (one cycle of -sin with Gaussian envelope).
    idx = np.arange(N_SAMPLES, dtype=np.float64)
    sigma = 40.0 / 4.0
    envelope = np.exp(-0.5 * ((idx - N_SAMPLES // 2) / sigma) ** 2)
    smooth = envelope * -np.sin(2 * np.pi * (idx - N_SAMPLES // 2) / 40.0)
    d_smooth = complexity.higuchi_fractal_dimension(smooth)
    d_noise = complexity.higuchi_fractal_dimension(_white_noise(seed=6))
    assert d_smooth < d_noise
