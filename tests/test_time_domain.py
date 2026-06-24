"""Tests for myocard_egm_features.time_domain.

Each feature's tests use synthetic signals with hand-derivable expected
values from the worked examples in ``docs/theory.md`` §1. The intent is
that a code reviewer who has read the theory doc can verify each test
by checking the worked-example math.
"""

from __future__ import annotations

import numpy as np
import pytest

from myocard_egm_features import time_domain

# ---------------------------------------------------------------------------
# Shared synthetic-signal fixtures (used across multiple feature tests).
# ---------------------------------------------------------------------------


def _positive_bump(
    n_samples: int,
    center: int,
    width: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    """A single positive Gaussian-like bump centered at ``center``.

    Used as the simplified single-bump activation model for the §1.4
    worked examples (positive-only, not biphasic). One peak in ``|x|``
    at the bump's center.
    """
    x = np.zeros(n_samples, dtype=np.float64)
    idx = np.arange(n_samples)
    # Gaussian envelope; standard deviation = width / 4 so the bump width
    # at half-max is roughly ``width``.
    sigma = width / 4.0
    x = amplitude * np.exp(-0.5 * ((idx - center) / sigma) ** 2)
    return x


def _biphasic_pulse(
    n_samples: int,
    center: int,
    width: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Gaussian-modulated ``-sin`` centered at ``center``.

    The §1.3 worked example signal in spirit: positive lobe, zero crossing
    at ``center`` with the steepest slope, negative lobe. We use a Gaussian
    envelope (σ = width / 4) on top of one cycle of ``-sin`` rather than a
    pure ``-sin`` clipped at the boundaries — the clipped version creates a
    boundary discontinuity whose dV/dt magnitude TIES the activation-edge
    slope, making ``np.argmax(|np.diff|)`` land ambiguously at either the
    pulse center or one of the cut edges. A Gaussian envelope tapers the
    pulse smoothly to ~0 well before any boundary, giving the activation
    edge an unambiguous argmax. This matches real biphasic EGM activations
    much better than the clipped-sin idealization.
    """
    idx = np.arange(n_samples)
    u = (idx - center).astype(np.float64)
    sigma = width / 4.0
    envelope = np.exp(-0.5 * (u / sigma) ** 2)
    carrier = -np.sin(2 * np.pi * u / width)
    return amplitude * envelope * carrier


# ---------------------------------------------------------------------------
# peak_to_peak — theory.md §1.1
# ---------------------------------------------------------------------------


def test_peak_to_peak_sine_gives_twice_amplitude() -> None:
    """Pure sinusoid of amplitude A: peak_to_peak = 2A (theory §1.1 worked example)."""
    fs_hz = 1000.0
    t = np.arange(0, 0.512, 1.0 / fs_hz)  # 512 samples
    amplitude = 2.0
    x = amplitude * np.sin(2 * np.pi * 50 * t)
    assert time_domain.peak_to_peak(x) == pytest.approx(2 * amplitude, rel=1e-3)


def test_peak_to_peak_constant_signal_is_zero() -> None:
    """Constant signal has zero range."""
    x = np.full(512, 0.5, dtype=np.float64)
    assert time_domain.peak_to_peak(x) == pytest.approx(0.0)


def test_peak_to_peak_is_nonnegative_for_negative_offset() -> None:
    """Negative-offset sine still yields a non-negative result."""
    fs_hz = 1000.0
    t = np.arange(0, 0.512, 1.0 / fs_hz)
    x = -2.0 + np.sin(2 * np.pi * 50 * t)  # ranges roughly [-3, -1]
    result = time_domain.peak_to_peak(x)
    assert result == pytest.approx(2.0, rel=1e-3)
    assert result >= 0


# ---------------------------------------------------------------------------
# zero_crossings — theory.md §1.2
# ---------------------------------------------------------------------------


def test_zero_crossings_sine_matches_2_f_t() -> None:
    """Sine of frequency f over T_seconds has ~2·f·T_seconds crossings (theory §1.2)."""
    fs_hz = 1000.0
    t_seconds = 0.512
    f = 50.0
    t = np.arange(0, t_seconds, 1.0 / fs_hz)
    x = np.sin(2 * np.pi * f * t)
    expected = 2 * f * t_seconds  # 51.2
    actual = time_domain.zero_crossings(x)
    # One-crossing tolerance for off-grid sampling effects.
    assert abs(actual - expected) <= 1


def test_zero_crossings_constant_signal_is_zero() -> None:
    """A constant signal away from zero has 0 sign changes."""
    x = np.full(512, 0.5, dtype=np.float64)
    assert time_domain.zero_crossings(x) == 0


def test_zero_crossings_polarity_flip_counts_once() -> None:
    """A signal that flips polarity exactly once registers exactly 1 crossing."""
    x = np.concatenate([np.full(100, -1.0), np.full(100, 1.0)])
    assert time_domain.zero_crossings(x) == 1


# ---------------------------------------------------------------------------
# activation_position — theory.md §1.3
# ---------------------------------------------------------------------------


def test_activation_position_dvdt_max_finds_steepest_slope_in_biphasic() -> None:
    """For a biphasic pulse centered at sample k, dV/dt-max method returns ~ k/(T-1).

    The biphasic pulse's steepest slope is at the zero crossing between
    its two lobes, located at sample k (theory §1.3 worked example).
    """
    n_samples = 512
    k = 256
    x = _biphasic_pulse(n_samples=n_samples, center=k, width=40)
    pos = time_domain.activation_position(x, method="dvdt_max")
    # Discrete diff argmax can land at k-1 or k depending on rounding;
    # one-sample tolerance.
    expected = k / (n_samples - 1)
    assert pos == pytest.approx(expected, abs=2.0 / (n_samples - 1))


def test_activation_position_abs_peak_finds_argmax_of_abs_signal() -> None:
    """abs_peak method returns argmax(|signal|) / (T-1)."""
    n_samples = 512
    x = np.zeros(n_samples, dtype=np.float64)
    peak_idx = 100
    x[peak_idx] = 1.0
    pos = time_domain.activation_position(x, method="abs_peak")
    assert pos == pytest.approx(peak_idx / (n_samples - 1))


def test_activation_position_unknown_method_raises_value_error() -> None:
    """Passing an unrecognized method string raises a clear ValueError."""
    x = np.zeros(512, dtype=np.float64)
    x[100] = 1.0
    with pytest.raises(ValueError, match="Unknown method"):
        time_domain.activation_position(x, method="not_a_real_method")


def test_activation_position_method_is_required() -> None:
    """method is a required keyword-only argument; omitting it is a TypeError."""
    x = np.zeros(512, dtype=np.float64)
    x[100] = 1.0
    with pytest.raises(TypeError):
        time_domain.activation_position(x)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# sec_peak_count — theory.md §1.4
# ---------------------------------------------------------------------------


def test_sec_peak_count_single_positive_bump_is_zero() -> None:
    """A single positive bump has one local max in |x|; sec_peak_count = 0.

    Matches the §1.4 'Worked example 1 — single positive bump'.
    """
    n_samples = 512
    x = _positive_bump(n_samples=n_samples, center=256, width=40, amplitude=1.0)
    assert time_domain.sec_peak_count(x, threshold_frac=0.3) == 0


def test_sec_peak_count_primary_plus_half_height_secondary_is_one() -> None:
    """Primary bump + secondary half-height bump = 2 peaks in |x|, sec_peak_count = 1.

    Matches the §1.4 'Worked example 2 — fragmented two-component activation'.
    """
    n_samples = 512
    x = _positive_bump(n_samples=n_samples, center=180, width=40, amplitude=1.0)
    x += _positive_bump(n_samples=n_samples, center=340, width=40, amplitude=0.5)
    assert time_domain.sec_peak_count(x, threshold_frac=0.3) == 1


def test_sec_peak_count_clean_biphasic_registers_as_one() -> None:
    """Real biphasic activations register as 1 (not 0) — see §1.4 'Note on
    real biphasic EGM activations'. Both lobes of |x| clear the threshold.
    """
    n_samples = 512
    x = _biphasic_pulse(n_samples=n_samples, center=256, width=40, amplitude=1.0)
    assert time_domain.sec_peak_count(x, threshold_frac=0.3) == 1


def test_sec_peak_count_threshold_frac_filters_low_secondaries() -> None:
    """A small (0.2A) secondary is filtered out by threshold_frac=0.3 but
    counted by threshold_frac=0.1. Verifies the threshold knob works.
    """
    n_samples = 512
    x = _positive_bump(n_samples=n_samples, center=180, width=40, amplitude=1.0)
    x += _positive_bump(n_samples=n_samples, center=340, width=40, amplitude=0.2)
    # At threshold 0.3, the 0.2-amplitude secondary is below threshold (prominence < 0.3*1.0).
    assert time_domain.sec_peak_count(x, threshold_frac=0.3) == 0
    # At threshold 0.1, both bumps clear the threshold.
    assert time_domain.sec_peak_count(x, threshold_frac=0.1) == 1


def test_sec_peak_count_all_zero_signal_is_zero() -> None:
    """Degenerate input (all zeros) returns 0 without raising."""
    x = np.zeros(512, dtype=np.float64)
    assert time_domain.sec_peak_count(x, threshold_frac=0.3) == 0


def test_sec_peak_count_threshold_frac_is_required() -> None:
    """threshold_frac is a required keyword-only argument; omitting is a TypeError."""
    x = np.zeros(512, dtype=np.float64)
    x[100] = 1.0
    with pytest.raises(TypeError):
        time_domain.sec_peak_count(x)  # type: ignore[call-arg]
