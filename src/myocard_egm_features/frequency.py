"""Frequency-domain feature extractors for intracardiac bipolar EGM traces.

Three features, all built on a periodogram-based power spectral density
(PSD) estimate computed via ``scipy.signal.periodogram``. See
``docs/theory.md`` §2 for the math and parameter-choice rationale, and
``project/architecture.md`` for the overall API conventions.

Functions:

- :func:`periodogram` — shared PSD computation. Public so callers
  running multiple frequency features over the same trace can compute
  it once and pass the result into the individual feature functions.
- :func:`spectral_centroid` — power-weighted mean frequency (Hz).
- :func:`spectral_entropy` — Shannon entropy of the normalized PSD,
  normalized to ``[0, 1]``.
- :func:`dominant_frequency` — frequency at the PSD peak (Hz).

All three features require ``fs_hz`` as a keyword-only argument (no
default per the math-vs-policy split — sample rate depends on the
upstream acquisition / producer configuration). To avoid recomputing
the periodogram three times per trace when extracting all three
features in sequence (the ``bundle.extract_all`` use case), each feature
also accepts optional pre-computed ``freqs`` and ``psd`` arrays — the
output pair of :func:`periodogram`. Both must be supplied together;
passing only one raises ``ValueError``.

Example::

    from myocard_egm_features.frequency import (
        periodogram, spectral_centroid, spectral_entropy, dominant_frequency,
    )

    freqs, psd = periodogram(signal, fs_hz=1000.0)
    c = spectral_centroid(signal, fs_hz=1000.0, freqs=freqs, psd=psd)
    h = spectral_entropy(signal, fs_hz=1000.0, freqs=freqs, psd=psd)
    f = dominant_frequency(signal, fs_hz=1000.0, freqs=freqs, psd=psd)

When called without ``freqs``/``psd`` (i.e. as a one-off), each
function falls back to computing the periodogram internally — the
single-feature use case stays a one-liner.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import periodogram as _scipy_periodogram

__all__ = [
    "dominant_frequency",
    "periodogram",
    "spectral_centroid",
    "spectral_entropy",
]


def periodogram(
    signal: NDArray[np.floating],
    *,
    fs_hz: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Power spectral density via ``scipy.signal.periodogram``.

    Public so callers running multiple frequency features over the
    same trace can compute the PSD once and pass the result into each
    feature function via its ``freqs``/``psd`` kwargs — avoiding the
    redundant recomputation that would otherwise happen for every
    feature.

    See ``docs/theory.md`` §2.1 for the math.

    Parameters
    ----------
    signal
        1D trace of shape ``(T,)``.
    fs_hz
        Sample rate in Hz. Required keyword-only argument.

    Returns
    -------
    tuple[NDArray, NDArray]
        ``(freqs, psd)`` where ``freqs`` is the Nyquist-limited
        frequency grid (``[0, fs_hz/2]``, length ``T // 2 + 1``) and
        ``psd`` is the single-sided power spectral density on the same
        grid.
    """
    freqs, psd = _scipy_periodogram(signal, fs=fs_hz)
    return freqs, psd


def _resolve_psd(
    signal: NDArray[np.floating],
    fs_hz: float,
    freqs: NDArray[np.floating] | None,
    psd: NDArray[np.floating] | None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Helper: use caller-supplied ``freqs``/``psd`` if both given, else compute.

    Both-or-none semantics — supplying only one raises ``ValueError``
    so callers can't accidentally pass a stale ``psd`` against a
    freshly-computed ``freqs`` grid (or vice versa).
    """
    if freqs is None and psd is None:
        return periodogram(signal, fs_hz=fs_hz)
    if freqs is None or psd is None:
        raise ValueError(
            "freqs and psd must both be provided together (or both omitted); "
            f"got freqs={'set' if freqs is not None else 'None'}, "
            f"psd={'set' if psd is not None else 'None'}."
        )
    return freqs, psd


def spectral_centroid(
    signal: NDArray[np.floating],
    *,
    fs_hz: float,
    freqs: NDArray[np.floating] | None = None,
    psd: NDArray[np.floating] | None = None,
) -> float:
    """Power-weighted mean frequency.

    .. math::
        \\text{centroid}(x) = \\frac{\\sum_k f_k \\cdot \\hat{P}(f_k)}{\\sum_k \\hat{P}(f_k)}

    See ``docs/theory.md`` §2.2.

    Parameters
    ----------
    signal
        1D trace of shape ``(T,)``. Ignored when both ``freqs`` and
        ``psd`` are supplied.
    fs_hz
        Sample rate in Hz. Required keyword-only argument. Ignored
        when both ``freqs`` and ``psd`` are supplied.
    freqs, psd
        Optional pre-computed periodogram pair (the output of
        :func:`periodogram`). Pass both to skip the internal PSD call;
        useful when running multiple frequency features over the same
        trace. Passing only one of the two raises ``ValueError``.

    Returns
    -------
    float
        Centroid frequency in Hz. Returns ``0.0`` for a degenerate
        all-zero signal (no power to weight).
    """
    freqs, psd = _resolve_psd(signal, fs_hz, freqs, psd)
    total_power = float(psd.sum())
    if total_power <= 0.0:
        return 0.0
    return float(np.sum(freqs * psd) / total_power)


def spectral_entropy(
    signal: NDArray[np.floating],
    *,
    fs_hz: float,
    freqs: NDArray[np.floating] | None = None,
    psd: NDArray[np.floating] | None = None,
) -> float:
    """Shannon entropy of the normalized PSD, normalized to ``[0, 1]``.

    Treats the PSD as a probability distribution over frequency bins
    and computes its Shannon entropy in nats, divided by
    ``log(N_bins)`` so the result lives in ``[0, 1]``.

    .. math::
        p_k = \\hat{P}(f_k) / \\sum_j \\hat{P}(f_j) \\\\
        H = -\\sum_k p_k \\log(p_k) \\\\
        \\text{spectral\\_entropy}(x) = H / \\log(N_{bins})

    Anchors:

    - Pure sinusoid → ``~0`` (all power in one bin).
    - White noise → ``~1`` (power evenly spread across bins).

    See ``docs/theory.md`` §2.3.

    Parameters
    ----------
    signal
        1D trace of shape ``(T,)``. Ignored when both ``freqs`` and
        ``psd`` are supplied.
    fs_hz
        Sample rate in Hz. Required keyword-only argument. Ignored
        when both ``freqs`` and ``psd`` are supplied.
    freqs, psd
        Optional pre-computed periodogram pair. See
        :func:`spectral_centroid` for details.

    Returns
    -------
    float
        Normalized entropy in ``[0, 1]``. Returns ``0.0`` for a
        degenerate all-zero signal.
    """
    _freqs, psd = _resolve_psd(signal, fs_hz, freqs, psd)
    total_power = float(psd.sum())
    if total_power <= 0.0:
        return 0.0
    p = psd / total_power
    # Shannon entropy in nats; 0·log(0) = 0 by convention so we mask zeros.
    nonzero = p > 0.0
    h_nats = float(-np.sum(p[nonzero] * np.log(p[nonzero])))
    h_max = float(np.log(len(psd)))
    return h_nats / h_max


def dominant_frequency(
    signal: NDArray[np.floating],
    *,
    fs_hz: float,
    freqs: NDArray[np.floating] | None = None,
    psd: NDArray[np.floating] | None = None,
) -> float:
    """Frequency at which the PSD is maximal.

    .. math::
        \\text{dominant\\_frequency}(x) = f_{k^*}, \\quad k^* = \\arg\\max_k \\hat{P}(f_k)

    Tracks the highest-power peak — NOT the highest-frequency peak. For
    a signal with two peaks at different powers, this returns the
    frequency of the one with more power. See ``docs/theory.md`` §2.4.

    Parameters
    ----------
    signal
        1D trace of shape ``(T,)``. Ignored when both ``freqs`` and
        ``psd`` are supplied.
    fs_hz
        Sample rate in Hz. Required keyword-only argument. Ignored
        when both ``freqs`` and ``psd`` are supplied.
    freqs, psd
        Optional pre-computed periodogram pair. See
        :func:`spectral_centroid` for details.

    Returns
    -------
    float
        Frequency in Hz. For a degenerate all-zero signal, returns
        ``0.0`` (argmax of an all-zero PSD lands at the DC bin).
    """
    freqs, psd = _resolve_psd(signal, fs_hz, freqs, psd)
    return float(freqs[int(np.argmax(psd))])
