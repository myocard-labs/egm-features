"""Time-domain feature extractors for intracardiac bipolar EGM traces.

Four features. Each takes a 1D trace ``signal: NDArray, shape (T,)`` and
returns a scalar. See ``docs/theory.md`` §1 for the math, worked examples,
and parameter-choice rationale; see ``project/architecture.md`` for the
overall API conventions (pure functions, math-vs-policy default split).

Functions:

- :func:`peak_to_peak` — ``max(x) - min(x)``.
- :func:`zero_crossings` — count of sign changes around zero.
- :func:`activation_position` — fractional position in ``[0, 1]`` of the
  detected activation event. Requires the ``method`` policy parameter.
- :func:`sec_peak_count` — count of secondary peaks above a fraction of
  the primary peak amplitude. Requires the ``threshold_frac`` policy
  parameter.

The ``bundle.extract_all`` helper invokes these with the project-standard
policy values (``method="dvdt_max"`` and ``threshold_frac=0.3``). Direct
callers must supply policy values explicitly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

# Public API — explicit so the bundle's import loop is anchored on this list.
__all__ = [
    "activation_position",
    "peak_to_peak",
    "sec_peak_count",
    "zero_crossings",
]


def peak_to_peak(signal: NDArray[np.floating]) -> float:
    """Peak-to-peak amplitude of the trace.

    .. math::
        \\text{peak\\_to\\_peak}(x) = \\max(x) - \\min(x)

    See ``docs/theory.md`` §1.1.

    Parameters
    ----------
    signal
        1D trace of shape ``(T,)``. No assumption on units; the result is
        in the same unit (typically mV for calibrated EGMs).

    Returns
    -------
    float
        ``max(signal) - min(signal)``. Always non-negative for finite-valued
        input.
    """
    return float(signal.max() - signal.min())


def zero_crossings(signal: NDArray[np.floating]) -> int:
    """Count of zero-crossings between adjacent samples.

    Counts the number of sample-to-sample transitions where ``sign``
    changes. Each crossing is counted once regardless of direction.

    See ``docs/theory.md`` §1.2.

    Parameters
    ----------
    signal
        1D trace of shape ``(T,)``. EGM traces are approximately zero-mean
        post-bandpass, so "zero" is the natural reference.

    Returns
    -------
    int
        Number of sign changes in ``signal``. Range ``[0, T-1]``.
    """
    signs = np.sign(signal)
    # `np.diff(signs)` is non-zero exactly at indices where the sign changed
    # between consecutive samples; np.count_nonzero gives the crossing count.
    return int(np.count_nonzero(np.diff(signs)))


def activation_position(
    signal: NDArray[np.floating],
    *,
    method: str,
) -> float:
    """Fractional position in ``[0, 1]`` of the detected activation event.

    See ``docs/theory.md`` §1.3 for the math + limitations.

    Parameters
    ----------
    signal
        1D trace of shape ``(T,)``.
    method
        Required policy parameter (no default per the math-vs-policy
        split documented in ``project/architecture.md``). Must be one of:

        - ``"dvdt_max"`` — argmax of ``|dV/dt|`` approximated by the
          discrete first difference ``signal[i] - signal[i-1]``. The
          clinical activation-time convention (Marchlinski / Wittkampf
          school). Project standard, used by ``bundle.extract_all``.
        - ``"abs_peak"`` — argmax of ``|signal|``. Simpler; captures
          peak amplitude position rather than activation timing.

    Returns
    -------
    float
        Fractional position in ``[0, 1]``: ``peak_idx / (T - 1)``.

    Raises
    ------
    ValueError
        If ``method`` is not one of the two supported values.
    """
    if method == "dvdt_max":
        # Discrete first difference: np.diff returns length T-1 array.
        # argmax index corresponds to the gap between samples i and i+1.
        peak_idx = int(np.argmax(np.abs(np.diff(signal))))
    elif method == "abs_peak":
        peak_idx = int(np.argmax(np.abs(signal)))
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'dvdt_max' or 'abs_peak'.")
    return peak_idx / (len(signal) - 1)


def sec_peak_count(
    signal: NDArray[np.floating],
    *,
    threshold_frac: float,
) -> int:
    """Count secondary peaks above a fraction of the primary peak amplitude.

    See ``docs/theory.md`` §1.4. The algorithm:

    1. Rectify: ``abs_x = |signal|``.
    2. Find the primary peak amplitude ``A_max = max(abs_x)``.
    3. Find all local maxima of ``abs_x`` with prominence
       ``≥ threshold_frac · A_max``.
    4. Subtract 1 to exclude the primary peak itself; floor at 0.

    Parameters
    ----------
    signal
        1D trace of shape ``(T,)``.
    threshold_frac
        Required policy parameter (no default per the math-vs-policy
        split documented in ``project/architecture.md``). Fraction of
        the primary peak amplitude used as the prominence threshold.
        Project standard: ``0.3`` (30%), used by ``bundle.extract_all``.

    Returns
    -------
    int
        Number of secondary peaks. Range ``[0, T // 2]`` in principle;
        in practice ``0`` for a clean activation and a small integer
        otherwise.

    Notes
    -----
    For a real biphasic EGM activation (positive lobe + negative lobe),
    ``|x|`` has two peaks of comparable magnitude — so a clean biphasic
    registers as ``sec_peak_count = 1``. See ``docs/theory.md`` §1.4
    "Note on real biphasic EGM activations" for the framing.
    """
    abs_x = np.abs(signal)
    a_max = float(abs_x.max())
    if a_max <= 0.0:
        # Degenerate input (all-zero signal); no peaks to find.
        return 0
    prominence_threshold = threshold_frac * a_max
    peaks, _ = find_peaks(abs_x, prominence=prominence_threshold)
    return max(0, len(peaks) - 1)
