"""Batch feature extraction over an ``(N, T)`` array of traces.

This module is the primary consumer of the per-feature functions in
``time_domain``, ``frequency``, and ``complexity``. It runs all 11
features over a batch and returns a tidy ``pandas.DataFrame`` —
suitable for joining with per-trace metadata (sim_id, patient_id,
etc.) from the caller's bank record.

Public API:

- :func:`extract_time_domain` — 4 time-domain features.
- :func:`extract_frequency` — 3 frequency-domain features. Computes the
  periodogram once per trace and reuses it across all three (the
  optimization wired through :mod:`frequency`'s optional
  ``freqs``/``psd`` kwargs).
- :func:`extract_complexity` — 4 complexity features.
- :func:`extract_all` — all 11 features, concatenated column-wise.

The per-module helpers exist so callers who only want one feature
group don't pay for the others (sample_entropy in particular is
~O(T²) and slow). ``extract_all`` is a thin wrapper that calls all
three and concatenates.

**Project-standard policy values** are applied inside ``extract_all``
(and the per-module helpers that need them). Direct callers of the
per-feature functions are forced to think about which policy they
want — preventing accidental drift across the codebase.

Those constants now live in :mod:`providers`, which is where the code
that applies them lives; they are re-exported here under their original
names, so ``bundle.ACTIVATION_METHOD`` and friends remain part of the
public API unchanged. The import runs one way only — ``bundle`` imports
``providers``, never the reverse — which keeps the feature-set registry
out of an import cycle.

See ``project/architecture.md`` "Defaults policy — math constants vs
project-policy values" for the rationale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from myocard_egm_features import complexity, frequency, time_domain
from myocard_egm_features.providers import (
    ACTIVATION_METHOD,
    HIGUCHI_K_MAX,
    LZ_BINARIZE_METHOD,
    SAMPLE_ENTROPY_M,
    SAMPLE_ENTROPY_R_FRAC,
    SEC_PEAK_THRESHOLD_FRAC,
    SHANNON_ENTROPY_N_BINS,
)

__all__ = [
    "ACTIVATION_METHOD",
    "HIGUCHI_K_MAX",
    "LZ_BINARIZE_METHOD",
    "SAMPLE_ENTROPY_M",
    "SAMPLE_ENTROPY_R_FRAC",
    "SEC_PEAK_THRESHOLD_FRAC",
    "SHANNON_ENTROPY_N_BINS",
    "extract_all",
    "extract_complexity",
    "extract_frequency",
    "extract_time_domain",
]

# ---------------------------------------------------------------------------
# Column ordering — matches ``docs/theory.md`` section order so a reader
# moving between the theory doc and a feature DataFrame sees the same
# left-to-right ordering everywhere.
# ---------------------------------------------------------------------------

_TIME_COLUMNS = [
    "peak_to_peak",
    "zero_crossings",
    "activation_position",
    "sec_peak_count",
]
_FREQ_COLUMNS = [
    "spectral_centroid",
    "spectral_entropy",
    "dominant_frequency",
]
_COMPLEXITY_COLUMNS = [
    "sample_entropy",
    "shannon_entropy",
    "lempel_ziv_complexity",
    "higuchi_fractal_dimension",
]


def _validate_signals(signals: NDArray[np.floating]) -> None:
    """Shape check: signals must be ``(N, T)``.

    A 1D input ``(T,)`` is a common mistake — the per-feature functions
    accept 1D but the bundle is batch-only. Raise rather than auto-
    promote so the caller's intent is unambiguous.
    """
    if signals.ndim != 2:
        raise ValueError(
            f"signals must be a 2D array of shape (N, T); got ndim={signals.ndim}, "
            f"shape={signals.shape}. For a single trace, reshape with "
            "signals[np.newaxis] before calling."
        )


def extract_time_domain(signals: NDArray[np.floating]) -> pd.DataFrame:
    """Extract the 4 time-domain features over a batch of traces.

    Uses the project-standard policy values
    (:data:`ACTIVATION_METHOD`, :data:`SEC_PEAK_THRESHOLD_FRAC`).

    Parameters
    ----------
    signals
        ``(N, T)`` float array of N traces.

    Returns
    -------
    pandas.DataFrame
        N rows, 4 columns: ``peak_to_peak``, ``zero_crossings``,
        ``activation_position``, ``sec_peak_count``.
    """
    _validate_signals(signals)
    rows = [
        {
            "peak_to_peak": time_domain.peak_to_peak(x),
            "zero_crossings": time_domain.zero_crossings(x),
            "activation_position": time_domain.activation_position(x, method=ACTIVATION_METHOD),
            "sec_peak_count": time_domain.sec_peak_count(x, threshold_frac=SEC_PEAK_THRESHOLD_FRAC),
        }
        for x in signals
    ]
    return pd.DataFrame(rows, columns=_TIME_COLUMNS)


def extract_frequency(
    signals: NDArray[np.floating],
    *,
    fs_hz: float,
) -> pd.DataFrame:
    """Extract the 3 frequency-domain features over a batch of traces.

    Computes the periodogram once per trace and reuses it across all
    three features (via the ``freqs``/``psd`` kwargs on each feature
    function) — avoiding the redundant 2× recomputation that would
    happen if each feature was called independently.

    Parameters
    ----------
    signals
        ``(N, T)`` float array of N traces.
    fs_hz
        Sample rate in Hz. Required (policy value).

    Returns
    -------
    pandas.DataFrame
        N rows, 3 columns: ``spectral_centroid``, ``spectral_entropy``,
        ``dominant_frequency``.
    """
    _validate_signals(signals)
    rows = []
    for x in signals:
        freqs, psd = frequency.periodogram(x, fs_hz=fs_hz)
        rows.append(
            {
                "spectral_centroid": frequency.spectral_centroid(
                    x, fs_hz=fs_hz, freqs=freqs, psd=psd
                ),
                "spectral_entropy": frequency.spectral_entropy(
                    x, fs_hz=fs_hz, freqs=freqs, psd=psd
                ),
                "dominant_frequency": frequency.dominant_frequency(
                    x, fs_hz=fs_hz, freqs=freqs, psd=psd
                ),
            }
        )
    return pd.DataFrame(rows, columns=_FREQ_COLUMNS)


def extract_complexity(signals: NDArray[np.floating]) -> pd.DataFrame:
    """Extract the 4 complexity features over a batch of traces.

    Uses the project-standard policy value
    (:data:`LZ_BINARIZE_METHOD`) and math-default constants
    (:data:`SAMPLE_ENTROPY_M`, :data:`SAMPLE_ENTROPY_R_FRAC`,
    :data:`SHANNON_ENTROPY_N_BINS`, :data:`HIGUCHI_K_MAX`).

    **Slowest of the three per-module helpers** — sample_entropy is
    ~O(T²) per trace. For large batches, run on a workstation rather
    than expecting near-realtime extraction.

    Parameters
    ----------
    signals
        ``(N, T)`` float array of N traces.

    Returns
    -------
    pandas.DataFrame
        N rows, 4 columns: ``sample_entropy``, ``shannon_entropy``,
        ``lempel_ziv_complexity``, ``higuchi_fractal_dimension``.
    """
    _validate_signals(signals)
    rows = [
        {
            "sample_entropy": complexity.sample_entropy(
                x, m=SAMPLE_ENTROPY_M, r_frac=SAMPLE_ENTROPY_R_FRAC
            ),
            "shannon_entropy": complexity.shannon_entropy(x, n_bins=SHANNON_ENTROPY_N_BINS),
            "lempel_ziv_complexity": complexity.lempel_ziv_complexity(
                x, binarize_method=LZ_BINARIZE_METHOD
            ),
            "higuchi_fractal_dimension": complexity.higuchi_fractal_dimension(
                x, k_max=HIGUCHI_K_MAX
            ),
        }
        for x in signals
    ]
    return pd.DataFrame(rows, columns=_COMPLEXITY_COLUMNS)


def extract_all(
    signals: NDArray[np.floating],
    *,
    fs_hz: float,
) -> pd.DataFrame:
    """Extract all 11 features over a batch of traces.

    Thin wrapper that concatenates :func:`extract_time_domain`,
    :func:`extract_frequency`, and :func:`extract_complexity`
    column-wise.

    Parameters
    ----------
    signals
        ``(N, T)`` float array of N traces.
    fs_hz
        Sample rate in Hz. Required (used by the frequency features).

    Returns
    -------
    pandas.DataFrame
        N rows, 11 columns in ``docs/theory.md`` section order:

        - Time-domain (§1): ``peak_to_peak``, ``zero_crossings``,
          ``activation_position``, ``sec_peak_count``.
        - Frequency (§2): ``spectral_centroid``, ``spectral_entropy``,
          ``dominant_frequency``.
        - Complexity (§3): ``sample_entropy``, ``shannon_entropy``,
          ``lempel_ziv_complexity``, ``higuchi_fractal_dimension``.
    """
    df_time = extract_time_domain(signals)
    df_freq = extract_frequency(signals, fs_hz=fs_hz)
    df_complex = extract_complexity(signals)
    return pd.concat([df_time, df_freq, df_complex], axis=1)
