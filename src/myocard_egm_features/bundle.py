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
- :func:`extract_catch22` — the catch22 set (``docs/theory.md`` §4).
  Needs the optional ``catch22`` extra.
- :func:`extract_all` — the eleven by default, or any selection.

The per-module helpers exist so callers who only want one feature
group don't pay for the others.

Selecting features
------------------

``extract_all(signals, fs_hz=...)`` returns the same eleven columns it
has since v0.1.0 — that default is unchanged and will stay unchanged.
Passing ``features=`` narrows or widens it: a feature-set name
(``"catch22"``), or an explicit list of names. Only what you ask for is
computed; see :mod:`sets` for the vocabulary and :mod:`providers` for
why each provider honours selection differently.

Warning on degenerate input
---------------------------

Features are ``NaN`` where the math is undefined — a constant trace, a
dropped channel — and those values are propagated rather than replaced
(``docs/theory.md`` §4.1.4). Propagating *silently* is its own failure
mode though: a long extraction can finish and hand back a column of
``NaN`` nobody notices. So a batch that produced any ``NaN`` raises
**one** aggregated :class:`RuntimeWarning` naming how many traces and
which features were affected. One warning per batch, not per trace —
per-trace would drown the output and slow the loop it is reporting on.

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

import warnings
from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from myocard_egm_features import complexity, frequency, sets, time_domain
from myocard_egm_features.catch22 import CATCH22_NAMES, CATCH24_NAMES
from myocard_egm_features.providers import (
    ACTIVATION_METHOD,
    HIGUCHI_K_MAX,
    LZ_BINARIZE_METHOD,
    REQUIRES_FS_HZ,
    SAMPLE_ENTROPY_M,
    SAMPLE_ENTROPY_R_FRAC,
    SEC_PEAK_THRESHOLD_FRAC,
    SHANNON_ENTROPY_N_BINS,
    nan_feature_names,
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
    "extract_catch22",
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
    fs_hz: float | None = None,
    features: str | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Extract features over a batch of traces.

    Parameters
    ----------
    signals
        ``(N, T)`` float array of N traces.
    fs_hz
        Sample rate in Hz. Required **only if** a requested feature needs
        it — the three spectral features of §2, listed in
        :data:`providers.REQUIRES_FS_HZ`. A selection that asks for none
        of them (any catch22 set, or a time-domain/complexity subset)
        can omit it. Note that not *needing* it is not the same as being
        rate-independent; see :func:`extract_catch22`.
    features
        What to compute. ``None`` — the default, and unchanged since
        v0.1.0 — means the eleven native features. Otherwise a
        feature-set name (``"catch22"``, ``"egm_features+catch22"``) or
        an explicit list of feature names. See :mod:`sets`.

    Returns
    -------
    pandas.DataFrame
        N rows, one column per requested feature, in canonical order
        (``docs/theory.md`` §1-§3 then §4) regardless of the order they
        were requested in. With ``features=None``:

        - Time-domain (§1): ``peak_to_peak``, ``zero_crossings``,
          ``activation_position``, ``sec_peak_count``.
        - Frequency (§2): ``spectral_centroid``, ``spectral_entropy``,
          ``dominant_frequency``.
        - Complexity (§3): ``sample_entropy``, ``shannon_entropy``,
          ``lempel_ziv_complexity``, ``higuchi_fractal_dimension``.

    Warns
    -----
    RuntimeWarning
        Once, if any trace produced a ``NaN``, naming the count and the
        features involved. See the module docstring.

    Raises
    ------
    ImportError
        If a requested feature needs the optional ``catch22`` extra and
        it is not installed. Checked **before** any extraction, so a
        long batch fails immediately rather than at the end.
    ValueError
        If ``signals`` is not 2D, a requested name is unknown, or a
        requested feature needs ``fs_hz`` and it was not supplied.
    """
    _validate_signals(signals)

    if features is None:
        # The v0.1.0 default. Deliberately not `sets.resolve("egm_features")`:
        # this path must keep working identically even if the registry changes.
        names: tuple[str, ...] = tuple(_TIME_COLUMNS + _FREQ_COLUMNS + _COMPLEXITY_COLUMNS)
    else:
        names = sets.resolve(features)

    sets.check_available(names)
    return _extract(signals, names=names, fs_hz=_require_fs_hz(fs_hz, names))


def _require_fs_hz(fs_hz: float | None, names: Sequence[str]) -> float:
    """Supply the sample rate, or explain precisely which feature wanted it.

    Names the offending features rather than saying "fs_hz is required":
    with a selection API, "why?" is not obvious, and a caller who asked for
    twelve features should not have to work out which one needs a rate.
    """
    if fs_hz is not None:
        return fs_hz
    needed = sorted(set(names) & REQUIRES_FS_HZ)
    if needed:
        raise ValueError(
            f"fs_hz is required for {needed}: these features return frequencies "
            "in Hz and cannot be computed without a sample rate. Pass "
            "fs_hz=<sample rate in Hz>, or select features that do not need one."
        )
    return _FS_UNUSED


#: :func:`_extract`'s ``fs_hz`` default, used where the requested features
#: provably do not read it. Deliberately NaN rather than a plausible number:
#: if a feature on that path ever became rate-dependent, the result comes back
#: NaN — visibly wrong — instead of silently computed against a made-up 1.0.
_FS_UNUSED = float("nan")


def extract_catch22(
    signals: NDArray[np.floating],
    *,
    catch24: bool = False,
) -> pd.DataFrame:
    """Extract the catch22 set over a batch of traces (``docs/theory.md`` §4).

    Convenience over ``extract_all(..., features="catch22")``.

    **Takes no ``fs_hz``.** Every catch22 feature is defined in samples and
    lags rather than in Hz, so there is nothing to pass. (:class:`providers.
    Catch22Provider` does accept it, because the registry calls every provider
    through one signature and cannot know which needs what — that uniformity
    is a Protocol concern, and this function is catch22-only by construction.)

    Note that not needing ``fs_hz`` is **not** the same as being sample-rate
    independent: ``forecast_error`` looks three samples ahead, which is 3 ms at
    1 kHz and 6 ms at 500 Hz. Values are only comparable between datasets
    recorded at the same rate; the argument is simply not how that constraint
    is expressed. See ``docs/theory.md`` "Preprocessing assumptions".

    Parameters
    ----------
    signals
        ``(N, T)`` float array of N traces.
    catch24
        Also return ``mean`` and ``std_dev`` (§4.11).

    Raises
    ------
    ImportError
        If the optional ``catch22`` extra is not installed.
    """
    _validate_signals(signals)
    names = CATCH24_NAMES if catch24 else CATCH22_NAMES
    sets.check_available(names)
    return _extract(signals, names=names)


def _extract(
    signals: NDArray[np.floating],
    *,
    names: Sequence[str],
    fs_hz: float = _FS_UNUSED,
) -> pd.DataFrame:
    """Run the requested features over every trace, then warn about NaN once.

    ``fs_hz`` defaults to :data:`_FS_UNUSED`, so a caller that provably needs
    no sample rate simply omits it rather than passing a placeholder. Only
    :func:`extract_all`, which cannot know in advance, supplies one.

    Providers are grouped **once**, outside the loop: the grouping depends
    only on the names, and re-deriving it per trace would add work
    proportional to the batch for no reason.
    """
    grouped = sets.group_by_provider(names)

    rows: list[dict[str, float]] = []
    nan_counts: Counter[str] = Counter()
    traces_with_nan = 0

    for signal in signals:
        values: dict[str, float] = {}
        for provider, owned in grouped:
            values.update(provider.extract(signal, fs_hz=fs_hz, features=owned))
        bad = nan_feature_names(values)
        if bad:
            traces_with_nan += 1
            nan_counts.update(bad)
        rows.append(values)

    if traces_with_nan:
        _warn_about_nan(traces_with_nan, len(rows), nan_counts)

    return pd.DataFrame(rows, columns=list(names))


def _warn_about_nan(traces: int, total: int, counts: Counter[str]) -> None:
    """One warning per batch, naming what to go and look at.

    The message leads with the trace count because that is what tells a
    caller whether this is a couple of bad channels or a broken export,
    and lists features by frequency because the worst offender is the
    one worth investigating first.
    """
    worst = ", ".join(f"{name} ({n})" for name, n in counts.most_common(5))
    if len(counts) > 5:
        worst += f", and {len(counts) - 5} more"
    warnings.warn(
        f"{traces} of {total} traces produced NaN features: {worst}. "
        "NaN is propagated, not replaced, so these reach your DataFrame as-is "
        "(docs/theory.md §4.1.4). Usual causes: a constant or near-constant "
        "trace, a dropped channel, or non-finite samples.",
        RuntimeWarning,
        stacklevel=3,
    )
