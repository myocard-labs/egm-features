"""Complexity feature extractors for intracardiac bipolar EGM traces.

Four features that quantify "how complex / unpredictable / fractal" a
trace is from different angles:

- :func:`sample_entropy` — Richman & Moorman 2000 temporal-regularity
  metric. Delegates to ``antropy.sample_entropy``.
- :func:`shannon_entropy` — Shannon entropy of the amplitude histogram.
  Direct numpy / scipy implementation (no antropy).
- :func:`lempel_ziv_complexity` — LZ76 complexity of the median-binarized
  trace. Delegates to ``antropy.lziv_complexity``.
- :func:`higuchi_fractal_dimension` — Higuchi 1988 fractal-dimension
  estimator. Delegates to ``antropy.higuchi_fd``.

See ``docs/theory.md`` §3 for the math, worked examples, and
parameter-choice rationale; see ``project/architecture.md`` "Defaults
policy — math constants vs project-policy values" for the split
between math-default kwargs (``m``, ``r_frac``, ``n_bins``, ``k_max``)
and required policy kwargs (``binarize_method``).

For ``lempel_ziv_complexity``, ``bundle.extract_all`` invokes with the
project-standard ``binarize_method="median"``. Direct callers must
supply ``binarize_method`` explicitly.
"""

from __future__ import annotations

import antropy
import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy as _scipy_entropy

__all__ = [
    "higuchi_fractal_dimension",
    "lempel_ziv_complexity",
    "sample_entropy",
    "shannon_entropy",
]


def sample_entropy(
    signal: NDArray[np.floating],
    *,
    m: int = 2,
    r_frac: float = 0.2,
) -> float:
    """Sample entropy (Richman & Moorman 2000).

    Negative log-probability that two length-``m`` patterns that match
    continue to match when extended by one sample. Lower values indicate
    more regular/predictable signals. See ``docs/theory.md`` §3.1.

    Anchors:

    - Pure sinusoid → ``~0`` (perfectly predictable).
    - White Gaussian noise → ``~2-3`` for ``m=2, r_frac=0.2``.

    Parameters
    ----------
    signal
        1D trace of shape ``(T,)``.
    m
        Embedding dimension (math default ``2``, the Pincus 1991 / Richman
        2000 standard).
    r_frac
        Tolerance fraction; the actual tolerance applied is
        ``r = r_frac * x.std(ddof=0)``. Math default ``0.2`` (Pincus 1991
        standard).

    Returns
    -------
    float
        Sample entropy. Range ``[0, ~3]`` for typical signals; can be
        ``inf`` if no length-``(m+1)`` matches are found (a known
        degenerate case for very short or very regular signals).
    """
    tolerance = r_frac * float(np.std(signal, ddof=0))
    return float(antropy.sample_entropy(signal, order=m, tolerance=tolerance))


def shannon_entropy(
    signal: NDArray[np.floating],
    *,
    n_bins: int = 10,
) -> float:
    """Shannon entropy of the amplitude histogram (in nats).

    Bins ``signal`` into ``n_bins`` equal-width bins between ``min(x)``
    and ``max(x)``, converts the counts to probabilities, and computes
    the Shannon entropy ``-Σ p_k · log(p_k)``. NOT normalized — the
    return value lives in ``[0, log(n_bins)]``. (Contrast with
    :func:`frequency.spectral_entropy`, which IS normalized.)

    See ``docs/theory.md`` §3.2.

    Anchors:

    - Constant signal → ``0`` (all samples in one bin).
    - Uniform-amplitude signal → ``log(n_bins) ≈ 2.303`` for ``n_bins=10``.
    - White Gaussian noise → ``~2.0`` for ``n_bins=10``.

    Parameters
    ----------
    signal
        1D trace of shape ``(T,)``.
    n_bins
        Number of histogram bins (math default ``10``, Sturges' rule
        for ``T=512``).

    Returns
    -------
    float
        Entropy in nats. Range ``[0, log(n_bins)]``.
    """
    counts, _ = np.histogram(signal, bins=n_bins)
    total = int(counts.sum())
    if total == 0:
        # Degenerate: empty signal. Defined as zero entropy.
        return 0.0
    probs = counts / total
    return float(_scipy_entropy(probs))


def lempel_ziv_complexity(
    signal: NDArray[np.floating],
    *,
    binarize_method: str,
) -> float:
    """Normalized Lempel-Ziv 1976 complexity of the binarized trace.

    Binarizes the signal first (using ``binarize_method``), then runs
    the LZ76 distinct-substring count and normalizes by the random-
    sequence asymptote ``T / log_2(T)``. Result is roughly in
    ``(0, 1]``; can slightly exceed 1 for finite sequences due to the
    normalization being asymptotic.

    See ``docs/theory.md`` §3.3 for the algorithm walkthrough.

    Anchors:

    - Periodic sinusoid (alternating binary) → ``< 0.1``.
    - Random binary sequence → ``~1`` (asymptote).

    Parameters
    ----------
    signal
        1D trace of shape ``(T,)``.
    binarize_method
        Required policy parameter (no default per the math-vs-policy
        split in ``project/architecture.md``). Must be one of:

        - ``"median"`` — ``b[i] = 1 if x[i] > median(x), else 0``.
          Project standard, used by ``bundle.extract_all``. Robust to
          baseline drift.
        - ``"zero"`` — ``b[i] = 1 if x[i] > 0, else 0``. Equivalent to
          ``"median"`` for zero-mean bandpassed EGMs.

    Returns
    -------
    float
        Normalized LZ76 complexity. Roughly ``(0, 1]``.

    Raises
    ------
    ValueError
        If ``binarize_method`` is not one of the two supported values.
    """
    if binarize_method == "median":
        threshold = float(np.median(signal))
    elif binarize_method == "zero":
        threshold = 0.0
    else:
        raise ValueError(
            f"Unknown binarize_method {binarize_method!r}; expected 'median' or 'zero'."
        )
    binarized = (signal > threshold).astype(np.int8)
    return float(antropy.lziv_complexity(binarized, normalize=True))


def higuchi_fractal_dimension(
    signal: NDArray[np.floating],
    *,
    k_max: int = 10,
) -> float:
    """Higuchi 1988 fractal-dimension estimate.

    Treats the time series as a curve and estimates its fractal
    dimension via a length-vs-scale linear fit. Lives in ``[1, 2]`` for
    real-valued 1D signals: ``1`` for a smooth curve, ``2`` for
    maximally rough / space-filling.

    See ``docs/theory.md`` §3.4.

    Anchors:

    - Straight line → ``1.0`` (exact).
    - White Gaussian noise → ``~2.0``.

    Parameters
    ----------
    signal
        1D trace of shape ``(T,)``.
    k_max
        Maximum scale for the length-vs-scale fit (math-ish default
        ``10``; literature ranges 8-64).

    Returns
    -------
    float
        Estimated fractal dimension. Range ``[1, 2]`` for typical
        signals.
    """
    return float(antropy.higuchi_fd(signal, kmax=k_max))
