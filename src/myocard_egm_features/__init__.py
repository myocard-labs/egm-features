"""myocard_egm_features — per-trace feature extractors for intracardiac bipolar EGMs.

Three feature families, one bundle helper:

- :mod:`time_domain` — peak_to_peak, zero_crossings, activation_position, sec_peak_count.
- :mod:`frequency` — spectral_centroid, spectral_entropy, dominant_frequency.
- :mod:`complexity` — sample_entropy, shannon_entropy, lempel_ziv_complexity,
  higuchi_fractal_dimension.
- :mod:`bundle` — :func:`extract_all` runs every feature over a batch of traces
  and returns a tidy pandas DataFrame.

The library is NumPy/SciPy/antropy-only — no torch, no HDF5 I/O, no DSP
primitives like bandpass (those live in egm-signal). Callers hand egm-features
an already-bandpassed, calibrated trace and a sample rate; egm-features returns
scalar features.

Each function expects a one-dimensional trace ``signal: NDArray, shape (T,)``.
Batch extraction over an ``(N, T)`` array is the job of :func:`bundle.extract_all`.

See ``docs/theory.md`` for the math behind each feature and the parameter
choices it carries.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
]
