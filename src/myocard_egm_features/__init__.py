"""myocard_egm_features — per-trace feature extractors for intracardiac bipolar EGMs.

Two feature families. The **native eleven**, chosen because the clinical
literature names them (``docs/theory.md`` §1-§3):

- :mod:`time_domain` — peak_to_peak, zero_crossings, activation_position, sec_peak_count.
- :mod:`frequency` — spectral_centroid, spectral_entropy, dominant_frequency.
- :mod:`complexity` — sample_entropy, shannon_entropy, lempel_ziv_complexity,
  higuchi_fractal_dimension.

And the **catch22 set** (§4), chosen statistically rather than clinically:

- :mod:`catch22` — the 22 canonical features of Lubba et al. 2019, plus the
  ``catch24`` mean/std pair. Requires the optional extra:
  ``pip install "myocard-egm-features[catch22]"``.

They are complementary, not competing: catch22 runs on the *z-scored* trace and
so cannot see amplitude at all — the axis the ``< 0.5 mV`` scar threshold lives
on — while contributing morphology and dynamics axes the eleven have no
equivalent for. See §4.1.1.

Selecting and running them:

- :mod:`sets` — what exists, who computes it, and the named groupings.
- :mod:`providers` — the seam between a feature *name* and the code for it.
- :mod:`bundle` — :func:`extract_all` runs a selection over a batch of traces
  and returns a tidy pandas DataFrame; only what you ask for is computed.

The base library is NumPy/SciPy/antropy-only — no torch, no HDF5 I/O, no DSP
primitives like bandpass (those live in egm-signal). Callers hand egm-features
an already-bandpassed, calibrated trace and a sample rate; egm-features returns
scalar features.

Each function expects a one-dimensional trace ``signal: NDArray, shape (T,)``.
Batch extraction over an ``(N, T)`` array is the job of :func:`bundle.extract_all`.

See ``docs/theory.md`` for the math behind each feature and the parameter
choices it carries.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

# Single-sourced from pyproject.toml's [project] version so the two can never
# drift. A hardcoded constant here has silently gone stale in sibling repos and
# been stamped into on-disk artifacts as provenance — a believed-but-wrong
# version is worse than none. The fallback covers running from a source tree
# with no installed distribution metadata.
try:
    __version__ = _dist_version("myocard-egm-features")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
]
