"""Feature providers — the seam between a feature *name* and the code that computes it.

A provider answers three questions about a group of features: what are they
called, can they run here, and what are their values for this trace. That is
the whole interface.

Why the indirection exists
--------------------------

The library computes two kinds of feature that differ in ways a caller should
not have to care about:

- the **native** eleven (``time_domain`` / ``frequency`` / ``complexity``),
  pure numpy+scipy+antropy, always installed, several of them carrying
  project-standard policy arguments; and
- the **catch22** twenty-two, which delegate to a C extension shipped as an
  optional extra and may not be installed at all.

Without a seam, every consumer that wants "these eight features" has to know
which camp each name belongs to, whether the optional dependency is present,
and which policy arguments to pass. With one, the feature-set registry can
resolve a name to a provider and call it, and the day a toolchain-less target
makes ``pycatch22`` unusable, the fix is one new provider class rather than a
change to every call site.

Uniform signature, and why catch22 takes ``fs_hz`` it does not use
------------------------------------------------------------------

:meth:`FeatureProvider.extract` always takes ``fs_hz``. The native frequency
features need it; the catch22 features do not, because they are *defined* in
samples and lags rather than in Hz.

That is not the same as being sample-rate independent. ``forecast_error``
forecasts three samples ahead, which is 3 ms at 1 kHz and 6 ms at 500 Hz, and
the autocorrelation-timescale features measure lags in samples. Their values
are only comparable between two datasets recorded at the same rate — the
argument is simply not how that constraint is expressed. See
``docs/theory.md`` "Preprocessing assumptions".

Accepting the unused argument keeps the registry free of per-provider
branching, which is the point of the seam.

Where the policy constants live
-------------------------------

The project-standard policy values (``ACTIVATION_METHOD`` and friends) are
defined *here*, not in :mod:`bundle`, because the native provider is what
applies them. :mod:`bundle` re-exports them under their original names, so
they remain part of the public API exactly as before. The import direction is
one-way — ``bundle`` imports ``providers``, never the reverse — which is what
keeps the registry (which also imports ``providers``) out of an import cycle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from myocard_egm_features import complexity, frequency, time_domain
from myocard_egm_features.catch22 import (
    CATCH22_EXTRA_HINT,
    CATCH24_NAMES,
    catch22_features,
    pycatch22_available,
    require_pycatch22,
)

__all__ = [
    "ACTIVATION_METHOD",
    "CATCH22_EXTRA_HINT",
    "HIGUCHI_K_MAX",
    "LZ_BINARIZE_METHOD",
    "REQUIRES_FS_HZ",
    "SAMPLE_ENTROPY_M",
    "SAMPLE_ENTROPY_R_FRAC",
    "SEC_PEAK_THRESHOLD_FRAC",
    "SHANNON_ENTROPY_N_BINS",
    "Catch22Provider",
    "FeatureProvider",
    "NativeProvider",
    "nan_feature_names",
    "pycatch22_available",
    "require_pycatch22",
    "validate_selection",
]

#: The features that need the ``fs_hz`` **argument** to be computed at all —
#: the three native ones that read the power spectrum, whose values are in Hz.
#: Grouped so the shared periodogram is computed only when one is requested,
#: and exposed so a caller can tell whether it needs a sample rate before
#: supplying one.
#:
#: **Not the same as "the rate-sensitive features."** Every catch22 feature is
#: defined in samples and lags, so none appears here — but ``forecast_error``
#: looks three samples ahead, which is 3 ms at 1 kHz and 6 ms at 500 Hz, and
#: the autocorrelation timescales measure lags in samples. Those values are
#: still only comparable between datasets recorded at one rate; they simply do
#: not take the argument. See ``docs/theory.md`` "Preprocessing assumptions".
REQUIRES_FS_HZ: frozenset[str] = frozenset(
    {"spectral_centroid", "spectral_entropy", "dominant_frequency"}
)

_FREQUENCY_NAMES = REQUIRES_FS_HZ


def validate_selection(
    features: Sequence[str] | None,
    offered: Sequence[str],
    provider_name: str,
) -> frozenset[str]:
    """Normalise a ``features=`` argument, rejecting anything not on offer.

    ``None`` means "everything". An unknown name is an error rather than a
    silently-dropped column, because a typo in a feature set would otherwise
    produce a quietly narrower comparison instead of a failure.
    """
    if features is None:
        return frozenset(offered)
    wanted = frozenset(features)
    unknown = sorted(wanted - frozenset(offered))
    if unknown:
        raise ValueError(
            f"{provider_name} does not provide {unknown}. Available: {sorted(offered)}"
        )
    return wanted


# ---------------------------------------------------------------------------
# Project-standard policy values — see docs/theory.md and architecture.md
# ("Defaults policy — math constants vs project-policy values"). These are the
# arguments the native provider applies so that every consumer computes the
# same statistic; a direct caller of an individual extractor still has to pass
# them explicitly, which is what prevents silent cross-consumer drift.
# ---------------------------------------------------------------------------

#: Activation-time convention for :func:`time_domain.activation_position`.
#: Clinical default (Marchlinski / Wittkampf school); see ``docs/theory.md`` §1.3.
ACTIVATION_METHOD: str = "dvdt_max"

#: Prominence threshold (fraction of primary peak amplitude) for
#: :func:`time_domain.sec_peak_count`. See ``docs/theory.md`` §1.4.
SEC_PEAK_THRESHOLD_FRAC: float = 0.3

#: Binarization rule for :func:`complexity.lempel_ziv_complexity`. Robust to
#: baseline drift; see ``docs/theory.md`` §3.3.
LZ_BINARIZE_METHOD: str = "median"

#: Embedding dimension for :func:`complexity.sample_entropy`. Standard
#: Pincus 1991 / Richman 2000 choice; see ``docs/theory.md`` §3.1.
SAMPLE_ENTROPY_M: int = 2

#: Tolerance fraction for :func:`complexity.sample_entropy`. Standard
#: Pincus 1991 choice; see ``docs/theory.md`` §3.1.
SAMPLE_ENTROPY_R_FRAC: float = 0.2

#: Histogram bin count for :func:`complexity.shannon_entropy`. Sturges' rule;
#: see ``docs/theory.md`` §3.2 (and its note on T = 192 vs 512).
SHANNON_ENTROPY_N_BINS: int = 10

#: Maximum scale for :func:`complexity.higuchi_fractal_dimension`. Pragmatic
#: literature mid-range; see ``docs/theory.md`` §3.4.
HIGUCHI_K_MAX: int = 10


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


@runtime_checkable
class FeatureProvider(Protocol):
    """One family of features that can be named, checked, and computed.

    Implementations are expected to be cheap to construct and stateless, so a
    registry can hold one instance per provider for the process lifetime.
    """

    #: Stable identifier, used in registry keys and error messages.
    name: str

    #: Every feature this provider can compute, in documentation order. Stable
    #: across calls, and available even when :meth:`available` is ``False`` —
    #: a caller must be able to ask what *would* be computed before deciding
    #: whether to install anything.
    feature_names: tuple[str, ...]

    def available(self) -> bool:
        """Whether :meth:`extract` can run in this environment."""
        ...

    def extract(
        self,
        signal: NDArray[np.floating],
        *,
        fs_hz: float,
        features: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Compute features for one trace.

        Parameters
        ----------
        signal
            1D trace of shape ``(T,)``.
        fs_hz
            Sample rate in Hz. Required by the signature even for providers
            that do not use it — see the module docstring.
        features
            Which features to compute. ``None`` means all of
            :attr:`feature_names`. A provider **must not** compute what was not
            asked for when skipping is cheaper — see the note below.

        Returns
        -------
        dict[str, float]
            One entry per requested name, in :attr:`feature_names` order. Values
            may be ``NaN`` for degenerate input; see ``docs/theory.md`` §4.1.4.

        Raises
        ------
        ValueError
            If ``features`` names anything this provider does not offer.

        Notes
        -----
        **Selection is a performance contract, and the right way to honour it
        differs per provider.** The native features are independent Python
        calls, so computing only what was asked is a large saving — at
        ``T = 192`` all eleven cost ~0.68 ms per trace, of which a caller
        wanting one cheap feature was previously paying 216x more than
        necessary. The catch22 features are the opposite: one C call returns
        all 22 at once, so the cheapest way to serve a subset is to compute
        everything and filter. Each provider decides; callers just ask.

        Notes
        -----
        ``float`` here is the PEP 484 numeric tower, not a runtime guarantee.
        Two native features are **counts** and return a genuine ``int``
        (``zero_crossings``, ``sec_peak_count``); the bundle turns those into
        integer DataFrame columns, and it has done so since v0.1.0. Coercing
        them to ``float`` in a provider would silently change those column
        dtypes for every existing consumer, so providers return the natural
        type and the contract is "a real number", not "a Python float".
        """
        ...


class NativeProvider:
    """The eleven features of ``docs/theory.md`` §1-§3.

    Always available: numpy, scipy, pandas, and antropy are base dependencies.
    Applies the project-standard policy constants defined above, so a caller
    going through this provider gets the same values as ``bundle.extract_all``.
    """

    name = "egm_features"

    #: Documentation order (theory.md §1 → §2 → §3), which is also the column
    #: order the bundle has always produced.
    feature_names: tuple[str, ...] = (
        "peak_to_peak",
        "zero_crossings",
        "activation_position",
        "sec_peak_count",
        "spectral_centroid",
        "spectral_entropy",
        "dominant_frequency",
        "sample_entropy",
        "shannon_entropy",
        "lempel_ziv_complexity",
        "higuchi_fractal_dimension",
    )

    def available(self) -> bool:
        """Always ``True`` — these features have no optional dependency."""
        return True

    def extract(
        self,
        signal: NDArray[np.floating],
        *,
        fs_hz: float,
        features: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Compute the requested native features for one trace.

        Computes **only** what was asked for. The eleven are independent calls,
        so a caller wanting three pays for three — see the note on
        :meth:`FeatureProvider.extract` for why the catch22 provider does the
        opposite.

        The periodogram is shared across the three frequency features and is
        computed **only if at least one of them is requested**, matching
        ``bundle.extract_frequency``'s optimisation. At ``T = 192`` it costs
        0.14 ms per trace, so skipping it matters for a purely time-domain or
        complexity selection.
        """
        wanted = validate_selection(features, self.feature_names, self.name)
        out: dict[str, float] = {}
        if not wanted:
            return out

        if wanted & _FREQUENCY_NAMES:
            freqs, psd = frequency.periodogram(signal, fs_hz=fs_hz)
        if "peak_to_peak" in wanted:
            out["peak_to_peak"] = time_domain.peak_to_peak(signal)
        if "zero_crossings" in wanted:
            out["zero_crossings"] = time_domain.zero_crossings(signal)
        if "activation_position" in wanted:
            out["activation_position"] = time_domain.activation_position(
                signal, method=ACTIVATION_METHOD
            )
        if "sec_peak_count" in wanted:
            out["sec_peak_count"] = time_domain.sec_peak_count(
                signal, threshold_frac=SEC_PEAK_THRESHOLD_FRAC
            )
        if "spectral_centroid" in wanted:
            out["spectral_centroid"] = frequency.spectral_centroid(
                signal, fs_hz=fs_hz, freqs=freqs, psd=psd
            )
        if "spectral_entropy" in wanted:
            out["spectral_entropy"] = frequency.spectral_entropy(
                signal, fs_hz=fs_hz, freqs=freqs, psd=psd
            )
        if "dominant_frequency" in wanted:
            out["dominant_frequency"] = frequency.dominant_frequency(
                signal, fs_hz=fs_hz, freqs=freqs, psd=psd
            )
        if "sample_entropy" in wanted:
            out["sample_entropy"] = complexity.sample_entropy(
                signal, m=SAMPLE_ENTROPY_M, r_frac=SAMPLE_ENTROPY_R_FRAC
            )
        if "shannon_entropy" in wanted:
            out["shannon_entropy"] = complexity.shannon_entropy(
                signal, n_bins=SHANNON_ENTROPY_N_BINS
            )
        if "lempel_ziv_complexity" in wanted:
            out["lempel_ziv_complexity"] = complexity.lempel_ziv_complexity(
                signal, binarize_method=LZ_BINARIZE_METHOD
            )
        if "higuchi_fractal_dimension" in wanted:
            out["higuchi_fractal_dimension"] = complexity.higuchi_fractal_dimension(
                signal, k_max=HIGUCHI_K_MAX
            )
        # Documentation order, restricted to what was asked for.
        return {name: out[name] for name in self.feature_names if name in out}


class Catch22Provider:
    """The 22 catch22 features plus the catch24 pair — ``docs/theory.md`` §4.

    Available only when the optional ``catch22`` extra is installed. Offers all
    24 names; the project's *default* feature sets exclude ``mean`` and
    ``std_dev``, because ``peak_to_peak`` (§1.1) is the clinically-calibrated
    amplitude statistic and a bandpassed EGM's mean is near zero by
    construction — but a caller wanting canonical catch24 can ask for them.
    """

    name = "catch22"

    #: All 24, in ``docs/theory.md`` §4 documentation order.
    feature_names: tuple[str, ...] = CATCH24_NAMES

    def available(self) -> bool:
        """Whether the optional extra is installed and importable."""
        return pycatch22_available()

    def extract(
        self,
        signal: NDArray[np.floating],
        *,
        fs_hz: float,
        features: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Compute the requested catch22 features for one trace.

        Computes **only** what was asked for, the same as
        :class:`NativeProvider`. ``pycatch22`` exposes each feature as its own
        entry point, and its bundled ``catch22_all`` carries no discount — it
        is a loop over those same functions. Measured at ``T = 192``: 22
        bundled cost 0.373 ms, 22 individually cost 0.368 ms, one feature costs
        0.003 ms. Selection is therefore a straight win at every size, with no
        crossover to reason about.

        ``fs_hz`` is accepted and unused — these features are defined in
        samples and lags, not Hz. That does *not* make them sample-rate
        independent; see the module docstring.
        """
        wanted = validate_selection(features, self.feature_names, self.name)
        if not wanted:
            return {}
        return catch22_features(signal, [n for n in self.feature_names if n in wanted])


def nan_feature_names(values: Mapping[str, float]) -> tuple[str, ...]:
    """Which entries came back ``NaN``, in the order given.

    The detection half of the warn-on-degenerate-input policy
    (``docs/theory.md`` §4.1.4). It lives here rather than in :mod:`catch22`
    because the native features can produce ``NaN`` too, and a batch caller
    wants one answer across whatever providers it used.

    Deliberately does **not** warn. A per-trace warning would fire thousands of
    times on a bad channel and slow the loop it is trying to report on; the
    batch layer aggregates these into a single message naming how many traces
    and which features were affected.
    """
    return tuple(name for name, value in values.items() if value != value)
