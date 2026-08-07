"""Feature-set registry — which features exist, who computes them, and named groupings.

Pure metadata. Nothing here computes a feature or touches a signal; it answers
"what does `catch22` contain?" and "who owns `trev`?" so that :mod:`bundle`
and downstream consumers do not each maintain their own list.

Two things it provides:

- :data:`FEATURE_REGISTRY` — every feature name mapped to the provider that
  computes it, so a mixed selection can be split across providers and each
  called once.
- :data:`FEATURE_SETS` — the named groupings the project actually uses,
  including the two configurations the STU4 parameter estimator selects between
  (``egm_features`` alone, or ``egm_features+catch22``).

Resolution never needs the optional extra
-----------------------------------------

:func:`resolve` works whether or not ``pycatch22`` is installed, because asking
*what* a set contains is a different question from *computing* it — a caller has
to be able to see what a set would give them before deciding whether installing
a C toolchain is worth it.

The point where the extra becomes necessary is extraction, and
:func:`check_available` is what turns that into the actionable message
(``providers.CATCH22_EXTRA_HINT``) rather than a column of ``NaN`` or a
``KeyError`` three frames down. :mod:`bundle` calls it before extracting.

What belongs here, and what does not
------------------------------------

:data:`FEATURE_SETS` holds only groupings that are **structural or canonical** —
what a provider offers, or a set the literature defines. It deliberately holds
**no study-specific curation**.

That boundary matters because a curated set encodes a judgement that is true for
one analysis and false for the next. "The catch22 features that behave at
``T = 192``" is a real and useful judgement (``docs/theory.md`` §4.1.3 gives the
reasoning and the measurements), but it belongs to the *study*, not to a library
whose charter is arrays in, feature values out — a consumer working at ``T =
512``, or on multi-beat windows, would inherit a set named for their situation
and wrong for it. Libraries here ship no policy defaults; those live in the
executable consumer's config.

So this module tells you **what exists and who computes it**, ``docs/theory.md``
§4 tells you **what is known about each feature**, and the consumer decides
**which to use**. Building a study set is one call:
``resolve(["peak_to_peak", "trev", "entropy_pairs"])``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from myocard_egm_features.catch22 import CATCH22_NAMES, CATCH24_NAMES
from myocard_egm_features.providers import (
    Catch22Provider,
    FeatureProvider,
    NativeProvider,
)

__all__ = [
    "ALL_FEATURE_NAMES",
    "FEATURE_REGISTRY",
    "FEATURE_SETS",
    "PROVIDERS",
    "check_available",
    "group_by_provider",
    "provider_for",
    "resolve",
]

#: One instance per provider, held for the process lifetime — providers are
#: stateless and cheap, and a registry that rebuilt them per call would make
#: identity comparisons useless.
PROVIDERS: tuple[FeatureProvider, ...] = (NativeProvider(), Catch22Provider())


def _build_registry() -> dict[str, FeatureProvider]:
    """Map every feature name to its owning provider, rejecting collisions.

    A collision would mean two providers claiming one name, which silently
    picks a winner and makes results depend on registration order. Better to
    fail at import than to compute the wrong statistic under the right label.
    """
    registry: dict[str, FeatureProvider] = {}
    for provider in PROVIDERS:
        for name in provider.feature_names:
            if name in registry:
                raise RuntimeError(
                    f"feature name {name!r} is claimed by both "
                    f"{registry[name].name!r} and {provider.name!r}."
                )
            registry[name] = provider
    return registry


#: Every feature this library can compute, mapped to its provider.
FEATURE_REGISTRY: dict[str, FeatureProvider] = _build_registry()

#: Canonical order: the native eleven (``docs/theory.md`` §1-§3) followed by
#: the catch22 twenty-four (§4). Every function here returns names in this
#: order regardless of the order they were requested in, so two callers asking
#: for the same set get identically-ordered columns.
ALL_FEATURE_NAMES: tuple[str, ...] = tuple(FEATURE_REGISTRY)

_NATIVE_NAMES: tuple[str, ...] = NativeProvider().feature_names

#: Structural and canonical groupings only — see the module docstring for why
#: study-specific sets are the consumer's to define. ``egm_features`` and
#: ``egm_features+catch22`` are the two configurations the STU4 parameter
#: estimator selects between.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    # The eleven of docs/theory.md §1-§3. The only set that needs no extra.
    "egm_features": _NATIVE_NAMES,
    # The canonical 22 — amplitude-blind by construction (§4.1.1).
    "catch22": CATCH22_NAMES,
    # The 22 plus mean and std_dev, i.e. canonical catch24 (§4.11). Not a
    # default anywhere: peak_to_peak is the clinically-calibrated amplitude
    # statistic, and a bandpassed EGM's mean is near zero by construction.
    "catch24": CATCH24_NAMES,
    # Both families together — morphology and dynamics from catch22, amplitude
    # and activation from the native eleven. The pairing §4.1.1 argues for.
    "egm_features+catch22": _NATIVE_NAMES + CATCH22_NAMES,
}


#: A one-sample array used only to make an unavailable provider raise its own
#: ImportError. Never used for a feature value.
_AVAILABILITY_PROBE = np.zeros(1)


def provider_for(name: str) -> FeatureProvider:
    """The provider that computes ``name``.

    Raises
    ------
    KeyError
        If no provider offers it.
    """
    return FEATURE_REGISTRY[name]


def resolve(selection: str | Sequence[str]) -> tuple[str, ...]:
    """Turn a set name or a list of feature names into canonical-order names.

    Never raises :class:`ImportError` — this is metadata, and it must answer
    even on an install without the optional extra. Use :func:`check_available`
    when you are about to compute.

    Parameters
    ----------
    selection
        A key of :data:`FEATURE_SETS`, or an explicit sequence of feature
        names.

    Returns
    -------
    tuple[str, ...]
        The names, deduplicated and ordered by :data:`ALL_FEATURE_NAMES`.

    Raises
    ------
    ValueError
        If a set name is unknown, or a feature name is not one we compute. The
        message lists what is available, because the most common cause is a
        typo and the second most common is not knowing a set exists.
    """
    if isinstance(selection, str):
        try:
            names: Sequence[str] = FEATURE_SETS[selection]
        except KeyError:
            raise ValueError(
                f"unknown feature set {selection!r}. "
                f"Available sets: {sorted(FEATURE_SETS)}. "
                "To request features by name, pass a list rather than a string."
            ) from None
    else:
        names = selection

    unknown = sorted(set(names) - set(FEATURE_REGISTRY))
    if unknown:
        raise ValueError(
            f"not features this library computes: {unknown}. "
            f"Available features: {sorted(FEATURE_REGISTRY)}. "
            f"Available sets: {sorted(FEATURE_SETS)}."
        )

    wanted = set(names)
    return tuple(name for name in ALL_FEATURE_NAMES if name in wanted)


def group_by_provider(
    names: Sequence[str],
) -> list[tuple[FeatureProvider, list[str]]]:
    """Split names by owning provider, so each provider is called **once**.

    The alternative — looking up and calling per feature — would defeat both
    providers' selection strategies: the native one shares a single periodogram
    across its three frequency features, and calling either provider repeatedly
    repays no setup cost. Returns providers in :data:`PROVIDERS` order, and
    omits any with nothing to do.
    """
    grouped: list[tuple[FeatureProvider, list[str]]] = []
    for provider in PROVIDERS:
        owned = [name for name in names if FEATURE_REGISTRY.get(name) is provider]
        if owned:
            grouped.append((provider, owned))
    return grouped


def check_available(names: Sequence[str]) -> None:
    """Raise if any requested feature's provider cannot run here.

    Called before extraction so a missing optional extra surfaces as the
    actionable install message at the point the user asked for the feature,
    rather than as a ``NaN`` column or an error several frames deeper.

    Raises
    ------
    ImportError
        Naming the unavailable features, with the provider's own install
        instructions attached.
    """
    for provider, owned in group_by_provider(names):
        if provider.available():
            continue
        # Provoke the provider's own error rather than restating its install
        # instructions here: duplicating them would give this module a second
        # copy to drift, and would need updating for every provider added.
        # _AVAILABILITY_PROBE never contributes a value — we are already
        # committed to raising by the time it is used.
        try:
            provider.extract(_AVAILABILITY_PROBE, fs_hz=1.0, features=owned[:1])
        except ImportError as exc:
            raise ImportError(f"{sorted(owned)} unavailable. {exc}") from exc
