"""Tests for the feature-set registry.

The registry is pure metadata, so most of these are cheap. The ones that earn
their place are the membership assertions — a set silently gaining or losing a
feature changes what a downstream comparison measures, and nothing else in the
library would notice.
"""

from __future__ import annotations

import builtins

import pytest

from myocard_egm_features import sets
from myocard_egm_features.catch22 import CATCH22_NAMES, CATCH24_NAMES
from myocard_egm_features.providers import NativeProvider
from myocard_egm_features.sets import (
    ALL_FEATURE_NAMES,
    FEATURE_REGISTRY,
    FEATURE_SETS,
    check_available,
    group_by_provider,
    provider_for,
    resolve,
)


def _without_pycatch22(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the optional extra unimportable, as on a base install."""
    real_import = builtins.__import__

    def _blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "pycatch22":
            raise ImportError("No module named 'pycatch22'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _blocked)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_every_feature_has_exactly_one_provider() -> None:
    assert len(FEATURE_REGISTRY) == 11 + 24
    assert len(ALL_FEATURE_NAMES) == len(set(ALL_FEATURE_NAMES))


def test_canonical_order_is_native_then_catch22() -> None:
    """Two callers asking for the same set must get identically-ordered columns."""
    assert ALL_FEATURE_NAMES[:11] == NativeProvider().feature_names
    assert ALL_FEATURE_NAMES[11:] == CATCH24_NAMES


def test_provider_lookup() -> None:
    assert provider_for("peak_to_peak").name == "egm_features"
    assert provider_for("trev").name == "catch22"
    with pytest.raises(KeyError):
        provider_for("not_a_feature")


# ---------------------------------------------------------------------------
# Set membership — the assertions that actually matter
# ---------------------------------------------------------------------------


def test_set_sizes() -> None:
    assert len(FEATURE_SETS["egm_features"]) == 11
    assert len(FEATURE_SETS["catch22"]) == 22
    assert len(FEATURE_SETS["catch24"]) == 24
    assert len(FEATURE_SETS["egm_features+catch22"]) == 33


def test_no_study_specific_sets_are_shipped() -> None:
    """The registry holds structural and canonical groupings only.

    A curated set — "the features that behave at T = 192", say — encodes a
    judgement true for one analysis and false for the next, and this library
    has no way to know which situation a consumer is in. Such sets belong to
    the consumer's config; docs/theory.md §4.1.3 carries the reasoning a
    consumer would use to build one. Pinned as a test because adding a
    convenience set here is exactly the temptation the boundary guards against.
    """
    assert set(FEATURE_SETS) == {
        "egm_features",  # what the native provider offers
        "catch22",  # canonical, Lubba et al. 2019
        "catch24",  # canonical, plus the amplitude pair
        "egm_features+catch22",  # the structural union of both providers
    }


def test_a_study_set_is_one_call() -> None:
    """The replacement for a shipped curated set, and the thing consumers do."""
    assert resolve(["peak_to_peak", "trev", "entropy_pairs"]) == (
        "peak_to_peak",
        "trev",
        "entropy_pairs",
    )


def test_catch24_is_catch22_plus_the_amplitude_pair() -> None:
    assert set(FEATURE_SETS["catch24"]) - set(FEATURE_SETS["catch22"]) == {
        "mean",
        "std_dev",
    }


def test_the_amplitude_pair_is_in_no_other_set() -> None:
    """D9: registered so canonical catch24 is reachable, but not a default.

    peak_to_peak is the clinically-calibrated amplitude statistic, and a
    bandpassed EGM's mean is near zero by construction.
    """
    for name, members in FEATURE_SETS.items():
        if name == "catch24":
            continue
        assert "mean" not in members and "std_dev" not in members, name


def test_combined_set_is_the_two_families() -> None:
    combined = FEATURE_SETS["egm_features+catch22"]
    assert set(combined) == set(FEATURE_SETS["egm_features"]) | set(CATCH22_NAMES)
    assert len(combined) == len(set(combined))


def test_every_set_member_is_a_registered_feature() -> None:
    for name, members in FEATURE_SETS.items():
        unknown = set(members) - set(FEATURE_REGISTRY)
        assert not unknown, f"{name} references unknown {sorted(unknown)}"


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------


def test_resolve_a_set_name() -> None:
    assert resolve("catch22") == CATCH22_NAMES


def test_resolve_normalises_order_and_duplicates() -> None:
    assert resolve(["trev", "peak_to_peak", "trev", "mode_5"]) == (
        "peak_to_peak",
        "mode_5",
        "trev",
    )


def test_resolve_rejects_an_unknown_set_with_a_useful_message() -> None:
    with pytest.raises(ValueError) as excinfo:
        resolve("catch_22")
    message = str(excinfo.value)
    assert "unknown feature set" in message
    assert "catch22" in message  # lists what does exist
    assert "pass a list" in message  # names the other calling convention


def test_resolve_rejects_an_unknown_feature() -> None:
    with pytest.raises(ValueError, match="not features this library computes"):
        resolve(["peak_to_peak", "trevv"])


def test_resolve_needs_no_optional_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Metadata must answer on a base install.

    Asking what a set contains is how someone decides whether installing a C
    toolchain is worth it — it cannot require the toolchain.
    """
    _without_pycatch22(monkeypatch)
    assert resolve("catch22") == FEATURE_SETS["catch22"]
    assert len(resolve("egm_features+catch22")) == 33


# ---------------------------------------------------------------------------
# group_by_provider
# ---------------------------------------------------------------------------


def test_grouping_splits_a_mixed_selection_one_bucket_per_provider() -> None:
    grouped = group_by_provider(resolve(["peak_to_peak", "trev", "sec_peak_count"]))
    assert [(p.name, names) for p, names in grouped] == [
        ("egm_features", ["peak_to_peak", "sec_peak_count"]),
        ("catch22", ["trev"]),
    ]


def test_grouping_omits_providers_with_nothing_to_do() -> None:
    grouped = group_by_provider(resolve(["peak_to_peak"]))
    assert [p.name for p, _ in grouped] == ["egm_features"]


def test_grouping_covers_every_requested_name() -> None:
    names = resolve("egm_features+catch22")
    covered = [n for _, owned in group_by_provider(names) for n in owned]
    assert sorted(covered) == sorted(names)


# ---------------------------------------------------------------------------
# check_available
# ---------------------------------------------------------------------------


def test_check_available_passes_for_native_only_on_a_base_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_pycatch22(monkeypatch)
    check_available(resolve("egm_features"))  # must not raise


def test_check_available_raises_the_install_message_for_catch22(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point of the function: an actionable error, not a NaN column.

    The install instructions are the provider's, deliberately not restated
    here — so this asserts they arrive, not their exact wording.
    """
    _without_pycatch22(monkeypatch)
    with pytest.raises(ImportError) as excinfo:
        check_available(resolve("catch22"))
    message = str(excinfo.value)
    assert 'pip install "myocard-egm-features[catch22]"' in message
    assert "trev" in message  # names which features are unavailable
    assert isinstance(excinfo.value.__cause__, ImportError)


@pytest.mark.skipif(
    not sets.PROVIDERS[1].available(), reason="optional catch22 extra not installed"
)
def test_check_available_passes_when_the_extra_is_installed() -> None:
    check_available(resolve("egm_features+catch22"))  # must not raise
