"""Tests for myocard_egm_features.bundle.

Per ``project/architecture.md`` "Tests" section, the bundle is tested
for: (a) output shape + column names, (b) per-row consistency with the
individual extractors (i.e. the bundle's hardcoded policy values
actually flow through and the math isn't accidentally re-derived).

Math anchors are covered by the per-feature tests; here we focus on
plumbing.

Batches are kept small (N=3) because complexity features (sample_entropy
especially) are O(T²) and dominate runtime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from myocard_egm_features import bundle, complexity, frequency, time_domain

FS_HZ = 1000.0
T_SAMPLES = 512


def _batch(seed: int = 0, n: int = 3) -> np.ndarray:
    """Reproducible ``(N, T)`` batch of synthetic traces with varied character."""
    rng = np.random.default_rng(seed)
    t = np.arange(T_SAMPLES, dtype=np.float64) / FS_HZ
    rows = [
        # Pure 50 Hz sine.
        np.sin(2 * np.pi * 50 * t),
        # Two-component sum (50 Hz + 150 Hz, unequal amplitudes).
        np.sin(2 * np.pi * 50 * t) + 2.0 * np.sin(2 * np.pi * 150 * t),
        # White noise.
        rng.standard_normal(T_SAMPLES),
    ]
    return np.stack(rows[:n], axis=0)


# ---------------------------------------------------------------------------
# Shape + column-name contracts
# ---------------------------------------------------------------------------


def test_extract_time_domain_shape_and_columns() -> None:
    """N rows × 4 columns in the documented order."""
    df = bundle.extract_time_domain(_batch())
    assert df.shape == (3, 4)
    assert list(df.columns) == [
        "peak_to_peak",
        "zero_crossings",
        "activation_position",
        "sec_peak_count",
    ]


def test_extract_frequency_shape_and_columns() -> None:
    """N rows × 3 columns in the documented order."""
    df = bundle.extract_frequency(_batch(), fs_hz=FS_HZ)
    assert df.shape == (3, 3)
    assert list(df.columns) == [
        "spectral_centroid",
        "spectral_entropy",
        "dominant_frequency",
    ]


def test_extract_complexity_shape_and_columns() -> None:
    """N rows × 4 columns in the documented order."""
    df = bundle.extract_complexity(_batch())
    assert df.shape == (3, 4)
    assert list(df.columns) == [
        "sample_entropy",
        "shannon_entropy",
        "lempel_ziv_complexity",
        "higuchi_fractal_dimension",
    ]


def test_extract_all_shape_and_columns() -> None:
    """N rows × 11 columns in theory.md §1 → §2 → §3 order."""
    df = bundle.extract_all(_batch(), fs_hz=FS_HZ)
    assert df.shape == (3, 11)
    assert list(df.columns) == [
        # §1 time
        "peak_to_peak",
        "zero_crossings",
        "activation_position",
        "sec_peak_count",
        # §2 frequency
        "spectral_centroid",
        "spectral_entropy",
        "dominant_frequency",
        # §3 complexity
        "sample_entropy",
        "shannon_entropy",
        "lempel_ziv_complexity",
        "higuchi_fractal_dimension",
    ]


# ---------------------------------------------------------------------------
# Per-row consistency with individual extractors
# ---------------------------------------------------------------------------


def test_extract_time_domain_matches_individual_calls() -> None:
    """Each cell equals the corresponding per-feature call with bundle policy.

    Pins that the bundle's hardcoded :data:`ACTIVATION_METHOD` and
    :data:`SEC_PEAK_THRESHOLD_FRAC` are actually threaded through —
    not silently overridden somewhere.
    """
    signals = _batch()
    df = bundle.extract_time_domain(signals)
    for i, x in enumerate(signals):
        assert df.iloc[i]["peak_to_peak"] == time_domain.peak_to_peak(x)
        assert df.iloc[i]["zero_crossings"] == time_domain.zero_crossings(x)
        assert df.iloc[i]["activation_position"] == time_domain.activation_position(
            x, method=bundle.ACTIVATION_METHOD
        )
        assert df.iloc[i]["sec_peak_count"] == time_domain.sec_peak_count(
            x, threshold_frac=bundle.SEC_PEAK_THRESHOLD_FRAC
        )


def test_extract_frequency_matches_individual_calls() -> None:
    """Frequency cells equal the per-feature calls; the PSD-reuse path is
    bit-exact equivalent to the recompute path (already pinned by
    test_frequency.test_precomputed_psd_matches_recomputed)."""
    signals = _batch()
    df = bundle.extract_frequency(signals, fs_hz=FS_HZ)
    for i, x in enumerate(signals):
        assert df.iloc[i]["spectral_centroid"] == frequency.spectral_centroid(x, fs_hz=FS_HZ)
        assert df.iloc[i]["spectral_entropy"] == frequency.spectral_entropy(x, fs_hz=FS_HZ)
        assert df.iloc[i]["dominant_frequency"] == frequency.dominant_frequency(x, fs_hz=FS_HZ)


def test_extract_complexity_matches_individual_calls() -> None:
    """Each cell equals the corresponding per-feature call with bundle policy.

    Pins that :data:`LZ_BINARIZE_METHOD` and the four math-default
    constants flow through to the individual calls.
    """
    signals = _batch()
    df = bundle.extract_complexity(signals)
    for i, x in enumerate(signals):
        assert df.iloc[i]["sample_entropy"] == complexity.sample_entropy(
            x, m=bundle.SAMPLE_ENTROPY_M, r_frac=bundle.SAMPLE_ENTROPY_R_FRAC
        )
        assert df.iloc[i]["shannon_entropy"] == complexity.shannon_entropy(
            x, n_bins=bundle.SHANNON_ENTROPY_N_BINS
        )
        assert df.iloc[i]["lempel_ziv_complexity"] == complexity.lempel_ziv_complexity(
            x, binarize_method=bundle.LZ_BINARIZE_METHOD
        )
        assert df.iloc[i]["higuchi_fractal_dimension"] == complexity.higuchi_fractal_dimension(
            x, k_max=bundle.HIGUCHI_K_MAX
        )


def test_extract_all_equals_concatenated_per_module_calls() -> None:
    """extract_all is just the three per-module helpers concatenated."""
    signals = _batch()
    df_all = bundle.extract_all(signals, fs_hz=FS_HZ)
    df_concat = pd.concat(
        [
            bundle.extract_time_domain(signals),
            bundle.extract_frequency(signals, fs_hz=FS_HZ),
            bundle.extract_complexity(signals),
        ],
        axis=1,
    )
    pd.testing.assert_frame_equal(df_all, df_concat)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        bundle.extract_time_domain,
        lambda s: bundle.extract_frequency(s, fs_hz=FS_HZ),
        bundle.extract_complexity,
        lambda s: bundle.extract_all(s, fs_hz=FS_HZ),
    ],
    ids=["time_domain", "frequency", "complexity", "all"],
)
def test_1d_input_raises_with_helpful_message(fn: object) -> None:
    """Passing a 1D ``(T,)`` array (common mistake) raises ValueError.

    The error message points the caller to the ``[np.newaxis]`` fix so
    they don't have to dig through the docs.
    """
    x = np.zeros(T_SAMPLES, dtype=np.float64)
    with pytest.raises(ValueError, match="2D array"):
        fn(x)  # type: ignore[operator]


@pytest.mark.parametrize(
    "fn",
    [
        bundle.extract_time_domain,
        lambda s: bundle.extract_frequency(s, fs_hz=FS_HZ),
        bundle.extract_complexity,
        lambda s: bundle.extract_all(s, fs_hz=FS_HZ),
    ],
    ids=["time_domain", "frequency", "complexity", "all"],
)
def test_3d_input_raises(fn: object) -> None:
    """Passing a 3D array (multi-channel batch, not yet supported) raises."""
    x = np.zeros((2, 3, T_SAMPLES), dtype=np.float64)
    with pytest.raises(ValueError, match="2D array"):
        fn(x)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Policy-value constants are wired (visible from the module top level)
# ---------------------------------------------------------------------------


def test_policy_constants_match_canonical_values() -> None:
    """Anchor the project-standard policy values in one spot.

    If any of these gets accidentally changed, this test pins where to
    look. See ``docs/theory.md`` for the justifications.
    """
    assert bundle.ACTIVATION_METHOD == "dvdt_max"
    assert bundle.SEC_PEAK_THRESHOLD_FRAC == 0.3
    assert bundle.LZ_BINARIZE_METHOD == "median"
    assert bundle.SAMPLE_ENTROPY_M == 2
    assert bundle.SAMPLE_ENTROPY_R_FRAC == 0.2
    assert bundle.SHANNON_ENTROPY_N_BINS == 10
    assert bundle.HIGUCHI_K_MAX == 10
