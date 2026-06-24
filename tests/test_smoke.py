"""Smoke test — confirms the package imports and ``__version__`` is sane.

Keep this test alive for the life of the package; it's the canary for
broken installs and bad packaging metadata.
"""

from __future__ import annotations

import re

import myocard_egm_features


def test_package_imports() -> None:
    assert myocard_egm_features is not None


def test_version_is_pep440() -> None:
    # Loose PEP 440 check — accepts X.Y.Z, X.Y.Z.devN, X.Y.ZrcN, etc.
    assert re.match(r"^\d+\.\d+\.\d+", myocard_egm_features.__version__), (
        f"non-PEP440 version: {myocard_egm_features.__version__!r}"
    )
