"""Pytest fixtures shared across the package's test suite.

Conventions:

- Pure-Python fixtures live here. Synthetic signals with known feature values
  (pure sinusoid, white noise, biphasic activation analog) get added per-module
  as the feature implementations land.
- No real data files in the repo.

Currently empty; fixtures get added in Block 4 (per-module implementation)
as each feature gets its known-correct synthetic test signals.
"""

from __future__ import annotations
