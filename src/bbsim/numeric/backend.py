"""Numeric backend protocol for heavy field operations."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from bbsim.core.config import SeedConfig
from bbsim.core.seed import PersonalSeed


class NumericBackend(Protocol):
    """Protocol implemented by numeric backends.

    The first implementation is NumPy-based. A future C++ backend should implement
    the same contract to avoid changes in pipeline stages and UI.
    """

    def create_personal_seed(self, config: SeedConfig) -> tuple[PersonalSeed, np.ndarray]:
        """Create a personal seed and its primordial fluctuation field.

        Args:
            config: Seed generation configuration.

        Returns:
            Pair of seed metadata and a square float32 field.
        """
        ...

    def normalize_field(self, field: np.ndarray) -> np.ndarray:
        """Return a zero-mean, unit-variance copy of a numeric field."""
        ...

    def diffuse(self, field: np.ndarray, amount: float) -> np.ndarray:
        """Return a diffused copy of a numeric field."""
        ...
