"""2D field containers used by the simulation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class UniverseFields:
    """Mutable numeric fields for one universe run.

    Arrays are stored as float32 to reduce memory use and make a future C++ backend
    easier to match.
    """

    seed_delta: np.ndarray
    inflation_delta: np.ndarray
    dark_density: np.ndarray
    baryon_density: np.ndarray
    radiation: np.ndarray
    cmb: np.ndarray
    stars: np.ndarray
    metals: np.ndarray
    black_holes: np.ndarray
    ionization: np.ndarray


def create_empty_fields(grid_size: int) -> UniverseFields:
    """Create zero-initialized fields for a square universe grid.

    Args:
        grid_size: Width and height of the square field grid.

    Returns:
        New mutable universe fields.
    """

    shape = (grid_size, grid_size)
    zero = np.zeros(shape, dtype=np.float32)
    return UniverseFields(
        seed_delta=zero.copy(),
        inflation_delta=zero.copy(),
        dark_density=zero.copy(),
        baryon_density=zero.copy(),
        radiation=zero.copy(),
        cmb=zero.copy(),
        stars=zero.copy(),
        metals=zero.copy(),
        black_holes=zero.copy(),
        ionization=zero.copy(),
    )
