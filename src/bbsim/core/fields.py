"""2D field containers used by the simulation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class UniverseFields:
    """Mutable numeric fields for one universe run.

    Arrays are stored as float32 to reduce memory use and make a future C++ backend
    easier to match. Some fields are physical-ish densities; others are visual or
    gameplay diagnostic layers that explain why later objects appear where they do.
    """

    seed_delta: np.ndarray
    inflation_delta: np.ndarray
    dark_density: np.ndarray
    baryon_density: np.ndarray
    radiation: np.ndarray
    cmb: np.ndarray
    gravitational_potential: np.ndarray
    halo_density: np.ndarray
    future_star_sites: np.ndarray
    cold_gas_density: np.ndarray
    molecular_cooling: np.ndarray
    collapse_sites: np.ndarray
    first_star_density: np.ndarray
    stellar_ignition: np.ndarray
    ionized_bubbles: np.ndarray
    stellar_radiation: np.ndarray
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
        gravitational_potential=zero.copy(),
        halo_density=zero.copy(),
        future_star_sites=zero.copy(),
        cold_gas_density=zero.copy(),
        molecular_cooling=zero.copy(),
        collapse_sites=zero.copy(),
        first_star_density=zero.copy(),
        stellar_ignition=zero.copy(),
        ionized_bubbles=zero.copy(),
        stellar_radiation=zero.copy(),
        stars=zero.copy(),
        metals=zero.copy(),
        black_holes=zero.copy(),
        ionization=zero.copy(),
    )
