"""Simplified Friedmann-like expansion helpers."""

from __future__ import annotations

import math

from bbsim.core.config import CosmologyConfig


def compute_hubble_gyr_inv(a: float, cosmology: CosmologyConfig) -> float:
    """Compute the simplified Hubble parameter for scale factor `a`.

    Args:
        a: Positive scale factor.
        cosmology: Cosmological density parameters.

    Returns:
        Hubble parameter in inverse gigayears.

    Raises:
        ValueError: If `a` is not positive.
    """

    if a <= 0.0:
        raise ValueError("scale factor must be positive")

    omega_m = cosmology.omega_b + cosmology.omega_dm
    density_term = (
        cosmology.omega_r / a**4
        + omega_m / a**3
        + cosmology.omega_k / a**2
        + cosmology.omega_lambda
    )
    return cosmology.h0_gyr_inv * math.sqrt(max(density_term, 0.0))


def detect_era(a: float, cosmology: CosmologyConfig) -> str:
    """Detect the dominant component era for the simplified model.

    Args:
        a: Positive scale factor.
        cosmology: Cosmological density parameters.

    Returns:
        Era identifier: `radiation`, `matter`, or `dark_energy`.
    """

    if a <= 0.0:
        raise ValueError("scale factor must be positive")

    omega_m = cosmology.omega_b + cosmology.omega_dm
    rho_r = cosmology.omega_r / a**4
    rho_m = omega_m / a**3
    rho_l = cosmology.omega_lambda

    if rho_r > rho_m and rho_r > rho_l:
        return "radiation"
    if rho_m > rho_l:
        return "matter"
    return "dark_energy"
