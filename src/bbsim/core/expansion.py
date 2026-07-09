"""Simplified Friedmann-like expansion engine."""

from __future__ import annotations

import math
from dataclasses import dataclass

from bbsim.core.config import CosmologyConfig
from bbsim.core.state import UniverseState


@dataclass(frozen=True, slots=True)
class DensityComponents:
    """Density terms used by the simplified Friedmann model.

    Values are normalized prototype density terms, not SI units. `matter` combines
    ordinary baryonic matter and dark matter because both scale as a^-3 in this model.
    """

    radiation: float
    matter: float
    curvature: float
    dark_energy: float

    @property
    def friedmann_term(self) -> float:
        """Return the signed density sum used inside H(a)."""

        return self.radiation + self.matter + self.curvature + self.dark_energy


@dataclass(frozen=True, slots=True)
class DensityFractions:
    """Display-friendly fractional contribution of density components."""

    radiation: float
    matter: float
    curvature: float
    dark_energy: float


@dataclass(frozen=True, slots=True)
class ExpansionSample:
    """One sampled state of the simplified cosmological expansion model."""

    a: float
    h_gyr_inv: float
    era: str
    components: DensityComponents
    fractions: DensityFractions


class ExpansionEngine:
    """Small scalar engine for H(a), density components and history samples."""

    @staticmethod
    def sample(a: float, cosmology: CosmologyConfig) -> ExpansionSample:
        """Create a scalar expansion sample for a scale factor.

        Args:
            a: Positive scale factor.
            cosmology: Cosmological density parameters.

        Returns:
            Expansion sample with H(a), density terms, display fractions and dominant era.

        Raises:
            ValueError: If `a` is not positive.
        """

        components = compute_density_components(a, cosmology)
        h_value = compute_hubble_gyr_inv(a, cosmology)
        return ExpansionSample(
            a=a,
            h_gyr_inv=h_value,
            era=detect_era(a, cosmology),
            components=components,
            fractions=compute_density_fractions(components),
        )

    @staticmethod
    def update_state(
        state: UniverseState,
        cosmology: CosmologyConfig,
        *,
        update_era: bool = True,
        append_history: bool = True,
    ) -> ExpansionSample:
        """Update scalar expansion fields on a mutable universe state.

        Args:
            state: Mutable universe state.
            cosmology: Cosmological density parameters.
            update_era: Whether to replace `state.era` with the dominant density era.
                Early visual stages such as `seed` and `inflation` can keep their own
                semantic era while still recording H(a) and component histories.
            append_history: Whether to append the sample to history arrays used by UI plots.

        Returns:
            The computed expansion sample.
        """

        sample = ExpansionEngine.sample(state.a, cosmology)
        state.h_gyr_inv = sample.h_gyr_inv
        state.rho_r = sample.components.radiation
        state.rho_m = sample.components.matter
        state.rho_k = sample.components.curvature
        state.rho_lambda = sample.components.dark_energy
        state.frac_r = sample.fractions.radiation
        state.frac_m = sample.fractions.matter
        state.frac_k = sample.fractions.curvature
        state.frac_lambda = sample.fractions.dark_energy
        if update_era:
            state.era = sample.era

        if append_history:
            state.a_history.append(state.a)
            state.t_history.append(state.t_gyr)
            state.h_history.append(state.h_gyr_inv)
            state.temperature_history.append(state.temperature_k)
            state.radiation_fraction_history.append(state.frac_r)
            state.matter_fraction_history.append(state.frac_m)
            state.curvature_fraction_history.append(state.frac_k)
            state.dark_energy_fraction_history.append(state.frac_lambda)
            state.ionization_fraction_history.append(state.ionization_fraction)
            state.opacity_history.append(state.opacity)
            state.era_history.append(state.era)

        return sample


def compute_density_components(a: float, cosmology: CosmologyConfig) -> DensityComponents:
    """Compute simplified density terms at scale factor `a`.

    Args:
        a: Positive scale factor.
        cosmology: Cosmological density parameters.

    Returns:
        Density terms for radiation, matter, curvature and dark energy.

    Raises:
        ValueError: If `a` is not positive.
    """

    if a <= 0.0:
        raise ValueError("scale factor must be positive")

    omega_m = cosmology.omega_b + cosmology.omega_dm
    return DensityComponents(
        radiation=cosmology.omega_r / a**4,
        matter=omega_m / a**3,
        curvature=cosmology.omega_k / a**2,
        dark_energy=cosmology.omega_lambda,
    )


def compute_density_fractions(components: DensityComponents) -> DensityFractions:
    """Compute display-friendly component fractions.

    Signed curvature and negative dark-energy toy settings are possible in the prototype.
    For plotting, fractions use absolute contribution magnitudes so the graph remains
    stable and readable even when a component is negative.
    """

    radiation = abs(components.radiation)
    matter = abs(components.matter)
    curvature = abs(components.curvature)
    dark_energy = abs(components.dark_energy)
    total = radiation + matter + curvature + dark_energy
    if total <= 1.0e-300:
        return DensityFractions(0.0, 0.0, 0.0, 0.0)
    return DensityFractions(
        radiation=radiation / total,
        matter=matter / total,
        curvature=curvature / total,
        dark_energy=dark_energy / total,
    )


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

    components = compute_density_components(a, cosmology)
    return cosmology.h0_gyr_inv * math.sqrt(max(components.friedmann_term, 0.0))


def detect_era(a: float, cosmology: CosmologyConfig) -> str:
    """Detect the dominant component era for the simplified model.

    Args:
        a: Positive scale factor.
        cosmology: Cosmological density parameters.

    Returns:
        Era identifier: `radiation`, `matter`, or `dark_energy`.
    """

    components = compute_density_components(a, cosmology)

    if components.radiation > components.matter and components.radiation > components.dark_energy:
        return "radiation"
    if components.matter > components.dark_energy:
        return "matter"
    return "dark_energy"
