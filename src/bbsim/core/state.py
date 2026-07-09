"""Mutable global state of a universe run."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class UniverseState:
    """Global scalar state shared by pipeline stages."""

    t_gyr: float = 0.0
    a: float = 1.0e-32
    h_gyr_inv: float = 0.0
    temperature_k: float = 1.0e9
    era: str = "initial"
    rho_r: float = 0.0
    rho_m: float = 0.0
    rho_k: float = 0.0
    rho_lambda: float = 0.0
    frac_r: float = 0.0
    frac_m: float = 0.0
    frac_k: float = 0.0
    frac_lambda: float = 0.0
    current_stage: str = "not_started"
    stage_progress: float = 0.0
    ionization_fraction: float = 1.0
    opacity: float = 1.0
    cmb_released: bool = False
    curvature: float = 0.0
    hydrogen_fraction: float = 0.0
    helium_fraction: float = 0.0
    lithium_trace: float = 0.0
    heavy_elements_fraction: float = 0.0
    dark_matter_contrast: float = 0.0
    baryon_contrast: float = 0.0
    gas_lag: float = 0.0
    halo_count: int = 0
    halo_mass_fraction: float = 0.0
    future_star_site_count: int = 0
    cold_gas_fraction: float = 0.0
    gas_cooling_fraction: float = 0.0
    collapse_site_count: int = 0
    collapsed_gas_fraction: float = 0.0
    star_formation_readiness: float = 0.0
    gas_temperature_k: float = 0.0
    first_star_count: int = 0
    first_star_mass_fraction: float = 0.0
    star_formation_fraction: float = 0.0
    stellar_radiation_intensity: float = 0.0
    ionized_bubble_fraction: float = 0.0
    ionized_fraction: float = 0.0
    neutral_fraction: float = 1.0
    bubble_overlap_fraction: float = 0.0
    photoheating_feedback: float = 0.0
    reionization_progress: float = 0.0
    a_history: list[float] = field(default_factory=list)
    t_history: list[float] = field(default_factory=list)
    h_history: list[float] = field(default_factory=list)
    temperature_history: list[float] = field(default_factory=list)
    radiation_fraction_history: list[float] = field(default_factory=list)
    matter_fraction_history: list[float] = field(default_factory=list)
    curvature_fraction_history: list[float] = field(default_factory=list)
    dark_energy_fraction_history: list[float] = field(default_factory=list)
    ionization_fraction_history: list[float] = field(default_factory=list)
    opacity_history: list[float] = field(default_factory=list)
    dark_matter_contrast_history: list[float] = field(default_factory=list)
    baryon_contrast_history: list[float] = field(default_factory=list)
    halo_count_history: list[int] = field(default_factory=list)
    gas_lag_history: list[float] = field(default_factory=list)
    gas_cooling_fraction_history: list[float] = field(default_factory=list)
    collapse_site_count_history: list[int] = field(default_factory=list)
    star_formation_readiness_history: list[float] = field(default_factory=list)
    first_star_count_history: list[int] = field(default_factory=list)
    stellar_radiation_history: list[float] = field(default_factory=list)
    ionized_bubble_fraction_history: list[float] = field(default_factory=list)
    ionized_fraction_history: list[float] = field(default_factory=list)
    neutral_fraction_history: list[float] = field(default_factory=list)
    bubble_overlap_fraction_history: list[float] = field(default_factory=list)
    photoheating_feedback_history: list[float] = field(default_factory=list)
    era_history: list[str] = field(default_factory=list)
