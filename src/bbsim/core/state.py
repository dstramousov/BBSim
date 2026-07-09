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
    curvature: float = 0.0
    hydrogen_fraction: float = 0.0
    helium_fraction: float = 0.0
    lithium_trace: float = 0.0
    heavy_elements_fraction: float = 0.0
    a_history: list[float] = field(default_factory=list)
    t_history: list[float] = field(default_factory=list)
    h_history: list[float] = field(default_factory=list)
    temperature_history: list[float] = field(default_factory=list)
    radiation_fraction_history: list[float] = field(default_factory=list)
    matter_fraction_history: list[float] = field(default_factory=list)
    curvature_fraction_history: list[float] = field(default_factory=list)
    dark_energy_fraction_history: list[float] = field(default_factory=list)
    era_history: list[str] = field(default_factory=list)
