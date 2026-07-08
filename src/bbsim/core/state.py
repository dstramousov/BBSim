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
    current_stage: str = "not_started"
    stage_progress: float = 0.0
    curvature: float = 0.0
    a_history: list[float] = field(default_factory=list)
    t_history: list[float] = field(default_factory=list)
