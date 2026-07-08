"""Personal primordial seed data."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SeedMetrics:
    """Measured properties of the generated primordial fluctuation field."""

    ripple_contrast: float
    large_scale_power: float
    fine_grain_power: float
    isotropy: float
    void_potential: float
    collapse_risk: float


@dataclass(frozen=True, slots=True)
class PersonalSeed:
    """Immutable identity and metrics for a personal seed.

    The heavy field data is stored in `UniverseFields`; this object stores identity,
    reproducibility metadata, and summary metrics.
    """

    phrase_display: str
    phrase_canonical: str
    root_hash_hex: str
    public_code: str
    metrics: SeedMetrics
