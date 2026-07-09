"""Configuration objects for a single universe run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CosmologyConfig:
    """Cosmological parameters used by the expansion model.

    Values are normalized for the prototype. `h0_gyr_inv` is the Hubble parameter in
    inverse gigayears. Omega values are density fractions relative to today's critical
    density in the simplified model.
    """

    h0_gyr_inv: float = 0.069
    omega_b: float = 0.049
    omega_dm: float = 0.265
    omega_lambda: float = 0.686
    omega_r: float = 0.0001
    omega_k: float = 0.0


@dataclass(frozen=True, slots=True)
class InflationConfig:
    """Parameters controlling the prototype inflation stage."""

    strength: float = 0.72
    duration: float = 0.68
    smoothing: float = 0.85
    visual_duration_s: float = 8.0


@dataclass(frozen=True, slots=True)
class SeedConfig:
    """Parameters controlling the personal primordial fluctuation field."""

    phrase: str = "Dimas"
    grid_size: int = 192
    fluctuation_amplitude: float = 0.35
    fluctuation_scale: float = 0.50
    spectral_tilt: float = 0.965


@dataclass(frozen=True, slots=True)
class EarlyUniverseConfig:
    """Simple outputs for early live universe stages."""

    reheating_visual_duration_s: float = 6.0
    nucleosynthesis_visual_duration_s: float = 6.0
    recombination_visual_duration_s: float = 7.0


@dataclass(frozen=True, slots=True)
class TimeDirectorConfig:
    """Cinematic screen-time direction for epoch playback."""

    mode: str = "cinematic"
    duration_scale: float = 1.0
    personal_seed_visual_duration_s: float = 12.0
    inflation_visual_duration_s: float = 38.0
    reheating_visual_duration_s: float = 28.0
    nucleosynthesis_visual_duration_s: float = 28.0
    recombination_visual_duration_s: float = 38.0
    dark_ages_visual_duration_s: float = 48.0


@dataclass(frozen=True, slots=True)
class ScaleConfig:
    """Displayed comoving box scale used for human-readable distance overlays."""

    box_size_today_mpc: float = 1000.0
    show_scale_overlay: bool = True


@dataclass(frozen=True, slots=True)
class VisualDirectorConfig:
    """Tunable visual direction parameters for live epoch rendering."""

    epoch_transition_fraction: float = 0.18
    inflation_zoom_strength: float = 1.55
    reheating_pulse_strength: float = 0.14
    recombination_clearing_strength: float = 0.36


@dataclass(frozen=True, slots=True)
class StructureConfig:
    """Prototype parameters for later structure growth stages."""

    gravity_strength: float = 1.0
    baryon_infall: float = 0.8
    gas_pressure: float = 0.3
    cooling_efficiency: float = 0.55
    star_formation_efficiency: float = 0.45
    feedback_strength: float = 0.40
    metal_yield: float = 0.25
    black_hole_efficiency: float = 0.15


@dataclass(frozen=True, slots=True)
class UniverseConfig:
    """Immutable configuration for one deterministic universe run."""

    seed: SeedConfig
    cosmology: CosmologyConfig
    inflation: InflationConfig
    early_universe: EarlyUniverseConfig
    time_director: TimeDirectorConfig
    scale: ScaleConfig
    visual_director: VisualDirectorConfig
    structure: StructureConfig

    @staticmethod
    def default(player_seed_phrase: str = "Dimas") -> "UniverseConfig":
        """Create the default Lambda-CDM-like prototype configuration.

        Args:
            player_seed_phrase: Phrase used to derive the personal primordial seed.

        Returns:
            Immutable universe configuration.
        """

        return UniverseConfig(
            seed=SeedConfig(phrase=player_seed_phrase),
            cosmology=CosmologyConfig(),
            inflation=InflationConfig(),
            early_universe=EarlyUniverseConfig(),
            time_director=TimeDirectorConfig(),
            scale=ScaleConfig(),
            visual_director=VisualDirectorConfig(),
            structure=StructureConfig(),
        )
