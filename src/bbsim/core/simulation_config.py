"""Simulation-level configuration loaded from TOML."""

from __future__ import annotations

import os
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

from bbsim.core.config import (
    CosmologyConfig,
    EarlyUniverseConfig,
    InflationConfig,
    SeedConfig,
    StructureConfig,
    UniverseConfig,
)


def load_simulation_config(
    path: Path | str | None = None,
    *,
    phrase_override: str | None = None,
) -> UniverseConfig:
    """Load deterministic universe configuration from TOML.

    The default lookup order is:
    1. `BBSIM_SIMULATION_CONFIG` environment variable, if set;
    2. `config/simulation.toml` relative to the current working directory;
    3. built-in defaults.

    Args:
        path: Optional explicit path to a TOML config file.
        phrase_override: Optional phrase value that overrides `[seed].phrase`.

    Returns:
        Immutable universe configuration with validated default fallbacks.
    """

    defaults = UniverseConfig.default()
    config_path = _resolve_config_path(path)
    if config_path is None or not config_path.exists():
        return _with_phrase_override(defaults, phrase_override)

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    config = _parse_simulation_config(raw, defaults)
    return _with_phrase_override(config, phrase_override)


def _resolve_config_path(path: Path | str | None) -> Path | None:
    if path is not None:
        return Path(path)

    env_path = os.environ.get("BBSIM_SIMULATION_CONFIG")
    if env_path:
        return Path(env_path)

    cwd_config = Path.cwd() / "config" / "simulation.toml"
    if cwd_config.exists():
        return cwd_config

    return None


def _parse_simulation_config(raw: dict[str, Any], defaults: UniverseConfig) -> UniverseConfig:
    return UniverseConfig(
        seed=_parse_seed_config(raw.get("seed", {}), defaults.seed),
        cosmology=_parse_cosmology_config(raw.get("cosmology", {}), defaults.cosmology),
        inflation=_parse_inflation_config(raw.get("inflation", {}), defaults.inflation),
        early_universe=_parse_early_universe_config(
            raw.get("early_universe", {}), defaults.early_universe
        ),
        structure=_parse_structure_config(raw.get("structure", {}), defaults.structure),
    )


def _parse_seed_config(raw: Any, defaults: SeedConfig) -> SeedConfig:
    if not isinstance(raw, dict):
        return defaults

    phrase = _string(raw.get("phrase"), defaults.phrase)
    grid_size = _bounded_int(raw.get("grid_size"), defaults.grid_size, minimum=16, maximum=1024)
    fluctuation_amplitude = _bounded_float(
        raw.get("fluctuation_amplitude"),
        defaults.fluctuation_amplitude,
        minimum=0.0,
        maximum=2.0,
    )
    fluctuation_scale = _bounded_float(
        raw.get("fluctuation_scale"),
        defaults.fluctuation_scale,
        minimum=0.05,
        maximum=1.0,
    )
    spectral_tilt = _bounded_float(
        raw.get("spectral_tilt"),
        defaults.spectral_tilt,
        minimum=0.1,
        maximum=2.0,
    )
    return replace(
        defaults,
        phrase=phrase,
        grid_size=grid_size,
        fluctuation_amplitude=fluctuation_amplitude,
        fluctuation_scale=fluctuation_scale,
        spectral_tilt=spectral_tilt,
    )


def _parse_cosmology_config(raw: Any, defaults: CosmologyConfig) -> CosmologyConfig:
    if not isinstance(raw, dict):
        return defaults

    h0_value = raw.get("h0_gyr_inv", raw.get("H0", raw.get("h0", defaults.h0_gyr_inv)))
    return replace(
        defaults,
        h0_gyr_inv=_bounded_float(h0_value, defaults.h0_gyr_inv, minimum=0.001, maximum=1.0),
        omega_b=_bounded_float(raw.get("omega_b"), defaults.omega_b, minimum=0.0, maximum=2.0),
        omega_dm=_bounded_float(raw.get("omega_dm"), defaults.omega_dm, minimum=0.0, maximum=2.0),
        omega_lambda=_bounded_float(
            raw.get("omega_lambda"), defaults.omega_lambda, minimum=-2.0, maximum=3.0
        ),
        omega_r=_bounded_float(raw.get("omega_r"), defaults.omega_r, minimum=0.0, maximum=1.0),
        omega_k=_bounded_float(raw.get("omega_k"), defaults.omega_k, minimum=-3.0, maximum=3.0),
    )


def _parse_inflation_config(raw: Any, defaults: InflationConfig) -> InflationConfig:
    if not isinstance(raw, dict):
        return defaults

    return replace(
        defaults,
        strength=_bounded_float(raw.get("strength"), defaults.strength, minimum=0.0, maximum=3.0),
        duration=_bounded_float(raw.get("duration"), defaults.duration, minimum=0.0, maximum=3.0),
        smoothing=_bounded_float(raw.get("smoothing"), defaults.smoothing, minimum=0.0, maximum=1.0),
        visual_duration_s=_bounded_float(
            raw.get("visual_duration_s"),
            defaults.visual_duration_s,
            minimum=1.0,
            maximum=600.0,
        ),
    )


def _parse_early_universe_config(raw: Any, defaults: EarlyUniverseConfig) -> EarlyUniverseConfig:
    if not isinstance(raw, dict):
        return defaults

    return replace(
        defaults,
        reheating_visual_duration_s=_bounded_float(
            raw.get("reheating_visual_duration_s"),
            defaults.reheating_visual_duration_s,
            minimum=1.0,
            maximum=600.0,
        ),
        nucleosynthesis_visual_duration_s=_bounded_float(
            raw.get("nucleosynthesis_visual_duration_s"),
            defaults.nucleosynthesis_visual_duration_s,
            minimum=1.0,
            maximum=600.0,
        ),
        recombination_visual_duration_s=_bounded_float(
            raw.get("recombination_visual_duration_s"),
            defaults.recombination_visual_duration_s,
            minimum=1.0,
            maximum=600.0,
        ),
    )


def _parse_structure_config(raw: Any, defaults: StructureConfig) -> StructureConfig:
    if not isinstance(raw, dict):
        return defaults

    return replace(
        defaults,
        gravity_strength=_bounded_float(
            raw.get("gravity_strength"), defaults.gravity_strength, minimum=0.0, maximum=5.0
        ),
        baryon_infall=_bounded_float(raw.get("baryon_infall"), defaults.baryon_infall, 0.0, 5.0),
        gas_pressure=_bounded_float(raw.get("gas_pressure"), defaults.gas_pressure, 0.0, 5.0),
        cooling_efficiency=_bounded_float(
            raw.get("cooling_efficiency"), defaults.cooling_efficiency, 0.0, 5.0
        ),
        star_formation_efficiency=_bounded_float(
            raw.get("star_formation_efficiency"), defaults.star_formation_efficiency, 0.0, 5.0
        ),
        feedback_strength=_bounded_float(
            raw.get("feedback_strength"), defaults.feedback_strength, 0.0, 5.0
        ),
        metal_yield=_bounded_float(raw.get("metal_yield"), defaults.metal_yield, 0.0, 5.0),
        black_hole_efficiency=_bounded_float(
            raw.get("black_hole_efficiency"), defaults.black_hole_efficiency, 0.0, 5.0
        ),
    )


def _with_phrase_override(config: UniverseConfig, phrase_override: str | None) -> UniverseConfig:
    if phrase_override is None:
        return config
    return replace(config, seed=replace(config.seed, phrase=phrase_override))


def _string(value: Any, default: str) -> str:
    if not isinstance(value, str):
        return default
    cleaned = value.strip()
    return cleaned if cleaned else default


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        return default
    if not isinstance(value, int):
        return default
    return int(min(max(value, minimum), maximum))


def _bounded_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        return default
    if not isinstance(value, (int, float)):
        return default
    numeric = float(value)
    return float(min(max(numeric, minimum), maximum))
