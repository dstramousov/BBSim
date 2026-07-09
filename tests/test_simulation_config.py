from __future__ import annotations

from pathlib import Path

from bbsim.core.config import UniverseConfig
from bbsim.core.simulation_config import load_simulation_config


def test_missing_simulation_config_uses_defaults(tmp_path: Path) -> None:
    config = load_simulation_config(tmp_path / "missing.toml")

    assert config == UniverseConfig.default()
    assert config.seed.phrase == "Dimas"
    assert config.seed.grid_size == 192


def test_simulation_config_loads_core_parameters(tmp_path: Path) -> None:
    config_path = tmp_path / "simulation.toml"
    config_path.write_text(
        """
[seed]
phrase = "Andromeda"
grid_size = 128
fluctuation_amplitude = 0.42
fluctuation_scale = 0.33
spectral_tilt = 1.2

[inflation]
strength = 0.9
duration = 0.7
smoothing = 0.6
visual_duration_s = 25.0

[cosmology]
h0_gyr_inv = 0.08
omega_b = 0.06
omega_dm = 0.31
omega_lambda = 0.62
omega_r = 0.0002
omega_k = 0.01

[early_universe]
reheating_visual_duration_s = 4.0
nucleosynthesis_visual_duration_s = 5.0
recombination_visual_duration_s = 6.0

[time_director]
mode = "deep"
duration_scale = 1.5
personal_seed_visual_duration_s = 11.0
inflation_visual_duration_s = 22.0
reheating_visual_duration_s = 33.0
nucleosynthesis_visual_duration_s = 44.0
recombination_visual_duration_s = 55.0
dark_ages_visual_duration_s = 66.0

[scale]
box_size_today_mpc = 500.0
show_scale_overlay = false
""".strip(),
        encoding="utf-8",
    )

    config = load_simulation_config(config_path)

    assert config.seed.phrase == "Andromeda"
    assert config.seed.grid_size == 128
    assert config.seed.fluctuation_amplitude == 0.42
    assert config.seed.fluctuation_scale == 0.33
    assert config.seed.spectral_tilt == 1.2
    assert config.inflation.strength == 0.9
    assert config.inflation.duration == 0.7
    assert config.inflation.smoothing == 0.6
    assert config.inflation.visual_duration_s == 25.0
    assert config.cosmology.h0_gyr_inv == 0.08
    assert config.cosmology.omega_b == 0.06
    assert config.cosmology.omega_dm == 0.31
    assert config.cosmology.omega_lambda == 0.62
    assert config.cosmology.omega_r == 0.0002
    assert config.cosmology.omega_k == 0.01
    assert config.early_universe.reheating_visual_duration_s == 4.0
    assert config.early_universe.nucleosynthesis_visual_duration_s == 5.0
    assert config.early_universe.recombination_visual_duration_s == 6.0
    assert config.time_director.mode == "deep"
    assert config.time_director.duration_scale == 1.5
    assert config.time_director.personal_seed_visual_duration_s == 11.0
    assert config.time_director.inflation_visual_duration_s == 22.0
    assert config.time_director.reheating_visual_duration_s == 33.0
    assert config.time_director.nucleosynthesis_visual_duration_s == 44.0
    assert config.time_director.recombination_visual_duration_s == 55.0
    assert config.time_director.dark_ages_visual_duration_s == 66.0
    assert config.scale.box_size_today_mpc == 500.0
    assert config.scale.show_scale_overlay is False


def test_simulation_config_phrase_override_wins(tmp_path: Path) -> None:
    config_path = tmp_path / "simulation.toml"
    config_path.write_text('[seed]\nphrase = "From File"\n', encoding="utf-8")

    config = load_simulation_config(config_path, phrase_override="From CLI")

    assert config.seed.phrase == "From CLI"


def test_invalid_simulation_config_values_fall_back_or_clamp(tmp_path: Path) -> None:
    config_path = tmp_path / "simulation.toml"
    config_path.write_text(
        """
[seed]
phrase = ""
grid_size = 4
fluctuation_amplitude = 99
fluctuation_scale = -1
spectral_tilt = "blue"

[inflation]
strength = -10
duration = 99
smoothing = 9
visual_duration_s = 0

[cosmology]
h0_gyr_inv = "fast"
omega_b = -1
omega_dm = 10
omega_lambda = 10
omega_r = -3
omega_k = 10

[early_universe]
reheating_visual_duration_s = 0
nucleosynthesis_visual_duration_s = 999
recombination_visual_duration_s = "slow"

[time_director]
mode = "turbo"
duration_scale = 0
personal_seed_visual_duration_s = 0
inflation_visual_duration_s = 999
reheating_visual_duration_s = "fast"
nucleosynthesis_visual_duration_s = -1
recombination_visual_duration_s = 999
dark_ages_visual_duration_s = 0

[scale]
box_size_today_mpc = -4
show_scale_overlay = "yes"
""".strip(),
        encoding="utf-8",
    )

    config = load_simulation_config(config_path)

    assert config.seed.phrase == "Dimas"
    assert config.seed.grid_size == 16
    assert config.seed.fluctuation_amplitude == 2.0
    assert config.seed.fluctuation_scale == 0.05
    assert config.seed.spectral_tilt == UniverseConfig.default().seed.spectral_tilt
    assert config.inflation.strength == 0.0
    assert config.inflation.duration == 3.0
    assert config.inflation.smoothing == 1.0
    assert config.inflation.visual_duration_s == 1.0
    assert config.cosmology.h0_gyr_inv == UniverseConfig.default().cosmology.h0_gyr_inv
    assert config.cosmology.omega_b == 0.0
    assert config.cosmology.omega_dm == 2.0
    assert config.cosmology.omega_lambda == 3.0
    assert config.cosmology.omega_r == 0.0
    assert config.cosmology.omega_k == 3.0
    assert config.early_universe.reheating_visual_duration_s == 1.0
    assert config.early_universe.nucleosynthesis_visual_duration_s == 600.0
    assert (
        config.early_universe.recombination_visual_duration_s
        == UniverseConfig.default().early_universe.recombination_visual_duration_s
    )
    assert config.time_director.mode == UniverseConfig.default().time_director.mode
    assert config.time_director.duration_scale == 0.1
    assert config.time_director.personal_seed_visual_duration_s == 1.0
    assert config.time_director.inflation_visual_duration_s == 600.0
    assert (
        config.time_director.reheating_visual_duration_s
        == UniverseConfig.default().time_director.reheating_visual_duration_s
    )
    assert config.time_director.nucleosynthesis_visual_duration_s == 1.0
    assert config.time_director.recombination_visual_duration_s == 600.0
    assert config.time_director.dark_ages_visual_duration_s == 1.0
    assert config.scale.box_size_today_mpc == 0.001
    assert config.scale.show_scale_overlay == UniverseConfig.default().scale.show_scale_overlay
