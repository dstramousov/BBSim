from __future__ import annotations

import pytest

from bbsim.core.config import CosmologyConfig
from bbsim.core.expansion import (
    ExpansionEngine,
    compute_density_components,
    compute_density_fractions,
    compute_hubble_gyr_inv,
    detect_era,
)
from bbsim.core.state import UniverseState


def test_hubble_requires_positive_scale_factor() -> None:
    with pytest.raises(ValueError):
        compute_hubble_gyr_inv(0.0, CosmologyConfig())


def test_hubble_is_positive_for_default_config() -> None:
    assert compute_hubble_gyr_inv(1.0, CosmologyConfig()) > 0.0


def test_detect_era_changes_with_scale_factor() -> None:
    config = CosmologyConfig()
    assert detect_era(1.0e-6, config) == "radiation"
    assert detect_era(1.0, config) == "dark_energy"


def test_density_components_follow_expected_scaling() -> None:
    config = CosmologyConfig()
    early = compute_density_components(1.0e-4, config)
    today = compute_density_components(1.0, config)

    assert early.radiation > today.radiation
    assert early.matter > today.matter
    assert today.dark_energy == config.omega_lambda


def test_density_fractions_are_display_normalized() -> None:
    fractions = compute_density_fractions(compute_density_components(1.0, CosmologyConfig()))

    total = fractions.radiation + fractions.matter + fractions.curvature + fractions.dark_energy
    assert total == pytest.approx(1.0)
    assert fractions.dark_energy > fractions.radiation


def test_expansion_engine_updates_state_and_history() -> None:
    state = UniverseState(a=1.0e-6, temperature_k=1000.0)

    sample = ExpansionEngine.update_state(state, CosmologyConfig())

    assert sample.era == "radiation"
    assert state.h_gyr_inv > 0.0
    assert state.rho_r > state.rho_m
    assert state.frac_r > state.frac_m
    assert state.a_history == [state.a]
    assert state.h_history == [state.h_gyr_inv]
    assert state.radiation_fraction_history == [state.frac_r]
    assert state.temperature_history == [state.temperature_k]
