from __future__ import annotations

from dataclasses import replace

from bbsim.core.config import UniverseConfig
from bbsim.core.time_director import sample_time_scale, stage_screen_duration_s


def test_stage_screen_duration_uses_time_director_config() -> None:
    config = UniverseConfig.default()
    config = replace(
        config,
        time_director=replace(
            config.time_director,
            duration_scale=2.0,
            inflation_visual_duration_s=10.0,
        ),
    )

    assert stage_screen_duration_s(config, "inflation") == 20.0


def test_time_scale_sample_explains_physical_time_mapping() -> None:
    config = UniverseConfig.default()

    sample = sample_time_scale(config, "recombination", 0.5)

    assert sample is not None
    assert sample.stage_id == "recombination"
    assert sample.physical_time_s is not None
    assert "сек экрана" in sample.time_scale_text
    assert "экранная длительность" in sample.screen_duration_text


def test_dark_ages_has_own_screen_duration_and_time_scale() -> None:
    config = UniverseConfig.default()

    assert stage_screen_duration_s(config, "dark_ages") == config.time_director.dark_ages_visual_duration_s
    sample = sample_time_scale(config, "dark_ages", 0.5)

    assert sample is not None
    assert sample.physical_time_s is not None
    assert "лет" in sample.physical_time_text


def test_gas_collapse_has_own_screen_duration_and_time_scale() -> None:
    config = UniverseConfig.default()

    assert (
        stage_screen_duration_s(config, "gas_collapse")
        == config.time_director.gas_collapse_visual_duration_s
    )
    sample = sample_time_scale(config, "gas_collapse", 0.5)

    assert sample is not None
    assert sample.physical_time_s is not None
    assert "млн лет" in sample.physical_time_text
