from __future__ import annotations

from dataclasses import replace

from bbsim.core.config import UniverseConfig
from bbsim.core.context import create_run_context
from bbsim.core.pipeline import create_default_pipeline
from bbsim.numeric.numpy_backend import NumpyBackend


def _advance_to_gas_collapse(config: UniverseConfig):
    context = create_run_context(config=config, backend=NumpyBackend())
    pipeline = create_default_pipeline()
    while pipeline.current_stage is not None and pipeline.current_stage.stage_id != "gas_collapse":
        pipeline.step_to_checkpoint(context)
        pipeline.advance(context)
    assert pipeline.current_stage is not None
    assert pipeline.current_stage.stage_id == "gas_collapse"
    return context, pipeline


def test_gas_collapse_live_growth_creates_cooling_and_collapse_layers() -> None:
    config = UniverseConfig.default(player_seed_phrase="Dimas")
    config = replace(
        config,
        time_director=replace(
            config.time_director,
            dark_ages_visual_duration_s=2.0,
            gas_collapse_visual_duration_s=10.0,
        ),
    )
    context, pipeline = _advance_to_gas_collapse(config)

    report = pipeline.step_live(context, dt=6.8)

    assert report is None
    assert context.state.current_stage == "gas_collapse"
    assert 0.0 < context.state.stage_progress < 1.0
    assert context.fields.cold_gas_density.any()
    assert context.fields.molecular_cooling.any()
    assert context.fields.collapse_sites.any()
    assert not context.fields.stars.any()
    assert context.state.gas_cooling_fraction > 0.0
    assert context.state.collapse_site_count > 0
    assert context.state.star_formation_readiness > 0.0


def test_gas_collapse_checkpoint_reports_pre_stellar_metrics() -> None:
    config = UniverseConfig.default(player_seed_phrase="Dimas")
    config = replace(
        config,
        time_director=replace(config.time_director, gas_collapse_visual_duration_s=3.0),
    )
    context, pipeline = _advance_to_gas_collapse(config)

    report = pipeline.step_live(context, dt=3.5)

    assert report is not None
    assert report.stage_id == "gas_collapse"
    assert report.metrics["gas_cooling_fraction"] > 0.0
    assert report.metrics["collapse_site_count"] > 0.0
    assert report.metrics["star_formation_readiness"] > 0.0
    assert "gas_temperature_k" in report.metrics
