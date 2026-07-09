from __future__ import annotations

from dataclasses import replace

from bbsim.core.config import UniverseConfig
from bbsim.core.context import create_run_context
from bbsim.core.pipeline import create_default_pipeline
from bbsim.numeric.numpy_backend import NumpyBackend


def _advance_to_reionization(config: UniverseConfig):
    context = create_run_context(config=config, backend=NumpyBackend())
    pipeline = create_default_pipeline()
    while pipeline.current_stage is not None and pipeline.current_stage.stage_id != "reionization":
        pipeline.step_to_checkpoint(context)
        pipeline.advance(context)
    assert pipeline.current_stage is not None
    assert pipeline.current_stage.stage_id == "reionization"
    return context, pipeline


def test_reionization_live_growth_expands_bubbles_and_reduces_neutral_gas() -> None:
    config = UniverseConfig.default(player_seed_phrase="Dimas")
    config = replace(
        config,
        time_director=replace(config.time_director, reionization_visual_duration_s=10.0),
    )
    context, pipeline = _advance_to_reionization(config)

    assert context.fields.first_star_density.any()
    assert context.fields.ionized_bubbles.any()
    initial_cold_gas_mean = float(context.fields.cold_gas_density.mean())

    report = pipeline.step_live(context, dt=7.0)

    assert report is None
    assert context.state.current_stage == "reionization"
    assert 0.0 < context.state.stage_progress < 1.0
    assert context.fields.ionization.any()
    assert context.state.ionized_fraction > 0.0
    assert context.state.neutral_fraction < 1.0
    assert context.state.bubble_overlap_fraction >= 0.0
    assert context.state.photoheating_feedback > 0.0
    assert float(context.fields.cold_gas_density.mean()) < initial_cold_gas_mean


def test_reionization_checkpoint_reports_ionization_metrics() -> None:
    config = UniverseConfig.default(player_seed_phrase="Dimas")
    config = replace(
        config,
        time_director=replace(config.time_director, reionization_visual_duration_s=3.0),
    )
    context, pipeline = _advance_to_reionization(config)

    report = pipeline.step_live(context, dt=3.5)

    assert report is not None
    assert report.stage_id == "reionization"
    assert report.metrics["ionized_fraction"] > 0.0
    assert report.metrics["neutral_fraction"] < 1.0
    assert report.metrics["bubble_overlap_fraction"] > 0.0
    assert report.metrics["photoheating_feedback"] > 0.0
    assert "reionization_star_correlation" in report.metrics
