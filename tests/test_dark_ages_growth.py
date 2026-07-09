from __future__ import annotations

from dataclasses import replace

from bbsim.core.config import UniverseConfig
from bbsim.core.context import create_run_context
from bbsim.core.pipeline import create_default_pipeline
from bbsim.numeric.numpy_backend import NumpyBackend


def _advance_to_dark_ages(config: UniverseConfig):
    context = create_run_context(config=config, backend=NumpyBackend())
    pipeline = create_default_pipeline()
    while pipeline.current_stage is not None and pipeline.current_stage.stage_id != "dark_ages":
        pipeline.step_to_checkpoint(context)
        pipeline.advance(context)
    assert pipeline.current_stage is not None
    assert pipeline.current_stage.stage_id == "dark_ages"
    return context, pipeline


def test_dark_ages_live_growth_creates_halo_and_future_site_layers() -> None:
    config = UniverseConfig.default(player_seed_phrase="Dimas")
    config = replace(
        config,
        time_director=replace(config.time_director, dark_ages_visual_duration_s=10.0),
    )
    context, pipeline = _advance_to_dark_ages(config)

    report = pipeline.step_live(context, dt=6.5)

    assert report is None
    assert context.state.current_stage == "dark_ages"
    assert 0.0 < context.state.stage_progress < 1.0
    assert context.fields.gravitational_potential.any()
    assert context.fields.halo_density.any()
    assert context.fields.future_star_sites.any()
    assert context.state.halo_count > 0
    assert context.state.future_star_site_count > 0
    assert context.state.gas_lag > 0.0
    assert context.state.dark_matter_contrast > context.state.baryon_contrast
    assert context.state.dark_matter_contrast_history
    assert context.state.halo_count_history


def test_dark_ages_checkpoint_reports_structure_metrics() -> None:
    config = UniverseConfig.default(player_seed_phrase="Dimas")
    config = replace(
        config,
        time_director=replace(config.time_director, dark_ages_visual_duration_s=3.0),
    )
    context, pipeline = _advance_to_dark_ages(config)

    report = pipeline.step_live(context, dt=3.5)

    assert report is not None
    assert report.stage_id == "dark_ages"
    assert report.metrics["halo_count"] > 0
    assert report.metrics["halo_mass_fraction"] > 0.0
    assert "future_star_site_count" in report.metrics
