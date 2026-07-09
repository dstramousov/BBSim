from __future__ import annotations

from dataclasses import replace

from bbsim.core.config import UniverseConfig
from bbsim.core.context import create_run_context
from bbsim.core.pipeline import create_default_pipeline
from bbsim.numeric.numpy_backend import NumpyBackend


def _advance_to_first_stars(config: UniverseConfig):
    context = create_run_context(config=config, backend=NumpyBackend())
    pipeline = create_default_pipeline()
    while pipeline.current_stage is not None and pipeline.current_stage.stage_id != "first_stars":
        pipeline.step_to_checkpoint(context)
        pipeline.advance(context)
    assert pipeline.current_stage is not None
    assert pipeline.current_stage.stage_id == "first_stars"
    return context, pipeline


def test_first_stars_live_growth_ignites_only_after_collapse_sites() -> None:
    config = UniverseConfig.default(player_seed_phrase="Dimas")
    config = replace(
        config,
        time_director=replace(config.time_director, first_stars_visual_duration_s=10.0),
    )
    context, pipeline = _advance_to_first_stars(config)

    assert context.fields.collapse_sites.any()
    assert not context.fields.stars.any()

    report = pipeline.step_live(context, dt=6.8)

    assert report is None
    assert context.state.current_stage == "first_stars"
    assert 0.0 < context.state.stage_progress < 1.0
    assert context.fields.stellar_ignition.any()
    assert context.fields.first_star_density.any()
    assert context.fields.stars.any()
    assert context.fields.stellar_radiation.any()
    assert context.fields.ionized_bubbles.any()
    assert context.state.first_star_count > 0
    assert context.state.star_formation_fraction > 0.0
    assert context.state.stellar_radiation_intensity > 0.0
    assert context.state.ionized_bubble_fraction > 0.0


def test_first_stars_checkpoint_reports_first_light_metrics() -> None:
    config = UniverseConfig.default(player_seed_phrase="Dimas")
    config = replace(
        config,
        time_director=replace(config.time_director, first_stars_visual_duration_s=3.0),
    )
    context, pipeline = _advance_to_first_stars(config)

    report = pipeline.step_live(context, dt=3.5)

    assert report is not None
    assert report.stage_id == "first_stars"
    assert report.metrics["first_star_count"] > 0.0
    assert report.metrics["star_formation_fraction"] > 0.0
    assert report.metrics["stellar_radiation_intensity"] > 0.0
    assert report.metrics["ionized_bubble_fraction"] > 0.0
    assert "first_star_collapse_correlation" in report.metrics
