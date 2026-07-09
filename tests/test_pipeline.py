from __future__ import annotations

from bbsim.core.config import UniverseConfig
from bbsim.core.context import create_run_context
from bbsim.core.pipeline import create_default_pipeline
from bbsim.numeric.numpy_backend import NumpyBackend


def test_default_pipeline_reaches_all_initial_checkpoints() -> None:
    context = create_run_context(
        config=UniverseConfig.default(player_seed_phrase="Dimas"),
        backend=NumpyBackend(),
    )
    pipeline = create_default_pipeline()

    while not pipeline.is_finished:
        pipeline.step_to_checkpoint(context)
        pipeline.advance(context)

    assert [report.stage_id for report in context.history.reports] == [
        "personal_seed",
        "inflation",
        "reheating",
        "nucleosynthesis",
        "recombination",
    ]


def test_live_step_advances_inflation_without_fast_forwarding() -> None:
    config = UniverseConfig.default(player_seed_phrase="Dimas")
    context = create_run_context(config=config, backend=NumpyBackend())
    pipeline = create_default_pipeline()

    # Complete the short seed reveal using the live API.
    report = None
    while report is None:
        report = pipeline.step_live(context, dt=0.5)
    assert report.stage_id == "personal_seed"
    pipeline.advance(context)

    # One small live tick enters inflation and moves it partially, not to checkpoint.
    report = pipeline.step_live(context, dt=0.5)

    assert report is None
    assert context.state.current_stage == "inflation"
    assert 0.0 < context.state.stage_progress < 1.0
    assert context.fields.inflation_delta.any()


def test_reheating_and_nucleosynthesis_update_early_fields() -> None:
    config = UniverseConfig.default(player_seed_phrase="Dimas")
    context = create_run_context(config=config, backend=NumpyBackend())
    pipeline = create_default_pipeline()

    while not pipeline.is_finished:
        pipeline.step_to_checkpoint(context)
        pipeline.advance(context)

    reports = {report.stage_id: report for report in context.history.reports}
    assert "reheating" in reports
    assert "nucleosynthesis" in reports
    assert context.fields.radiation.any()
    assert context.state.hydrogen_fraction > 0.7
    assert context.state.helium_fraction > 0.2
    assert context.state.h_history
    assert context.state.radiation_fraction_history
    assert context.state.matter_fraction_history


def test_recombination_releases_transparent_cmb() -> None:
    config = UniverseConfig.default(player_seed_phrase="Dimas")
    context = create_run_context(config=config, backend=NumpyBackend())
    pipeline = create_default_pipeline()

    while not pipeline.is_finished:
        pipeline.step_to_checkpoint(context)
        report = context.history.reports[-1]
        if report.stage_id == "recombination":
            break
        pipeline.advance(context)

    assert context.state.current_stage == "recombination"
    assert context.state.temperature_k <= 3500.0
    assert context.state.ionization_fraction < 0.01
    assert context.state.opacity < 0.01
    assert context.state.cmb_released is True
    assert context.fields.cmb.any()
    assert context.state.ionization_fraction_history
    assert context.state.opacity_history


def test_default_inflation_visual_duration_is_observable_not_stuck() -> None:
    config = UniverseConfig.default(player_seed_phrase="Dimas")

    assert 1.0 <= config.inflation.visual_duration_s <= 12.0


def test_live_pipeline_leaves_seed_stage_after_short_playback() -> None:
    config = UniverseConfig.default(player_seed_phrase="Dimas")
    context = create_run_context(config=config, backend=NumpyBackend())
    pipeline = create_default_pipeline()

    report = None
    for _ in range(80):
        report = pipeline.step_live(context, dt=0.033)
        if report is not None:
            break

    assert report is not None
    assert report.stage_id == "personal_seed"
    pipeline.advance(context)

    pipeline.step_live(context, dt=0.033)
    assert context.state.current_stage == "inflation"
