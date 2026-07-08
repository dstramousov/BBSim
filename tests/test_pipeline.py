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
        "recombination_preview",
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
