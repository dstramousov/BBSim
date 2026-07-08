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
