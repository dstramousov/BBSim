from __future__ import annotations

import numpy as np

from bbsim.core.config import UniverseConfig
from bbsim.core.context import create_run_context
from bbsim.core.pipeline import create_default_pipeline
from bbsim.numeric.numpy_backend import NumpyBackend


def _run_until_stage(stage_id: str):
    context = create_run_context(
        config=UniverseConfig.default(player_seed_phrase="Dimas"),
        backend=NumpyBackend(),
    )
    pipeline = create_default_pipeline()

    while not pipeline.is_finished:
        pipeline.step_to_checkpoint(context)
        report = context.history.reports[-1]
        if report.stage_id == stage_id:
            return context, report
        pipeline.advance(context)

    raise AssertionError(f"stage not reached: {stage_id}")


def test_inflation_creates_distinct_smoothed_field() -> None:
    context, report = _run_until_stage("inflation")

    assert report.metrics["n_e_folds"] > 0.0
    assert report.metrics["expansion_factor"] > 1.0
    assert np.any(context.fields.inflation_delta)
    assert not np.array_equal(context.fields.seed_delta, context.fields.inflation_delta)
    assert np.std(context.fields.inflation_delta) > 0.0


def test_recombination_uses_inflation_shaped_seed() -> None:
    context, report = _run_until_stage("recombination")

    assert report.metrics["cmb_contrast"] > 0.0
    assert np.any(context.fields.cmb)
    assert context.history.reports[1].stage_id == "inflation"
