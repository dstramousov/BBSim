from __future__ import annotations

from bbsim.core.timeline import (
    DEFAULT_TIMELINE_STAGES,
    timeline_checkpoint_position,
    timeline_progress_position,
    timeline_stage_index,
)


def test_timeline_stage_index() -> None:
    assert timeline_stage_index("personal_seed") == 0
    assert timeline_stage_index("inflation") == 1
    assert timeline_stage_index("missing") is None


def test_seed_stage_has_visible_progress_before_first_checkpoint() -> None:
    stages = DEFAULT_TIMELINE_STAGES
    seed_start = timeline_progress_position("personal_seed", 0.0)
    seed_half = timeline_progress_position("personal_seed", 0.5)
    seed_end = timeline_progress_position("personal_seed", 1.0)

    assert seed_start == 0.0
    assert 0.0 < seed_half < seed_end
    assert seed_end == 1.0 / len(stages)
    assert seed_end == timeline_checkpoint_position("personal_seed")


def test_running_stage_moves_between_previous_and_current_checkpoint() -> None:
    stages = DEFAULT_TIMELINE_STAGES
    inflation_start = timeline_progress_position("inflation", 0.0)
    inflation_half = timeline_progress_position("inflation", 0.5)
    inflation_end = timeline_progress_position("inflation", 1.0)

    assert inflation_start == timeline_checkpoint_position("personal_seed")
    assert inflation_start < inflation_half < inflation_end
    assert inflation_end == timeline_checkpoint_position("inflation")
    assert inflation_end == 2.0 / len(stages)


def test_progress_is_clamped() -> None:
    assert timeline_progress_position("inflation", -1.0) == timeline_checkpoint_position(
        "personal_seed"
    )
    assert timeline_progress_position("inflation", 2.0) == timeline_checkpoint_position(
        "inflation"
    )
