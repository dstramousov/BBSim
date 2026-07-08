from __future__ import annotations

from bbsim.core.timeline import DEFAULT_TIMELINE_STAGES, timeline_progress_position, timeline_stage_index


def test_timeline_stage_index() -> None:
    assert timeline_stage_index("personal_seed") == 0
    assert timeline_stage_index("inflation") == 1
    assert timeline_stage_index("missing") is None


def test_seed_checkpoint_is_timeline_start() -> None:
    assert timeline_progress_position("personal_seed", 1.0) == 0.0


def test_running_stage_moves_between_previous_and_current_checkpoint() -> None:
    stages = DEFAULT_TIMELINE_STAGES
    inflation_start = timeline_progress_position("inflation", 0.0)
    inflation_half = timeline_progress_position("inflation", 0.5)
    inflation_end = timeline_progress_position("inflation", 1.0)

    assert inflation_start == 0.0
    assert 0.0 < inflation_half < inflation_end
    assert inflation_end == 1.0 / (len(stages) - 1)


def test_progress_is_clamped() -> None:
    assert timeline_progress_position("inflation", -1.0) == 0.0
    assert timeline_progress_position("inflation", 2.0) == 1.0 / (len(DEFAULT_TIMELINE_STAGES) - 1)
