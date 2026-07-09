"""Timeline model for visual pipeline progress."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimelineStage:
    """One labeled checkpoint on the universe evolution timeline."""

    stage_id: str
    title: str


DEFAULT_TIMELINE_STAGES: tuple[TimelineStage, ...] = (
    TimelineStage("personal_seed", "Зерно"),
    TimelineStage("inflation", "Инфляция"),
    TimelineStage("reheating", "Разогрев"),
    TimelineStage("nucleosynthesis", "Нуклеосинтез"),
    TimelineStage("recombination", "CMB"),
    TimelineStage("dark_ages", "Тёмные века"),
    TimelineStage("first_stars", "Первые звёзды"),
    TimelineStage("galaxy_web", "Галактики"),
    TimelineStage("dark_energy", "Тёмная энергия"),
    TimelineStage("fate", "Судьба"),
)


def timeline_stage_index(
    stage_id: str | None,
    stages: tuple[TimelineStage, ...] = DEFAULT_TIMELINE_STAGES,
) -> int | None:
    """Return the index of a stage in the visual timeline."""

    if stage_id is None:
        return None
    for index, stage in enumerate(stages):
        if stage.stage_id == stage_id:
            return index
    return None


def timeline_progress_position(
    stage_id: str | None,
    local_stage_progress: float,
    stages: tuple[TimelineStage, ...] = DEFAULT_TIMELINE_STAGES,
) -> float:
    """Return normalized visual progress on the full timeline.

    The stage marker represents the checkpoint at the end of that stage. Therefore a
    running stage moves from the previous checkpoint to the current checkpoint.
    """

    if not stages:
        return 0.0
    stage_index = timeline_stage_index(stage_id, stages)
    if stage_index is None:
        return 0.0
    if len(stages) == 1:
        return 1.0

    clamped_local_progress = max(0.0, min(1.0, local_stage_progress))
    if stage_index <= 0:
        return 0.0

    position = (stage_index - 1 + clamped_local_progress) / (len(stages) - 1)
    return max(0.0, min(1.0, position))
