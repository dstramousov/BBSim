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
    TimelineStage("gas_collapse", "Сжатие газа"),
    TimelineStage("first_stars", "Первые звёзды"),
    TimelineStage("reionization", "Реионизация"),
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


def timeline_checkpoint_position(
    stage_id: str | None,
    stages: tuple[TimelineStage, ...] = DEFAULT_TIMELINE_STAGES,
) -> float:
    """Return normalized position of a stage checkpoint on the full timeline."""

    stage_index = timeline_stage_index(stage_id, stages)
    if stage_index is None or not stages:
        return 0.0
    return (stage_index + 1) / len(stages)


def timeline_progress_position(
    stage_id: str | None,
    local_stage_progress: float,
    stages: tuple[TimelineStage, ...] = DEFAULT_TIMELINE_STAGES,
) -> float:
    """Return normalized visual progress on the full timeline.

    The line starts before the first checkpoint, so the seed reveal itself is visible:
    the marker moves from the left edge to the `Зерно` checkpoint instead of looking
    frozen at the first dot. Each later stage moves from the previous checkpoint to
    the current one.
    """

    if not stages:
        return 0.0
    stage_index = timeline_stage_index(stage_id, stages)
    if stage_index is None:
        return 0.0

    clamped_local_progress = max(0.0, min(1.0, local_stage_progress))
    position = (stage_index + clamped_local_progress) / len(stages)
    return max(0.0, min(1.0, position))
