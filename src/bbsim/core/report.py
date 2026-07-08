"""Checkpoint report objects."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class StageReport:
    """Human-readable checkpoint report produced by a pipeline stage."""

    stage_id: str
    title: str
    summary_lines: tuple[str, ...]
    metrics: dict[str, float] = field(default_factory=dict)
