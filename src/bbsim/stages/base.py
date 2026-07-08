"""Base protocol for pipeline stages."""

from __future__ import annotations

from typing import Protocol

from bbsim.core.context import UniverseRunContext
from bbsim.core.report import StageReport


class SimulationStage(Protocol):
    """Protocol implemented by all pipeline stages."""

    stage_id: str
    title: str

    def enter(self, context: UniverseRunContext) -> None:
        """Initialize the stage before stepping."""
        ...

    def step(self, context: UniverseRunContext, dt: float) -> None:
        """Advance the stage by a visual timestep."""
        ...

    def is_complete(self, context: UniverseRunContext) -> bool:
        """Return whether the stage reached its checkpoint."""
        ...

    def build_report(self, context: UniverseRunContext) -> StageReport:
        """Build the checkpoint report for the current stage."""
        ...
