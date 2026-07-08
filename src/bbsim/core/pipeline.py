"""Pipeline runner for universe evolution stages."""

from __future__ import annotations

from dataclasses import dataclass

from bbsim.core.context import UniverseRunContext
from bbsim.core.report import StageReport
from bbsim.stages.base import SimulationStage
from bbsim.stages.early_universe import NucleosynthesisStage, ReheatingStage
from bbsim.stages.inflation import InflationStage
from bbsim.stages.personal_seed import PersonalSeedStage
from bbsim.stages.recombination import RecombinationPreviewStage


@dataclass(slots=True)
class UniversePipeline:
    """Ordered stage pipeline for one universe run."""

    stages: list[SimulationStage]
    current_index: int = 0
    _entered: bool = False

    @property
    def is_finished(self) -> bool:
        """Return whether all stages are complete."""

        return self.current_index >= len(self.stages)

    @property
    def current_stage(self) -> SimulationStage | None:
        """Return the current stage, or None when the pipeline is finished."""

        if self.is_finished:
            return None
        return self.stages[self.current_index]

    def enter_current(self, context: UniverseRunContext) -> None:
        """Enter the current stage if it has not been entered yet."""

        stage = self.current_stage
        if stage is None or self._entered:
            return
        stage.enter(context)
        self._entered = True

    def step(self, context: UniverseRunContext, dt: float) -> None:
        """Advance the current stage by one timestep."""

        self.enter_current(context)
        stage = self.current_stage
        if stage is None:
            return
        stage.step(context, dt)

    def step_live(self, context: UniverseRunContext, dt: float) -> StageReport | None:
        """Advance the current stage once and return a report at checkpoint.

        Unlike :meth:`step_to_checkpoint`, this method does not fast-forward. It is
        intended for GUI playback where every tick should update fields, graphs, and
        timeline position so the user can watch an epoch evolve continuously.
        """

        self.enter_current(context)
        stage = self.current_stage
        if stage is None:
            return None

        stage.step(context, dt)
        if not stage.is_complete(context):
            return None

        report = stage.build_report(context)
        context.history.add_report(report)
        return report

    def step_to_checkpoint(self, context: UniverseRunContext) -> None:
        """Advance the current stage until it reaches its checkpoint."""

        self.enter_current(context)
        stage = self.current_stage
        if stage is None:
            return
        guard = 0
        while not stage.is_complete(context):
            stage.step(context, dt=1.0)
            guard += 1
            if guard > 10000:
                raise RuntimeError(f"stage did not complete: {stage.stage_id}")
        context.history.add_report(stage.build_report(context))

    def advance(self, context: UniverseRunContext) -> None:
        """Move to the next stage and enter it when available."""

        _ = context
        if not self.is_finished:
            self.current_index += 1
        self._entered = False


def create_default_pipeline() -> UniversePipeline:
    """Create the initial prototype pipeline."""

    return UniversePipeline(
        stages=[
            PersonalSeedStage(),
            InflationStage(),
            ReheatingStage(),
            NucleosynthesisStage(),
            RecombinationPreviewStage(),
        ]
    )
