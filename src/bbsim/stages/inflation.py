"""Prototype inflation stage."""

from __future__ import annotations

import math

from bbsim.core.context import UniverseRunContext
from bbsim.core.report import StageReport


class InflationStage:
    """Apply prototype exponential expansion and curvature smoothing."""

    stage_id = "inflation"
    title = "Инфляция"

    def __init__(self) -> None:
        self._elapsed_s = 0.0
        self._initial_a = 1.0e-6
        self._initial_curvature = 0.0

    def enter(self, context: UniverseRunContext) -> None:
        """Capture initial state for the inflation stage."""

        context.state.current_stage = self.stage_id
        context.state.era = "inflation"
        context.state.stage_progress = 0.0
        self._elapsed_s = 0.0
        self._initial_a = context.state.a
        self._initial_curvature = context.state.curvature

    def step(self, context: UniverseRunContext, dt: float) -> None:
        """Advance the prototype inflation visual stage."""

        duration = max(context.config.inflation.visual_duration_s, 1.0e-6)
        self._elapsed_s = min(duration, self._elapsed_s + max(dt, 0.0))
        progress = self._elapsed_s / duration
        params = context.config.inflation
        n_e_folds = params.strength * params.duration * 60.0
        context.state.a = self._initial_a * math.exp(n_e_folds * progress)
        context.state.curvature = self._initial_curvature * math.exp(-params.smoothing * 10.0 * progress)
        context.state.stage_progress = progress
        context.state.a_history.append(context.state.a)
        context.state.t_history.append(context.state.t_gyr)

    def is_complete(self, context: UniverseRunContext) -> bool:
        """Return true when the visual inflation stage is complete."""

        return context.state.stage_progress >= 1.0

    def build_report(self, context: UniverseRunContext) -> StageReport:
        """Build the inflation checkpoint report."""

        params = context.config.inflation
        n_e_folds = params.strength * params.duration * 60.0
        return StageReport(
            stage_id=self.stage_id,
            title="Инфляция завершена",
            summary_lines=(
                f"Инфляционное растяжение: e^{n_e_folds:.1f}",
                f"Масштаб a(t): {context.state.a:.3e}",
                "Кривизна сглажена, первичная рябь сохранена как seed будущей структуры.",
            ),
            metrics={"n_e_folds": n_e_folds, "scale_factor": context.state.a},
        )
