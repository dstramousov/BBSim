"""Prototype recombination / CMB snapshot stage."""

from __future__ import annotations

from bbsim.core.context import UniverseRunContext
from bbsim.core.expansion import compute_hubble_gyr_inv, detect_era
from bbsim.core.report import StageReport


class RecombinationPreviewStage:
    """Create a CMB-like snapshot from the primordial fluctuation field."""

    stage_id = "recombination_preview"
    title = "Рекомбинация / CMB"

    def enter(self, context: UniverseRunContext) -> None:
        """Create the CMB-like field snapshot."""

        context.state.current_stage = self.stage_id
        context.state.a = max(context.state.a, 9.0e-4)
        context.state.h_gyr_inv = compute_hubble_gyr_inv(context.state.a, context.config.cosmology)
        context.state.era = detect_era(context.state.a, context.config.cosmology)
        context.state.temperature_k = 3000.0
        context.fields.cmb = context.backend.normalize_field(context.fields.seed_delta)
        context.state.stage_progress = 1.0
        context.state.a_history.append(context.state.a)
        context.state.t_history.append(context.state.t_gyr)

    def step(self, context: UniverseRunContext, dt: float) -> None:
        """No-op because the CMB snapshot is an immediate checkpoint."""

        _ = (context, dt)

    def is_complete(self, context: UniverseRunContext) -> bool:
        """Return true after the CMB-like snapshot is created."""

        return context.state.stage_progress >= 1.0

    def build_report(self, context: UniverseRunContext) -> StageReport:
        """Build the recombination preview checkpoint report."""

        return StageReport(
            stage_id=self.stage_id,
            title="Реликтовый отпечаток готов",
            summary_lines=(
                f"Эпоха: {context.state.era}",
                f"Температура: {context.state.temperature_k:.0f} K",
                f"H(a): {context.state.h_gyr_inv:.3e} 1/Gyr",
                "CMB-like карта построена из личного зерна флуктуаций.",
            ),
            metrics={"temperature_k": context.state.temperature_k, "h": context.state.h_gyr_inv},
        )
