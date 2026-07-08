"""Prototype recombination / CMB snapshot stage."""

from __future__ import annotations

import numpy as np

from bbsim.core.context import UniverseRunContext
from bbsim.core.expansion import compute_hubble_gyr_inv, detect_era
from bbsim.core.report import StageReport


class RecombinationPreviewStage:
    """Create a CMB-like snapshot from the inflation-shaped fluctuation field."""

    stage_id = "recombination_preview"
    title = "Рекомбинация / CMB"

    def enter(self, context: UniverseRunContext) -> None:
        """Create the CMB-like field snapshot."""

        context.state.current_stage = self.stage_id
        context.state.a = max(context.state.a, 9.0e-4)
        context.state.h_gyr_inv = compute_hubble_gyr_inv(context.state.a, context.config.cosmology)
        context.state.era = detect_era(context.state.a, context.config.cosmology)
        context.state.temperature_k = 3000.0

        source = context.fields.inflation_delta
        if not np.any(source):
            source = context.fields.seed_delta
        cmb = context.backend.apply_inflation_smoothing(source, smoothing=0.45)
        context.fields.cmb = context.backend.normalize_field(cmb)

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

        cmb_contrast = float(np.std(context.fields.cmb))
        return StageReport(
            stage_id=self.stage_id,
            title="Реликтовый отпечаток готов",
            summary_lines=(
                f"Эпоха: {context.state.era}",
                f"Температура: {context.state.temperature_k:.0f} K",
                f"H(a): {context.state.h_gyr_inv:.3e} 1/Gyr",
                f"Контраст CMB-like отпечатка: {cmb_contrast:.2f}",
                "Карта построена из личного зерна после инфляционного сглаживания.",
            ),
            metrics={
                "temperature_k": context.state.temperature_k,
                "h": context.state.h_gyr_inv,
                "cmb_contrast": cmb_contrast,
            },
        )
