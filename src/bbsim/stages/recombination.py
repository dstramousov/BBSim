"""Prototype recombination / CMB snapshot stage."""

from __future__ import annotations

import numpy as np

from bbsim.core.context import UniverseRunContext
from bbsim.core.expansion import compute_hubble_gyr_inv, detect_era
from bbsim.core.report import StageReport


def _smoothstep(value: float) -> float:
    clamped = float(np.clip(value, 0.0, 1.0))
    return clamped * clamped * (3.0 - 2.0 * clamped)


class RecombinationPreviewStage:
    """Create a CMB-like snapshot from the cooled early radiation field."""

    stage_id = "recombination_preview"
    title = "Рекомбинация / CMB"

    def __init__(self) -> None:
        self._elapsed_s = 0.0
        self._initial_a = 1.0e-32
        self._source: np.ndarray | None = None
        self._target_cmb: np.ndarray | None = None

    def enter(self, context: UniverseRunContext) -> None:
        """Prepare a live transition from post-inflation field to CMB imprint."""

        context.state.current_stage = self.stage_id
        context.state.stage_progress = 0.0
        self._elapsed_s = 0.0
        self._initial_a = context.state.a

        source = context.fields.radiation
        if not np.any(source):
            source = context.fields.inflation_delta
        if not np.any(source):
            source = context.fields.seed_delta
        self._source = context.backend.normalize_field(source)

        imprint_source = context.fields.inflation_delta
        if not np.any(imprint_source):
            imprint_source = self._source
        target = context.backend.apply_inflation_smoothing(imprint_source, smoothing=0.45)
        self._target_cmb = context.backend.normalize_field(target)
        context.fields.cmb = self._source.astype(np.float32, copy=True)
        context.state.a_history.append(context.state.a)
        context.state.t_history.append(context.state.t_gyr)

    def step(self, context: UniverseRunContext, dt: float) -> None:
        """Advance the visible CMB transition over the stage duration."""

        if self._source is None or self._target_cmb is None:
            raise RuntimeError("recombination stage entered without source fields")

        duration = max(context.config.early_universe.recombination_visual_duration_s, 1.0e-6)
        self._elapsed_s = min(duration, self._elapsed_s + max(dt, 0.0))
        progress = self._elapsed_s / duration
        visible_progress = _smoothstep(progress)

        target_a = 9.0e-4
        live_a = self._initial_a + (target_a - self._initial_a) * visible_progress
        context.state.a = max(self._initial_a, live_a)
        context.state.h_gyr_inv = compute_hubble_gyr_inv(context.state.a, context.config.cosmology)
        context.state.era = detect_era(context.state.a, context.config.cosmology)
        context.state.temperature_k = 1.0e9 + (3000.0 - 1.0e9) * visible_progress
        context.state.stage_progress = progress

        live_cmb = (1.0 - visible_progress) * self._source + visible_progress * self._target_cmb
        context.fields.cmb = context.backend.normalize_field(live_cmb)
        context.state.a_history.append(context.state.a)
        context.state.t_history.append(context.state.t_gyr)

    def is_complete(self, context: UniverseRunContext) -> bool:
        """Return true after the CMB-like snapshot is complete."""

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
                "Горячая плазма стала прозрачной и сохранила рябь как CMB-like отпечаток.",
            ),
            metrics={
                "temperature_k": context.state.temperature_k,
                "h": context.state.h_gyr_inv,
                "cmb_contrast": cmb_contrast,
            },
        )
