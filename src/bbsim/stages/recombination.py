"""Live recombination and CMB release stage."""

from __future__ import annotations

import math

import numpy as np

from bbsim.core.context import UniverseRunContext
from bbsim.core.expansion import ExpansionEngine
from bbsim.core.report import StageReport


def _smoothstep(value: float) -> float:
    clamped = float(np.clip(value, 0.0, 1.0))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def _log_lerp(start: float, end: float, progress: float) -> float:
    """Interpolate positive values on a logarithmic scale."""

    safe_start = max(float(start), 1.0e-40)
    safe_end = max(float(end), 1.0e-40)
    clamped = float(np.clip(progress, 0.0, 1.0))
    return math.exp(math.log(safe_start) + (math.log(safe_end) - math.log(safe_start)) * clamped)


def _shifted_drop(progress: float, start: float = 0.35, end: float = 0.92) -> float:
    """Return a delayed smooth 0..1 transition for recombination itself."""

    if progress <= start:
        return 0.0
    if progress >= end:
        return 1.0
    return _smoothstep((progress - start) / (end - start))


class RecombinationStage:
    """Let the plasma become transparent and release a CMB-like imprint."""

    stage_id = "recombination"
    title = "Рекомбинация / CMB"

    def __init__(self) -> None:
        self._elapsed_s = 0.0
        self._initial_a = 1.0e-10
        self._initial_t_gyr = 1.0e-15
        self._initial_temperature_k = 1.0e8
        self._source: np.ndarray | None = None
        self._hazy_plasma: np.ndarray | None = None
        self._target_cmb: np.ndarray | None = None

    def enter(self, context: UniverseRunContext) -> None:
        """Prepare a live transition from opaque plasma to transparent CMB."""

        context.state.current_stage = self.stage_id
        context.state.era = "recombination"
        context.state.stage_progress = 0.0
        context.state.ionization_fraction = 1.0
        context.state.opacity = 1.0
        context.state.cmb_released = False
        self._elapsed_s = 0.0
        self._initial_a = max(context.state.a, 1.0e-32)
        self._initial_t_gyr = max(context.state.t_gyr, 1.0e-40)
        self._initial_temperature_k = max(context.state.temperature_k, 1.0e5)

        source = context.fields.radiation
        if not np.any(source):
            source = context.fields.inflation_delta
        if not np.any(source):
            source = context.fields.seed_delta
        self._source = context.backend.normalize_field(source)
        self._hazy_plasma = context.backend.normalize_field(
            context.backend.diffuse(self._source, amount=0.35) + 0.28 * self._source
        )

        imprint_source = context.fields.inflation_delta
        if not np.any(imprint_source):
            imprint_source = self._source
        target = context.backend.apply_inflation_smoothing(imprint_source, smoothing=0.48)
        target = context.backend.diffuse(target, amount=0.18)
        self._target_cmb = context.backend.normalize_field(target)

        context.fields.cmb = self._hazy_plasma.astype(np.float32, copy=True)
        context.fields.radiation = self._hazy_plasma.astype(np.float32, copy=True)
        ExpansionEngine.update_state(context.state, context.config.cosmology)

    def step(self, context: UniverseRunContext, dt: float) -> None:
        """Advance ionization loss, opacity drop and visible CMB release."""

        if self._source is None or self._hazy_plasma is None or self._target_cmb is None:
            raise RuntimeError("recombination stage entered without source fields")

        duration = max(context.config.early_universe.recombination_visual_duration_s, 1.0e-6)
        self._elapsed_s = min(duration, self._elapsed_s + max(dt, 0.0))
        progress = self._elapsed_s / duration
        cooling_progress = _smoothstep(progress)
        recombination_progress = _shifted_drop(cooling_progress)

        target_a = 9.0e-4
        context.state.a = _log_lerp(self._initial_a, target_a, cooling_progress)
        context.state.t_gyr = _log_lerp(self._initial_t_gyr, 3.8e-4, cooling_progress)
        context.state.temperature_k = _log_lerp(self._initial_temperature_k, 3000.0, cooling_progress)

        ionization = 1.0 - recombination_progress
        opacity = ionization**1.7
        transparency = 1.0 - opacity
        context.state.ionization_fraction = float(np.clip(ionization, 0.0, 1.0))
        context.state.opacity = float(np.clip(opacity, 0.0, 1.0))
        context.state.cmb_released = recombination_progress >= 0.995
        context.state.stage_progress = progress

        residual_flicker = 0.14 * opacity * np.sin(self._source * 8.0 + progress * math.tau * 1.5)
        opaque_plasma = context.backend.normalize_field(self._hazy_plasma + residual_flicker)
        live_cmb = (1.0 - transparency) * opaque_plasma + transparency * self._target_cmb
        context.fields.cmb = context.backend.normalize_field(live_cmb)
        context.fields.radiation = context.fields.cmb.astype(np.float32, copy=True)
        ExpansionEngine.update_state(context.state, context.config.cosmology)

    def is_complete(self, context: UniverseRunContext) -> bool:
        """Return true after the CMB imprint has been released."""

        return context.state.stage_progress >= 1.0

    def build_report(self, context: UniverseRunContext) -> StageReport:
        """Build the recombination checkpoint report."""

        cmb_contrast = float(np.std(context.fields.cmb))
        return StageReport(
            stage_id=self.stage_id,
            title="Рекомбинация завершена",
            summary_lines=(
                "Электроны связались с ядрами: горячая плазма стала прозрачной.",
                f"Температура: {context.state.temperature_k:.0f} K",
                f"Масштаб a(t): {context.state.a:.3e}",
                f"Ионизация: {context.state.ionization_fraction:.3f}",
                f"Непрозрачность: {context.state.opacity:.3f}",
                f"H(a): {context.state.h_gyr_inv:.3e} 1/Gyr",
                f"Доминирующий компонент: {context.state.era}",
                f"Доля radiation: {context.state.frac_r:.2f}",
                f"Доля matter: {context.state.frac_m:.2f}",
                f"Контраст CMB-отпечатка: {cmb_contrast:.2f}",
                "CMB стал снимком младенческой Вселенной: рябь сохранена как будущие семена структуры.",
                "Дальше начнутся тёмные века: звёзд ещё нет, но тёмный каркас уже задан.",
            ),
            metrics={
                "temperature_k": context.state.temperature_k,
                "h": context.state.h_gyr_inv,
                "ionization_fraction": context.state.ionization_fraction,
                "opacity": context.state.opacity,
                "cmb_released": context.state.cmb_released,
                "cmb_contrast": cmb_contrast,
            },
        )
