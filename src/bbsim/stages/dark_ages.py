"""Dark ages stage: separate dark matter scaffold from baryonic gas."""

from __future__ import annotations

import math

import numpy as np

from bbsim.core.context import UniverseRunContext
from bbsim.core.expansion import ExpansionEngine
from bbsim.core.report import StageReport
from bbsim.core.scale import sample_scale
from bbsim.core.time_director import stage_screen_duration_s


def _smoothstep(value: float) -> float:
    clamped = float(np.clip(value, 0.0, 1.0))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def _log_lerp(start: float, end: float, progress: float) -> float:
    safe_start = max(float(start), 1.0e-40)
    safe_end = max(float(end), 1.0e-40)
    clamped = float(np.clip(progress, 0.0, 1.0))
    return math.exp(math.log(safe_start) + (math.log(safe_end) - math.log(safe_start)) * clamped)


def _delayed_progress(progress: float, start: float, end: float = 1.0) -> float:
    if progress <= start:
        return 0.0
    if progress >= end:
        return 1.0
    return _smoothstep((progress - start) / max(end - start, 1.0e-6))


def _has_signal(field: np.ndarray) -> bool:
    return bool(np.asarray(field).size and float(np.std(field)) > 1.0e-7)


def _blend(start: np.ndarray, end: np.ndarray, progress: float) -> np.ndarray:
    p = float(np.clip(progress, 0.0, 1.0))
    return ((1.0 - p) * start + p * end).astype(np.float32)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = np.asarray(a, dtype=np.float32).reshape(-1)
    b_flat = np.asarray(b, dtype=np.float32).reshape(-1)
    a_std = float(a_flat.std())
    b_std = float(b_flat.std())
    if a_std <= 1.0e-8 or b_std <= 1.0e-8:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


class DarkAgesStage:
    """Reveal the non-luminous matter scaffold after CMB release.

    This stage deliberately does not create stars yet. It separates two fields:
    dark matter becomes a sharper gravitational scaffold, while baryonic gas remains
    smoother and starts following the same wells with a visible delay.
    """

    stage_id = "dark_ages"
    title = "Тёмные века"

    def __init__(self) -> None:
        self._elapsed_s = 0.0
        self._initial_a = 9.0e-4
        self._initial_t_gyr = 3.8e-4
        self._initial_temperature_k = 3000.0
        self._initial_dark: np.ndarray | None = None
        self._initial_baryon: np.ndarray | None = None
        self._target_dark: np.ndarray | None = None
        self._target_baryon: np.ndarray | None = None
        self._cmb_source: np.ndarray | None = None

    def enter(self, context: UniverseRunContext) -> None:
        """Prepare dark matter and baryon fields from the CMB/imprint source."""

        context.state.current_stage = self.stage_id
        context.state.era = "dark_ages"
        context.state.stage_progress = 0.0
        self._elapsed_s = 0.0
        self._initial_a = max(context.state.a, 9.0e-4)
        self._initial_t_gyr = max(context.state.t_gyr, 3.8e-4)
        self._initial_temperature_k = max(context.state.temperature_k, 30.0)

        source = context.fields.cmb
        if not _has_signal(source):
            source = context.fields.inflation_delta
        if not _has_signal(source):
            source = context.fields.seed_delta
        self._cmb_source = context.backend.normalize_field(source)

        current_dark = context.fields.dark_density
        if not _has_signal(current_dark):
            current_dark = self._cmb_source
        current_baryon = context.fields.baryon_density
        if not _has_signal(current_baryon):
            current_baryon = context.backend.diffuse(self._cmb_source, amount=0.75)
        self._initial_dark = context.backend.normalize_field(current_dark)
        self._initial_baryon = context.backend.normalize_field(current_baryon)

        filament_seed = (
            self._cmb_source
            + 0.36 * np.roll(self._cmb_source, 5, axis=0)
            + 0.24 * np.roll(self._cmb_source, -7, axis=1)
            - 0.18 * np.roll(self._cmb_source, 11, axis=0)
        )
        scaffold = context.backend.normalize_field(filament_seed)
        scaffold = context.backend.diffuse(scaffold, amount=0.18)
        scaffold = context.backend.normalize_field(scaffold)
        contrast = 0.72 + 0.30 * context.config.structure.gravity_strength
        self._target_dark = (scaffold * contrast).astype(np.float32)

        gas = context.backend.diffuse(scaffold, amount=0.82)
        gas = context.backend.diffuse(gas, amount=min(0.95, 0.45 + context.config.structure.gas_pressure * 0.25))
        gas = context.backend.normalize_field(gas)
        self._target_baryon = (gas * (0.55 + 0.30 * context.config.structure.baryon_infall)).astype(
            np.float32
        )

        context.fields.dark_density = self._initial_dark.astype(np.float32, copy=True)
        context.fields.baryon_density = self._initial_baryon.astype(np.float32, copy=True)
        ExpansionEngine.update_state(context.state, context.config.cosmology)

    def step(self, context: UniverseRunContext, dt: float) -> None:
        """Grow the hidden scaffold and let gas begin to trace it."""

        if (
            self._initial_dark is None
            or self._initial_baryon is None
            or self._target_dark is None
            or self._target_baryon is None
            or self._cmb_source is None
        ):
            raise RuntimeError("dark ages stage entered without source fields")

        duration = max(stage_screen_duration_s(context.config, self.stage_id, 45.0), 1.0e-6)
        self._elapsed_s = min(duration, self._elapsed_s + max(dt, 0.0))
        progress = self._elapsed_s / duration
        dark_progress = _smoothstep(progress)
        baryon_progress = _delayed_progress(progress, start=0.25)

        # Let visible radiation die out while the non-luminous scaffold appears.
        context.fields.radiation = ((1.0 - dark_progress) * self._cmb_source).astype(np.float32)
        context.fields.dark_density = _blend(self._initial_dark, self._target_dark, dark_progress)
        context.fields.baryon_density = _blend(
            self._initial_baryon,
            self._target_baryon,
            baryon_progress,
        )

        context.state.a = _log_lerp(self._initial_a, 2.2e-2, dark_progress)
        context.state.t_gyr = _log_lerp(self._initial_t_gyr, 0.18, dark_progress)
        context.state.temperature_k = _log_lerp(self._initial_temperature_k, 60.0, dark_progress)
        context.state.stage_progress = progress
        context.state.ionization_fraction = 0.0
        context.state.opacity = 0.0
        ExpansionEngine.update_state(context.state, context.config.cosmology)

    def is_complete(self, context: UniverseRunContext) -> bool:
        """Return true after dark scaffold and gas lag are visible."""

        return context.state.stage_progress >= 1.0

    def build_report(self, context: UniverseRunContext) -> StageReport:
        """Build the dark-ages checkpoint report."""

        dark_contrast = float(np.std(context.fields.dark_density))
        baryon_contrast = float(np.std(context.fields.baryon_density))
        coupling = _correlation(context.fields.dark_density, context.fields.baryon_density)
        scale = sample_scale(context.state, context.config)
        return StageReport(
            stage_id=self.stage_id,
            title="Тёмные века: каркас проявился",
            summary_lines=(
                "CMB уже отпечатался, но звёзд ещё нет: Вселенная тёмная.",
                "Тёмная материя не светится, зато формирует гравитационный каркас.",
                "Обычный газ остаётся более гладким и только начинает подтягиваться к этим ямам.",
                f"Физическое время: {context.state.t_gyr:.3f} Gyr после Big Bang",
                f"Масштаб a(t): {context.state.a:.3e}",
                f"Температура фона: {context.state.temperature_k:.1f} K",
                f"Видимый участок сейчас: {scale.box_now_text}",
                f"Контраст тёмного каркаса: {dark_contrast:.2f}",
                f"Контраст газа: {baryon_contrast:.2f}",
                f"Связь газ ↔ тёмный каркас: {coupling:.2f}",
                "Следующий логичный этап: газ начнёт остывать и в первых плотных узлах зажгутся звёзды.",
            ),
            metrics={
                "dark_contrast": dark_contrast,
                "baryon_contrast": baryon_contrast,
                "dark_baryon_correlation": coupling,
                "temperature_k": context.state.temperature_k,
                "a": context.state.a,
                "t_gyr": context.state.t_gyr,
            },
        )
