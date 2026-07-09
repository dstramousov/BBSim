"""Live early-universe stages between inflation and recombination."""

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
    """Interpolate positive values on a logarithmic scale."""

    safe_start = max(float(start), 1.0e-40)
    safe_end = max(float(end), safe_start * 1.0e-6)
    clamped = float(np.clip(progress, 0.0, 1.0))
    return math.exp(math.log(safe_start) + (math.log(safe_end) - math.log(safe_start)) * clamped)


def _source_after_inflation(context: UniverseRunContext) -> np.ndarray:
    source = context.fields.inflation_delta
    if np.any(source):
        return source
    return context.fields.seed_delta


class ReheatingStage:
    """Convert the post-inflation field into a hot radiation-dominated plasma."""

    stage_id = "reheating"
    title = "Разогрев"

    def __init__(self) -> None:
        self._elapsed_s = 0.0
        self._initial_a = 1.0e-32
        self._initial_t_gyr = 0.0
        self._source: np.ndarray | None = None
        self._thermal_pattern: np.ndarray | None = None

    def enter(self, context: UniverseRunContext) -> None:
        """Prepare the live reheating plasma field."""

        context.state.current_stage = self.stage_id
        context.state.era = "radiation"
        context.state.stage_progress = 0.0
        self._elapsed_s = 0.0
        self._initial_a = max(context.state.a, 1.0e-32)
        self._initial_t_gyr = max(context.state.t_gyr, 1.0e-40)

        self._source = context.backend.normalize_field(_source_after_inflation(context))
        rolled = np.roll(self._source, 7, axis=0) - np.roll(self._source, -5, axis=1)
        self._thermal_pattern = context.backend.normalize_field(rolled)
        context.fields.radiation = self._source.astype(np.float32, copy=True)
        ExpansionEngine.update_state(context.state, context.config.cosmology)

    def step(self, context: UniverseRunContext, dt: float) -> None:
        """Advance the visible hot plasma transition."""

        if self._source is None or self._thermal_pattern is None:
            raise RuntimeError("reheating stage entered without source fields")

        duration = max(
            stage_screen_duration_s(
                context.config,
                self.stage_id,
                context.config.early_universe.reheating_visual_duration_s,
            ),
            1.0e-6,
        )
        self._elapsed_s = min(duration, self._elapsed_s + max(dt, 0.0))
        progress = self._elapsed_s / duration
        visible = _smoothstep(progress)

        phase = visible * math.tau
        thermal_wave = np.sin(self._source * 9.0 + phase) + np.cos(self._thermal_pattern * 7.0 - phase)
        thermal_wave = context.backend.normalize_field(thermal_wave)
        smoothed_source = context.backend.diffuse(self._source, amount=0.30 + 0.35 * visible)
        plasma = (1.0 - visible) * smoothed_source + visible * thermal_wave
        context.fields.radiation = context.backend.normalize_field(plasma)

        target_a = max(self._initial_a * 1.0e3, 1.0e-18)
        context.state.a = _log_lerp(self._initial_a, target_a, visible)
        context.state.t_gyr = _log_lerp(self._initial_t_gyr, 1.0e-18, visible)
        context.state.temperature_k = _log_lerp(1.0e15, 1.0e9, visible)
        context.state.stage_progress = progress
        ExpansionEngine.update_state(context.state, context.config.cosmology)

    def is_complete(self, context: UniverseRunContext) -> bool:
        """Return true when reheating reaches its checkpoint."""

        return context.state.stage_progress >= 1.0

    def build_report(self, context: UniverseRunContext) -> StageReport:
        """Build the reheating checkpoint report."""

        radiation_contrast = float(np.std(context.fields.radiation))
        scale = sample_scale(context.state, context.config)
        return StageReport(
            stage_id=self.stage_id,
            title="Разогрев завершён",
            summary_lines=(
                "Энергия инфляции переведена в горячую радиационно-доминирующую плазму.",
                f"Температура: {context.state.temperature_k:.2e} K",
                f"Масштаб a(t): {context.state.a:.3e}",
                f"Видимый участок сейчас: {scale.box_now_text}",
                f"Эпоха по плотностям: {context.state.era}",
                f"Доля radiation: {context.state.frac_r:.2f}",
                f"Доля matter: {context.state.frac_m:.2f}",
                f"Доля dark energy: {context.state.frac_lambda:.2f}",
                f"Контраст горячей плазмы: {radiation_contrast:.2f}",
                "Рост структуры пока подавлен: радиация держит обычную материю в горячем супе.",
            ),
            metrics={
                "temperature_k": context.state.temperature_k,
                "scale_factor": context.state.a,
                "radiation_contrast": radiation_contrast,
            },
        )


class NucleosynthesisStage:
    """Form the first light nuclei while the hot plasma cools."""

    stage_id = "nucleosynthesis"
    title = "Нуклеосинтез"

    def __init__(self) -> None:
        self._elapsed_s = 0.0
        self._initial_a = 1.0e-18
        self._initial_t_gyr = 1.0e-18
        self._source: np.ndarray | None = None
        self._target: np.ndarray | None = None

    def enter(self, context: UniverseRunContext) -> None:
        """Prepare the live light-element formation stage."""

        context.state.current_stage = self.stage_id
        context.state.era = "radiation"
        context.state.stage_progress = 0.0
        self._elapsed_s = 0.0
        self._initial_a = max(context.state.a, 1.0e-32)
        self._initial_t_gyr = max(context.state.t_gyr, 1.0e-40)

        source = context.fields.radiation
        if not np.any(source):
            source = _source_after_inflation(context)
        self._source = context.backend.normalize_field(source)
        self._target = context.backend.normalize_field(context.backend.diffuse(self._source, amount=0.82))
        ExpansionEngine.update_state(context.state, context.config.cosmology)

    def step(self, context: UniverseRunContext, dt: float) -> None:
        """Advance cooling plasma and light-element fractions."""

        if self._source is None or self._target is None:
            raise RuntimeError("nucleosynthesis stage entered without source fields")

        duration = max(
            stage_screen_duration_s(
                context.config,
                self.stage_id,
                context.config.early_universe.nucleosynthesis_visual_duration_s,
            ),
            1.0e-6,
        )
        self._elapsed_s = min(duration, self._elapsed_s + max(dt, 0.0))
        progress = self._elapsed_s / duration
        visible = _smoothstep(progress)

        cooling_texture = (1.0 - visible) * self._source + visible * self._target
        pulse = 0.10 * math.sin(visible * math.tau * 2.0)
        context.fields.radiation = context.backend.normalize_field(cooling_texture + pulse * self._source)

        target_a = max(self._initial_a * 1.0e6, 1.0e-10)
        context.state.a = _log_lerp(self._initial_a, target_a, visible)
        context.state.t_gyr = _log_lerp(self._initial_t_gyr, 6.0e-15, visible)
        context.state.temperature_k = _log_lerp(1.0e9, 1.0e8, visible)
        context.state.stage_progress = progress

        context.state.hydrogen_fraction = 0.75 * visible
        context.state.helium_fraction = 0.25 * visible
        context.state.lithium_trace = 1.0e-9 * visible
        context.state.heavy_elements_fraction = 0.0
        ExpansionEngine.update_state(context.state, context.config.cosmology)

    def is_complete(self, context: UniverseRunContext) -> bool:
        """Return true when light-element fractions are stable."""

        return context.state.stage_progress >= 1.0

    def build_report(self, context: UniverseRunContext) -> StageReport:
        """Build the nucleosynthesis checkpoint report."""

        scale = sample_scale(context.state, context.config)
        return StageReport(
            stage_id=self.stage_id,
            title="Первичный состав готов",
            summary_lines=(
                "Горячая ранняя Вселенная сформировала первые лёгкие ядра.",
                f"Водород: {context.state.hydrogen_fraction * 100.0:.1f}%",
                f"Гелий: {context.state.helium_fraction * 100.0:.1f}%",
                f"Следы лития: {context.state.lithium_trace:.1e}",
                "Тяжёлые элементы: почти 0 — они появятся только в звёздах.",
                f"Температура: {context.state.temperature_k:.2e} K",
                f"Видимый участок сейчас: {scale.box_now_text}",
                f"Доминирующий компонент: {context.state.era}",
                f"Доля radiation: {context.state.frac_r:.2f}",
                f"Доля matter: {context.state.frac_m:.2f}",
                "Будущие звёзды получили основное топливо: водород.",
            ),
            metrics={
                "hydrogen_fraction": context.state.hydrogen_fraction,
                "helium_fraction": context.state.helium_fraction,
                "lithium_trace": context.state.lithium_trace,
                "temperature_k": context.state.temperature_k,
            },
        )
