"""Reionization stage: first-star radiation turns neutral gas transparent again."""

from __future__ import annotations

import numpy as np

from bbsim.core.context import UniverseRunContext
from bbsim.core.expansion import ExpansionEngine
from bbsim.core.report import StageReport
from bbsim.core.scale import sample_scale
from bbsim.core.time_director import stage_screen_duration_s
from bbsim.stages.dark_ages import (
    _blend,
    _correlation,
    _delayed_progress,
    _has_signal,
    _log_lerp,
    _mass_fraction_above,
    _normalize01,
    _smoothstep,
)


def _area_fraction_above(field01: np.ndarray, threshold: float) -> float:
    """Return area fraction above a threshold for a normalized field."""

    data = np.clip(np.asarray(field01, dtype=np.float32), 0.0, 1.0)
    if data.size == 0:
        return 0.0
    return float(np.mean(data >= float(threshold)))


class ReionizationStage:
    """Grow ionized bubbles from first stars until they overlap into a clear IGM.

    This stage keeps the first-star payoff but adds the causal bridge after it: the
    earliest ultraviolet sources heat and ionize surrounding neutral hydrogen. The
    model is deliberately field-based and approximate; it tracks global ionized and
    neutral fractions instead of doing radiative transfer.
    """

    stage_id = "reionization"
    title = "Реионизация"

    def __init__(self) -> None:
        self._elapsed_s = 0.0
        self._initial_a = 4.6e-2
        self._initial_t_gyr = 0.45
        self._initial_temperature_k = 18.0
        self._initial_cold_gas: np.ndarray | None = None
        self._initial_baryon: np.ndarray | None = None
        self._initial_ionization: np.ndarray | None = None
        self._star_source: np.ndarray | None = None
        self._radiation_seed: np.ndarray | None = None
        self._bubble_mid: np.ndarray | None = None
        self._bubble_target: np.ndarray | None = None
        self._heated_gas_target: np.ndarray | None = None

    def enter(self, context: UniverseRunContext) -> None:
        """Prepare ionization fronts from the first-star radiation field."""

        context.state.current_stage = self.stage_id
        context.state.era = "reionization"
        context.state.stage_progress = 0.0
        self._elapsed_s = 0.0
        self._initial_a = max(context.state.a, 4.6e-2)
        self._initial_t_gyr = max(context.state.t_gyr, 0.45)
        self._initial_temperature_k = max(context.state.temperature_k, 8.0)

        stars = context.fields.first_star_density
        if not _has_signal(stars):
            stars = context.fields.stars
        if not _has_signal(stars):
            stars = context.fields.collapse_sites
        self._star_source = _normalize01(stars)

        radiation = context.fields.stellar_radiation
        if not _has_signal(radiation):
            radiation = context.backend.diffuse(self._star_source, amount=0.42)
        radiation01 = _normalize01(radiation)

        existing_bubbles = context.fields.ionized_bubbles
        if not _has_signal(existing_bubbles):
            existing_bubbles = context.fields.ionization
        if not _has_signal(existing_bubbles):
            existing_bubbles = np.zeros_like(self._star_source, dtype=np.float32)
        self._initial_ionization = _normalize01(existing_bubbles)

        cold_gas = context.fields.cold_gas_density
        if not _has_signal(cold_gas):
            cold_gas = context.fields.baryon_density
        self._initial_cold_gas = _normalize01(cold_gas)

        baryon = context.fields.baryon_density
        if not _has_signal(baryon):
            baryon = self._initial_cold_gas
        self._initial_baryon = _normalize01(baryon)

        # Narrow first-star bubbles become wider radiation fronts; later they
        # overlap into an ionized network. The second and third diffusions emulate
        # photon reach without pretending to solve radiative transfer.
        narrow = context.backend.diffuse(0.58 * radiation01 + 0.42 * self._star_source, amount=0.52)
        middle = context.backend.diffuse(narrow, amount=0.64)
        wide = context.backend.diffuse(middle, amount=0.78)
        very_wide = context.backend.diffuse(wide, amount=0.66)

        self._radiation_seed = _normalize01(0.55 * radiation01 + 0.45 * middle)
        self._bubble_mid = _normalize01(0.45 * self._initial_ionization + 0.55 * middle)
        self._bubble_target = _normalize01(
            0.25 * self._initial_ionization + 0.36 * wide + 0.39 * very_wide
        )

        feedback = float(np.clip(context.config.structure.feedback_strength, 0.0, 2.0))
        gas_suppression = np.clip(self._bubble_target * (0.30 + 0.18 * feedback), 0.0, 0.72)
        self._heated_gas_target = np.clip(
            self._initial_cold_gas * (1.0 - gas_suppression),
            0.0,
            1.0,
        ).astype(np.float32)

        context.fields.stellar_radiation = self._radiation_seed.astype(np.float32, copy=True)
        context.fields.ionized_bubbles = self._initial_ionization.astype(np.float32, copy=True)
        context.fields.ionization = self._initial_ionization.astype(np.float32, copy=True)
        self._update_reionization_metrics(context, bubble_progress=0.0, overlap_progress=0.0)
        ExpansionEngine.update_state(context.state, context.config.cosmology)

    def step(self, context: UniverseRunContext, dt: float) -> None:
        """Expand ionization fronts and heat neutral gas around first stars."""

        if (
            self._initial_cold_gas is None
            or self._initial_baryon is None
            or self._initial_ionization is None
            or self._star_source is None
            or self._radiation_seed is None
            or self._bubble_mid is None
            or self._bubble_target is None
            or self._heated_gas_target is None
        ):
            raise RuntimeError("reionization stage entered without first-star fields")

        duration = max(stage_screen_duration_s(context.config, self.stage_id, 64.0), 1.0e-6)
        self._elapsed_s = min(duration, self._elapsed_s + max(dt, 0.0))
        progress = self._elapsed_s / duration
        radiation_progress = _delayed_progress(progress, start=0.04, end=0.72)
        bubble_progress = _delayed_progress(progress, start=0.10, end=0.88)
        overlap_progress = _delayed_progress(progress, start=0.42, end=1.0)
        heating_progress = _delayed_progress(progress, start=0.28, end=1.0)

        flicker = 0.96 + 0.04 * np.sin(progress * np.pi * 14.0)
        radiation = np.clip(
            self._radiation_seed * (0.45 + 0.55 * radiation_progress) * flicker,
            0.0,
            1.0,
        )
        growing_bubbles = _blend(self._initial_ionization, self._bubble_mid, bubble_progress)
        overlapping_bubbles = _blend(growing_bubbles, self._bubble_target, overlap_progress)
        ionization = np.clip(
            overlapping_bubbles + radiation * (0.12 * bubble_progress) + self._star_source * 0.06,
            0.0,
            1.0,
        )

        context.fields.stellar_radiation = radiation.astype(np.float32)
        context.fields.ionized_bubbles = ionization.astype(np.float32)
        context.fields.ionization = ionization.astype(np.float32)
        context.fields.radiation = radiation.astype(np.float32)
        context.fields.cold_gas_density = _blend(
            self._initial_cold_gas,
            self._heated_gas_target,
            heating_progress,
        ).astype(np.float32)
        context.fields.baryon_density = _blend(
            self._initial_baryon,
            context.backend.diffuse(self._initial_baryon, amount=0.56),
            0.22 * heating_progress,
        ).astype(np.float32)

        smooth_progress = _smoothstep(progress)
        context.state.a = _log_lerp(self._initial_a, 9.0e-2, smooth_progress)
        context.state.t_gyr = _log_lerp(self._initial_t_gyr, 1.05, smooth_progress)
        context.state.temperature_k = _log_lerp(self._initial_temperature_k, 9.0, smooth_progress)
        context.state.gas_temperature_k = _log_lerp(900.0, 12000.0, heating_progress)
        context.state.heavy_elements_fraction = min(
            5.0e-5,
            max(
                context.state.heavy_elements_fraction,
                1.0e-5 + 4.0e-5 * radiation_progress * context.config.structure.metal_yield,
            ),
        )
        context.state.stage_progress = progress
        self._update_reionization_metrics(
            context,
            bubble_progress=bubble_progress,
            overlap_progress=overlap_progress,
        )
        ExpansionEngine.update_state(context.state, context.config.cosmology)

    def is_complete(self, context: UniverseRunContext) -> bool:
        """Return true when ionized bubbles have mostly overlapped."""

        return context.state.stage_progress >= 1.0

    def build_report(self, context: UniverseRunContext) -> StageReport:
        """Build the reionization checkpoint report."""

        coupling = _correlation(context.fields.ionization, context.fields.first_star_density)
        scale = sample_scale(context.state, context.config)
        return StageReport(
            stage_id=self.stage_id,
            title="Реионизация: тёмный газ прорезан светом",
            summary_lines=(
                "Свет первых звёзд расширил ионизированные пузыри вокруг ранних очагов.",
                "Пузыри начали соединяться: нейтральный водород больше не заполняет весь межгалактический газ.",
                "Это мост между первым светом и будущей галактической паутиной, а не отдельный фейерверк.",
                f"Физическое время: {context.state.t_gyr:.3f} Gyr после Big Bang",
                f"Масштаб a(t): {context.state.a:.3e}",
                f"Температура фона: {context.state.temperature_k:.1f} K",
                f"Температура нагретого газа: {context.state.gas_temperature_k:.0f} K",
                f"Видимый участок сейчас: {scale.box_now_text}",
                f"Ионизированная доля: {context.state.ionized_fraction:.1%}",
                f"Нейтральная доля: {context.state.neutral_fraction:.1%}",
                f"Перекрытие пузырей: {context.state.bubble_overlap_fraction:.1%}",
                f"Photoheating feedback: {context.state.photoheating_feedback:.2f}",
                f"Связь ионизации ↔ первые звёзды: {coupling:.2f}",
                "Следующий логичный этап: галактическая паутина, где светящиеся узлы станут галактиками.",
            ),
            metrics={
                "ionized_fraction": context.state.ionized_fraction,
                "neutral_fraction": context.state.neutral_fraction,
                "bubble_overlap_fraction": context.state.bubble_overlap_fraction,
                "photoheating_feedback": context.state.photoheating_feedback,
                "reionization_progress": context.state.reionization_progress,
                "reionization_star_correlation": coupling,
                "gas_temperature_k": context.state.gas_temperature_k,
                "temperature_k": context.state.temperature_k,
                "a": context.state.a,
                "t_gyr": context.state.t_gyr,
            },
        )

    def _update_reionization_metrics(
        self,
        context: UniverseRunContext,
        *,
        bubble_progress: float,
        overlap_progress: float,
    ) -> None:
        ionization = (
            _normalize01(context.fields.ionization)
            if _has_signal(context.fields.ionization)
            else np.zeros_like(context.fields.first_star_density)
        )
        radiation = (
            _normalize01(context.fields.stellar_radiation)
            if _has_signal(context.fields.stellar_radiation)
            else np.zeros_like(ionization)
        )
        cold_gas = (
            _normalize01(context.fields.cold_gas_density)
            if _has_signal(context.fields.cold_gas_density)
            else np.zeros_like(ionization)
        )

        ionized_area = _area_fraction_above(ionization, threshold=0.24) * float(bubble_progress)
        ionized_mass = _mass_fraction_above(ionization, threshold=0.24) * float(bubble_progress)
        ionized_fraction = float(np.clip(0.58 * ionized_area + 0.42 * ionized_mass, 0.0, 1.0))
        neutral_fraction = float(np.clip(1.0 - ionized_fraction, 0.0, 1.0))
        overlap = _area_fraction_above(ionization, threshold=0.48) * float(overlap_progress)
        photoheating = float(
            np.clip(np.mean(ionization * np.clip(1.0 - cold_gas, 0.0, 1.0)) * 2.6, 0.0, 1.0)
        )

        context.state.ionized_fraction = ionized_fraction
        context.state.neutral_fraction = neutral_fraction
        context.state.ionization_fraction = ionized_fraction
        context.state.opacity = neutral_fraction**1.25
        context.state.ionized_bubble_fraction = _area_fraction_above(ionization, threshold=0.24)
        context.state.bubble_overlap_fraction = float(np.clip(overlap, 0.0, 1.0))
        context.state.photoheating_feedback = photoheating
        context.state.stellar_radiation_intensity = float(np.clip(np.mean(radiation) * 4.2, 0.0, 1.0))
        context.state.reionization_progress = float(
            np.clip(0.72 * ionized_fraction + 0.28 * context.state.bubble_overlap_fraction, 0.0, 1.0)
        )
        context.state.ionization_fraction_history.append(context.state.ionization_fraction)
        context.state.opacity_history.append(context.state.opacity)
        context.state.ionized_fraction_history.append(context.state.ionized_fraction)
        context.state.neutral_fraction_history.append(context.state.neutral_fraction)
        context.state.bubble_overlap_fraction_history.append(context.state.bubble_overlap_fraction)
        context.state.photoheating_feedback_history.append(context.state.photoheating_feedback)
        context.state.stellar_radiation_history.append(context.state.stellar_radiation_intensity)
        context.state.ionized_bubble_fraction_history.append(context.state.ionized_bubble_fraction)
