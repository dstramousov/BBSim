"""First-stars stage: rare lights ignite inside prepared gas-collapse sites."""

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
    _count_local_peaks,
    _delayed_progress,
    _has_signal,
    _log_lerp,
    _normalize01,
    _soft_threshold,
    _smoothstep,
)


def _rare_core_mask(field01: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Return compact bright star-forming cores from a normalized field."""

    data = np.clip(np.asarray(field01, dtype=np.float32), 0.0, 1.0)
    positive = data[data > 1.0e-6]
    if not bool(positive.size):
        return np.zeros_like(data, dtype=np.float32)
    threshold = float(np.percentile(positive, percentile))
    # Some upstream masks are already saturated at 1.0 in compact peaks. Keep
    # those peaks instead of producing an all-zero soft threshold at max(data).
    threshold = min(threshold, float(positive.max()) - 0.08)
    return _soft_threshold(data, threshold, softness=0.08)


def _area_fraction_above(field01: np.ndarray, threshold: float) -> float:
    data = np.clip(np.asarray(field01, dtype=np.float32), 0.0, 1.0)
    if data.size == 0:
        return 0.0
    return float(np.mean(data >= float(threshold)))


class FirstStarsStage:
    """Ignite the first generation of stars only where gas collapse prepared them.

    This is intentionally not a random starfield. Dark matter halos, cooled gas,
    molecular cooling, and collapse sites from the previous epochs decide where the
    first lights appear. The stars are rare, bright, and begin to create small
    ionized bubbles around themselves.
    """

    stage_id = "first_stars"
    title = "Первые звёзды"

    def __init__(self) -> None:
        self._elapsed_s = 0.0
        self._initial_a = 3.0e-2
        self._initial_t_gyr = 0.26
        self._initial_temperature_k = 35.0
        self._initial_cold_gas: np.ndarray | None = None
        self._initial_collapse_sites: np.ndarray | None = None
        self._initial_ionization: np.ndarray | None = None
        self._ignition_seed: np.ndarray | None = None
        self._star_seed: np.ndarray | None = None
        self._bubble_seed: np.ndarray | None = None
        self._radiation_seed: np.ndarray | None = None

    def enter(self, context: UniverseRunContext) -> None:
        """Prepare first-light seeds from collapse, cooling, gas, and halo fields."""

        context.state.current_stage = self.stage_id
        context.state.era = "first_stars"
        context.state.stage_progress = 0.0
        self._elapsed_s = 0.0
        self._initial_a = max(context.state.a, 3.0e-2)
        self._initial_t_gyr = max(context.state.t_gyr, 0.26)
        self._initial_temperature_k = max(context.state.temperature_k, 20.0)

        collapse = context.fields.collapse_sites
        if not _has_signal(collapse):
            collapse = context.fields.future_star_sites
        if not _has_signal(collapse):
            collapse = context.fields.halo_density
        collapse01 = _normalize01(collapse)
        self._initial_collapse_sites = collapse01.astype(np.float32)

        cold_gas = context.fields.cold_gas_density
        if not _has_signal(cold_gas):
            cold_gas = context.fields.baryon_density
        cold01 = _normalize01(cold_gas)
        self._initial_cold_gas = cold01.astype(np.float32)

        cooling = context.fields.molecular_cooling
        if not _has_signal(cooling):
            cooling = cold01
        cooling01 = _normalize01(cooling)

        halo = context.fields.halo_density
        if not _has_signal(halo):
            halo = context.fields.future_star_sites
        halo01 = _normalize01(halo)

        efficiency = float(np.clip(context.config.structure.star_formation_efficiency, 0.0, 1.8))
        prepared = np.clip(
            (0.42 * collapse01 + 0.28 * cold01 + 0.20 * cooling01 + 0.10 * halo01)
            * (0.72 + 0.34 * efficiency),
            0.0,
            1.0,
        )
        ignition = np.clip(
            prepared
            * _soft_threshold(collapse01, 0.52, 0.24)
            * _soft_threshold(cold01, 0.44, 0.30)
            * (0.70 + 0.24 * efficiency),
            0.0,
            1.0,
        )
        ignition = _normalize01(ignition)
        rare_cores = _rare_core_mask(ignition, percentile=99.0)
        self._ignition_seed = ignition.astype(np.float32)
        self._star_seed = np.clip(ignition * rare_cores, 0.0, 1.0).astype(np.float32)
        # Bubbles and radiation are smoother than the star cores. They grow outward
        # from the rare ignition nodes instead of filling the whole map instantly.
        bubble = context.backend.diffuse(self._star_seed, amount=0.62)
        bubble = context.backend.diffuse(bubble, amount=0.54)
        self._bubble_seed = _normalize01(bubble)
        radiation = context.backend.diffuse(self._star_seed, amount=0.42)
        radiation = context.backend.diffuse(radiation, amount=0.34)
        self._radiation_seed = _normalize01(radiation)

        self._initial_ionization = (
            _normalize01(context.fields.ionization)
            if _has_signal(context.fields.ionization)
            else np.zeros_like(self._star_seed, dtype=np.float32)
        )

        context.fields.stellar_ignition = np.zeros_like(self._star_seed, dtype=np.float32)
        context.fields.first_star_density = np.zeros_like(self._star_seed, dtype=np.float32)
        context.fields.stars = np.zeros_like(self._star_seed, dtype=np.float32)
        context.fields.stellar_radiation = np.zeros_like(self._star_seed, dtype=np.float32)
        context.fields.ionized_bubbles = np.zeros_like(self._star_seed, dtype=np.float32)
        context.fields.radiation = np.zeros_like(self._star_seed, dtype=np.float32)
        self._update_first_star_metrics(context, ignition_progress=0.0, bubble_progress=0.0)
        ExpansionEngine.update_state(context.state, context.config.cosmology)

    def step(self, context: UniverseRunContext, dt: float) -> None:
        """Ignite first stars and expand their early ionized bubbles."""

        if (
            self._initial_cold_gas is None
            or self._initial_collapse_sites is None
            or self._initial_ionization is None
            or self._ignition_seed is None
            or self._star_seed is None
            or self._bubble_seed is None
            or self._radiation_seed is None
        ):
            raise RuntimeError("first-stars stage entered without collapse fields")

        duration = max(stage_screen_duration_s(context.config, self.stage_id, 58.0), 1.0e-6)
        self._elapsed_s = min(duration, self._elapsed_s + max(dt, 0.0))
        progress = self._elapsed_s / duration
        ignition_progress = _delayed_progress(progress, start=0.16, end=0.64)
        radiation_progress = _delayed_progress(progress, start=0.30, end=0.84)
        bubble_progress = _delayed_progress(progress, start=0.42, end=1.0)

        flicker = 0.92 + 0.08 * np.sin(progress * np.pi * 18.0)
        ignition = np.clip(self._ignition_seed * ignition_progress, 0.0, 1.0)
        stars = np.clip(self._star_seed * ignition_progress * flicker, 0.0, 1.0)
        radiation = np.clip(self._radiation_seed * radiation_progress, 0.0, 1.0)
        bubbles = np.clip(
            _blend(self._initial_ionization, self._bubble_seed, bubble_progress)
            + stars * (0.18 * bubble_progress),
            0.0,
            1.0,
        )

        # The cold gas is locally consumed/heated around first stars, but it does
        # not disappear globally yet. Reionization gets a later dedicated epoch.
        gas_consumption = np.clip(stars * (0.16 + 0.14 * bubble_progress), 0.0, 0.42)
        context.fields.cold_gas_density = np.clip(
            self._initial_cold_gas * (1.0 - gas_consumption)
            + self._initial_collapse_sites * (0.10 * (1.0 - ignition_progress)),
            0.0,
            1.0,
        ).astype(np.float32)
        context.fields.stellar_ignition = ignition.astype(np.float32)
        context.fields.first_star_density = stars.astype(np.float32)
        context.fields.stars = stars.astype(np.float32)
        context.fields.stellar_radiation = radiation.astype(np.float32)
        context.fields.radiation = radiation.astype(np.float32)
        context.fields.ionized_bubbles = bubbles.astype(np.float32)
        context.fields.ionization = bubbles.astype(np.float32)

        context.state.a = _log_lerp(self._initial_a, 4.6e-2, _smoothstep(progress))
        context.state.t_gyr = _log_lerp(self._initial_t_gyr, 0.45, _smoothstep(progress))
        context.state.temperature_k = _log_lerp(self._initial_temperature_k, 18.0, _smoothstep(progress))
        context.state.gas_temperature_k = _log_lerp(180.0, 900.0, min(1.0, radiation_progress + 0.2 * bubble_progress))
        context.state.heavy_elements_fraction = min(
            1.0e-5,
            max(context.state.heavy_elements_fraction, 1.0e-8 * ignition_progress * context.config.structure.metal_yield),
        )
        context.state.stage_progress = progress
        self._update_first_star_metrics(
            context,
            ignition_progress=ignition_progress,
            bubble_progress=bubble_progress,
        )
        ExpansionEngine.update_state(context.state, context.config.cosmology)

    def is_complete(self, context: UniverseRunContext) -> bool:
        """Return true when first lights and early bubbles are established."""

        return context.state.stage_progress >= 1.0

    def build_report(self, context: UniverseRunContext) -> StageReport:
        """Build the first-stars checkpoint report."""

        coupling = _correlation(context.fields.first_star_density, context.fields.collapse_sites)
        scale = sample_scale(context.state, context.config)
        return StageReport(
            stage_id=self.stage_id,
            title="Первые звёзды: первый свет появился",
            summary_lines=(
                "Первые звёзды вспыхнули не случайно, а только в подготовленных плотных облаках.",
                "Это редкие мощные очаги: вокруг них растёт излучение и первые ионизированные пузыри.",
                "Реионизация ещё не завершена — это только её начало.",
                f"Физическое время: {context.state.t_gyr:.3f} Gyr после Big Bang",
                f"Масштаб a(t): {context.state.a:.3e}",
                f"Температура фона: {context.state.temperature_k:.1f} K",
                f"Температура газа возле первых огней: {context.state.gas_temperature_k:.1f} K",
                f"Видимый участок сейчас: {scale.box_now_text}",
                f"Очагов первых звёзд: {context.state.first_star_count}",
                f"Доля газа в звездообразовании: {context.state.star_formation_fraction:.1%}",
                f"Интенсивность излучения: {context.state.stellar_radiation_intensity:.2f}",
                f"Доля ионизированных пузырей: {context.state.ionized_bubble_fraction:.1%}",
                f"Связь первые звёзды ↔ коллапс газа: {coupling:.2f}",
                "Следующий логичный этап: реионизация, когда пузыри света начнут соединяться.",
            ),
            metrics={
                "first_star_count": float(context.state.first_star_count),
                "first_star_mass_fraction": context.state.first_star_mass_fraction,
                "star_formation_fraction": context.state.star_formation_fraction,
                "stellar_radiation_intensity": context.state.stellar_radiation_intensity,
                "ionized_bubble_fraction": context.state.ionized_bubble_fraction,
                "reionization_progress": context.state.reionization_progress,
                "first_star_collapse_correlation": coupling,
                "gas_temperature_k": context.state.gas_temperature_k,
                "temperature_k": context.state.temperature_k,
                "a": context.state.a,
                "t_gyr": context.state.t_gyr,
            },
        )

    def _update_first_star_metrics(
        self,
        context: UniverseRunContext,
        *,
        ignition_progress: float,
        bubble_progress: float,
    ) -> None:
        stars = _normalize01(context.fields.first_star_density) if _has_signal(context.fields.first_star_density) else np.zeros_like(context.fields.collapse_sites)
        radiation = _normalize01(context.fields.stellar_radiation) if _has_signal(context.fields.stellar_radiation) else np.zeros_like(stars)
        bubbles = _normalize01(context.fields.ionized_bubbles) if _has_signal(context.fields.ionized_bubbles) else np.zeros_like(stars)
        ignition = _normalize01(context.fields.stellar_ignition) if _has_signal(context.fields.stellar_ignition) else np.zeros_like(stars)

        context.state.first_star_count = _count_local_peaks(stars, threshold=0.70)
        context.state.first_star_mass_fraction = _area_fraction_above(stars, threshold=0.58) * float(ignition_progress)
        context.state.star_formation_fraction = _area_fraction_above(ignition, threshold=0.50) * float(ignition_progress)
        context.state.stellar_radiation_intensity = float(np.clip(np.mean(radiation) * 5.0, 0.0, 1.0))
        context.state.ionized_bubble_fraction = _area_fraction_above(bubbles, threshold=0.32) * float(bubble_progress)
        context.state.reionization_progress = min(1.0, context.state.ionized_bubble_fraction * 1.7)
        context.state.first_star_count_history.append(context.state.first_star_count)
        context.state.stellar_radiation_history.append(context.state.stellar_radiation_intensity)
        context.state.ionized_bubble_fraction_history.append(context.state.ionized_bubble_fraction)
