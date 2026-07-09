"""Gas-collapse stage: baryonic gas cools and falls into dark halos."""

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
    _mass_fraction_above,
    _normalize01,
    _soft_threshold,
    _smoothstep,
)


class GasCollapseStage:
    """Cool baryonic gas and mark pre-stellar collapse zones.

    Dark Ages created the hidden gravitational scaffold. This stage keeps the story
    pre-stellar: gas becomes colder and denser in the deepest halos, but no stars are
    lit yet. The output is a set of collapse candidates for a later first-stars stage.
    """

    stage_id = "gas_collapse"
    title = "Сжатие газа"

    def __init__(self) -> None:
        self._elapsed_s = 0.0
        self._initial_a = 2.2e-2
        self._initial_t_gyr = 0.18
        self._initial_temperature_k = 60.0
        self._initial_baryon: np.ndarray | None = None
        self._initial_cold_gas: np.ndarray | None = None
        self._target_baryon: np.ndarray | None = None
        self._cooling_seed: np.ndarray | None = None
        self._collapse_seed: np.ndarray | None = None
        self._halo_source: np.ndarray | None = None
        self._dark_source: np.ndarray | None = None

    def enter(self, context: UniverseRunContext) -> None:
        """Prepare cooling and collapse maps from halo and gas fields."""

        context.state.current_stage = self.stage_id
        context.state.era = "gas_collapse"
        context.state.stage_progress = 0.0
        self._elapsed_s = 0.0
        self._initial_a = max(context.state.a, 2.2e-2)
        self._initial_t_gyr = max(context.state.t_gyr, 0.18)
        self._initial_temperature_k = max(context.state.temperature_k, 30.0)

        dark = context.fields.dark_density
        if not _has_signal(dark):
            dark = context.fields.gravitational_potential
        if not _has_signal(dark):
            dark = context.fields.cmb
        if not _has_signal(dark):
            dark = context.fields.seed_delta
        self._dark_source = context.backend.normalize_field(dark)

        halo = context.fields.halo_density
        if not _has_signal(halo):
            halo = context.fields.future_star_sites
        if not _has_signal(halo):
            halo = self._dark_source
        halo01 = _normalize01(halo)
        self._halo_source = halo01.astype(np.float32)

        baryon = context.fields.baryon_density
        if not _has_signal(baryon):
            baryon = context.backend.diffuse(self._dark_source, amount=0.88)
        self._initial_baryon = context.backend.normalize_field(baryon)
        self._initial_cold_gas = (
            _normalize01(context.fields.cold_gas_density)
            if _has_signal(context.fields.cold_gas_density)
            else np.zeros_like(self._halo_source, dtype=np.float32)
        )

        structure = context.config.structure
        cooling_efficiency = float(np.clip(structure.cooling_efficiency, 0.0, 1.5))
        infall = float(np.clip(structure.baryon_infall, 0.0, 1.5))
        pressure = float(np.clip(structure.gas_pressure, 0.0, 1.5))

        baryon01 = _normalize01(self._initial_baryon)
        deep_halos = _soft_threshold(self._halo_source, 0.54, 0.28)
        gas_in_halos = _soft_threshold(baryon01, 0.50, 0.32)
        cooling_seed = np.clip(
            deep_halos * (0.62 + 0.28 * cooling_efficiency)
            + gas_in_halos * deep_halos * (0.40 + 0.20 * cooling_efficiency),
            0.0,
            1.0,
        )
        cooling_seed = context.backend.diffuse(cooling_seed, amount=min(0.68, 0.32 + 0.18 * pressure))
        self._cooling_seed = _normalize01(cooling_seed)

        collapse_seed = np.clip(
            self._cooling_seed
            * _soft_threshold(self._halo_source, 0.66, 0.22)
            * _soft_threshold(baryon01, 0.48, 0.34)
            * (0.72 + 0.26 * cooling_efficiency),
            0.0,
            1.0,
        )
        self._collapse_seed = _normalize01(collapse_seed)

        gas_target = (
            0.50 * self._initial_baryon
            + 0.38 * context.backend.normalize_field(self._dark_source)
            + (0.38 + 0.24 * infall) * context.backend.normalize_field(self._cooling_seed)
            + (0.32 + 0.20 * infall) * context.backend.normalize_field(self._collapse_seed)
        )
        gas_target = context.backend.diffuse(gas_target, amount=min(0.62, 0.26 + 0.18 * pressure))
        self._target_baryon = context.backend.normalize_field(gas_target)

        context.fields.cold_gas_density = np.zeros_like(self._halo_source, dtype=np.float32)
        context.fields.molecular_cooling = np.zeros_like(self._halo_source, dtype=np.float32)
        context.fields.collapse_sites = np.zeros_like(self._halo_source, dtype=np.float32)
        context.fields.stars = np.zeros_like(self._halo_source, dtype=np.float32)
        self._update_gas_metrics(context, cooling_progress=0.0, collapse_progress=0.0)
        ExpansionEngine.update_state(context.state, context.config.cosmology)

    def step(self, context: UniverseRunContext, dt: float) -> None:
        """Cool gas, increase density in halos, and grow collapse candidates."""

        if (
            self._initial_baryon is None
            or self._initial_cold_gas is None
            or self._target_baryon is None
            or self._cooling_seed is None
            or self._collapse_seed is None
            or self._halo_source is None
            or self._dark_source is None
        ):
            raise RuntimeError("gas collapse stage entered without source fields")

        duration = max(stage_screen_duration_s(context.config, self.stage_id, 52.0), 1.0e-6)
        self._elapsed_s = min(duration, self._elapsed_s + max(dt, 0.0))
        progress = self._elapsed_s / duration
        infall_progress = _smoothstep(progress)
        cooling_progress = _delayed_progress(progress, start=0.10, end=0.82)
        collapse_progress = _delayed_progress(progress, start=0.46, end=1.0)

        context.fields.baryon_density = _blend(
            self._initial_baryon,
            self._target_baryon,
            infall_progress,
        )
        gas01 = _normalize01(context.fields.baryon_density)
        cold_gas = np.clip(
            _blend(self._initial_cold_gas, gas01 * self._cooling_seed, cooling_progress)
            + self._collapse_seed * (0.18 * collapse_progress),
            0.0,
            1.0,
        )
        context.fields.cold_gas_density = cold_gas.astype(np.float32)
        context.fields.molecular_cooling = np.clip(
            self._cooling_seed * (0.18 + 0.82 * cooling_progress),
            0.0,
            1.0,
        ).astype(np.float32)
        context.fields.collapse_sites = np.clip(
            self._collapse_seed * collapse_progress,
            0.0,
            1.0,
        ).astype(np.float32)
        # Future-star sites are refined, not lit. The stars layer stays dark until
        # the next epoch makes actual luminous objects.
        context.fields.future_star_sites = np.clip(
            0.42 * context.fields.future_star_sites
            + 0.58 * context.fields.collapse_sites,
            0.0,
            1.0,
        ).astype(np.float32)
        context.fields.stars = np.zeros_like(context.fields.collapse_sites, dtype=np.float32)

        context.state.a = _log_lerp(self._initial_a, 3.0e-2, infall_progress)
        context.state.t_gyr = _log_lerp(self._initial_t_gyr, 0.26, infall_progress)
        context.state.temperature_k = _log_lerp(self._initial_temperature_k, 35.0, infall_progress)
        context.state.gas_temperature_k = _log_lerp(900.0, 180.0, cooling_progress)
        context.state.stage_progress = progress
        self._update_gas_metrics(
            context,
            cooling_progress=cooling_progress,
            collapse_progress=collapse_progress,
        )
        ExpansionEngine.update_state(context.state, context.config.cosmology)

    def is_complete(self, context: UniverseRunContext) -> bool:
        """Return true when pre-stellar collapse candidates are ready."""

        return context.state.stage_progress >= 1.0

    def build_report(self, context: UniverseRunContext) -> StageReport:
        """Build the gas-collapse checkpoint report."""

        coupling = _correlation(context.fields.baryon_density, context.fields.collapse_sites)
        scale = sample_scale(context.state, context.config)
        return StageReport(
            stage_id=self.stage_id,
            title="Сжатие газа: предзвёздные облака готовы",
            summary_lines=(
                "Тёмный каркас уже был собран; теперь обычный газ начал физически догонять эти ямы.",
                "Охлаждение снижает давление, поэтому плотные облака становятся устойчивее.",
                "Звёзд всё ещё нет: это подготовка мест, где они смогут вспыхнуть в следующую эпоху.",
                f"Физическое время: {context.state.t_gyr:.3f} Gyr после Big Bang",
                f"Масштаб a(t): {context.state.a:.3e}",
                f"Температура фона: {context.state.temperature_k:.1f} K",
                f"Температура холодного газа: {context.state.gas_temperature_k:.1f} K",
                f"Видимый участок сейчас: {scale.box_now_text}",
                f"Доля охлаждённого газа: {context.state.gas_cooling_fraction:.1%}",
                f"Доля газа в коллапсе: {context.state.collapsed_gas_fraction:.1%}",
                f"Кандидатов облаков коллапса: {context.state.collapse_site_count}",
                f"Готовность к первым звёздам: {context.state.star_formation_readiness:.2f}",
                f"Связь газ ↔ зоны коллапса: {coupling:.2f}",
                "Следующий логичный этап: первые звёзды вспыхнут только в этих подготовленных узлах.",
            ),
            metrics={
                "gas_cooling_fraction": context.state.gas_cooling_fraction,
                "collapsed_gas_fraction": context.state.collapsed_gas_fraction,
                "collapse_site_count": float(context.state.collapse_site_count),
                "star_formation_readiness": context.state.star_formation_readiness,
                "gas_temperature_k": context.state.gas_temperature_k,
                "gas_collapse_correlation": coupling,
                "temperature_k": context.state.temperature_k,
                "a": context.state.a,
                "t_gyr": context.state.t_gyr,
            },
        )

    def _update_gas_metrics(
        self,
        context: UniverseRunContext,
        *,
        cooling_progress: float,
        collapse_progress: float,
    ) -> None:
        cold = _normalize01(context.fields.cold_gas_density) if _has_signal(context.fields.cold_gas_density) else np.zeros_like(context.fields.baryon_density)
        cooling = _normalize01(context.fields.molecular_cooling) if _has_signal(context.fields.molecular_cooling) else np.zeros_like(cold)
        collapse = _normalize01(context.fields.collapse_sites) if _has_signal(context.fields.collapse_sites) else np.zeros_like(cold)

        context.state.baryon_contrast = float(np.std(context.fields.baryon_density))
        context.state.cold_gas_fraction = _mass_fraction_above(cold, threshold=0.55)
        context.state.gas_cooling_fraction = _mass_fraction_above(cooling, threshold=0.50) * float(cooling_progress)
        context.state.collapse_site_count = _count_local_peaks(collapse, threshold=0.70)
        context.state.collapsed_gas_fraction = _mass_fraction_above(collapse, threshold=0.62) * float(collapse_progress)
        readiness = min(1.0, context.state.collapsed_gas_fraction * 5.0 + context.state.collapse_site_count / 120.0)
        context.state.star_formation_readiness = float(readiness)
        context.state.future_star_site_count = max(context.state.future_star_site_count, context.state.collapse_site_count)
        context.state.gas_cooling_fraction_history.append(context.state.gas_cooling_fraction)
        context.state.collapse_site_count_history.append(context.state.collapse_site_count)
        context.state.star_formation_readiness_history.append(context.state.star_formation_readiness)
