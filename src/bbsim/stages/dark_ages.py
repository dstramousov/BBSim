"""Dark ages stage: grow the hidden matter scaffold before first stars."""

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


def _normalize01(field: np.ndarray) -> np.ndarray:
    data = np.asarray(field, dtype=np.float32)
    min_value = float(np.nanmin(data))
    max_value = float(np.nanmax(data))
    span = max(max_value - min_value, 1.0e-8)
    return ((data - min_value) / span).astype(np.float32)


def _soft_threshold(field01: np.ndarray, threshold: float, softness: float) -> np.ndarray:
    soft = max(float(softness), 1.0e-6)
    return np.clip((field01 - float(threshold)) / soft, 0.0, 1.0).astype(np.float32)


def _contrast_boost(field: np.ndarray, amount: float) -> np.ndarray:
    data = np.asarray(field, dtype=np.float32)
    mean = float(data.mean())
    boosted = mean + (data - mean) * (1.0 + max(float(amount), 0.0))
    return boosted.astype(np.float32)


def _ridge_field(field: np.ndarray) -> np.ndarray:
    base = np.asarray(field, dtype=np.float32)
    wide = (
        np.roll(base, 3, axis=0)
        + np.roll(base, -3, axis=0)
        + np.roll(base, 3, axis=1)
        + np.roll(base, -3, axis=1)
        + np.roll(base, (5, -5), axis=(0, 1))
        + np.roll(base, (-5, 5), axis=(0, 1))
    ) / 6.0
    ridge = np.clip(base - 0.58 * wide, 0.0, None)
    return _normalize01(ridge)


def _count_local_peaks(field01: np.ndarray, threshold: float) -> int:
    data = np.asarray(field01, dtype=np.float32)
    mask = data >= float(threshold)
    if not bool(mask.any()):
        return 0
    # Count only compact local maxima, not every pixel in one bright patch.
    local_max = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            local_max &= data >= np.roll(np.roll(data, dy, axis=0), dx, axis=1)
    return int(local_max.sum())


def _mass_fraction_above(field01: np.ndarray, threshold: float) -> float:
    data = np.clip(np.asarray(field01, dtype=np.float32), 0.0, 1.0)
    positive_sum = float(data.sum())
    if positive_sum <= 1.0e-8:
        return 0.0
    return float(data[data >= threshold].sum() / positive_sum)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = np.asarray(a, dtype=np.float32).reshape(-1)
    b_flat = np.asarray(b, dtype=np.float32).reshape(-1)
    a_std = float(a_flat.std())
    b_std = float(b_flat.std())
    if a_std <= 1.0e-8 or b_std <= 1.0e-8:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


class DarkAgesStage:
    """Grow dark-matter structure and delayed baryonic gas before stars.

    This stage still deliberately avoids real stars. It gives the player the missing
    causal step: CMB leaves an imprint, dark matter amplifies it into halos and
    filaments, then ordinary gas starts falling into that scaffold with visible lag.
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
        self._potential: np.ndarray | None = None
        self._halo_seed: np.ndarray | None = None
        self._filament_seed: np.ndarray | None = None
        self._cmb_source: np.ndarray | None = None

    def enter(self, context: UniverseRunContext) -> None:
        """Prepare dark matter, halo, and baryon fields from the CMB source."""

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

        potential = context.backend.diffuse(self._cmb_source, amount=0.88)
        potential = context.backend.diffuse(potential, amount=0.72)
        potential = context.backend.normalize_field(potential)
        potential01 = _normalize01(potential)
        self._potential = potential.astype(np.float32)

        ridge = _ridge_field(potential01)
        halo_threshold = float(np.percentile(potential01, 88.0))
        halo_seed = _soft_threshold(potential01, halo_threshold, softness=0.16)
        halo_seed = np.clip(0.70 * halo_seed + 0.30 * ridge * halo_seed, 0.0, 1.0)
        self._halo_seed = halo_seed.astype(np.float32)
        self._filament_seed = ridge.astype(np.float32)

        gravity = max(context.config.structure.gravity_strength, 0.0)
        dark_scaffold = (
            0.72 * potential
            + (0.40 + 0.30 * gravity) * context.backend.normalize_field(ridge)
            + (0.55 + 0.30 * gravity) * context.backend.normalize_field(halo_seed)
        )
        dark_scaffold = context.backend.normalize_field(dark_scaffold)
        self._target_dark = (dark_scaffold * (0.85 + 0.28 * gravity)).astype(np.float32)

        gas_pressure = max(context.config.structure.gas_pressure, 0.0)
        infall = max(context.config.structure.baryon_infall, 0.0)
        gas = context.backend.diffuse(dark_scaffold, amount=min(0.96, 0.78 + 0.12 * gas_pressure))
        gas = context.backend.diffuse(gas, amount=min(0.96, 0.44 + 0.20 * gas_pressure))
        gas = context.backend.normalize_field(gas)
        self._target_baryon = (gas * (0.48 + 0.34 * infall)).astype(np.float32)

        context.fields.dark_density = self._initial_dark.astype(np.float32, copy=True)
        context.fields.baryon_density = self._initial_baryon.astype(np.float32, copy=True)
        context.fields.gravitational_potential = self._potential.astype(np.float32, copy=True)
        context.fields.halo_density = np.zeros_like(self._halo_seed, dtype=np.float32)
        context.fields.future_star_sites = np.zeros_like(self._halo_seed, dtype=np.float32)
        self._update_structure_metrics(context, dark_progress=0.0, baryon_progress=0.0)
        ExpansionEngine.update_state(context.state, context.config.cosmology)

    def step(self, context: UniverseRunContext, dt: float) -> None:
        """Amplify dark structure and let gas fall in with delay."""

        if (
            self._initial_dark is None
            or self._initial_baryon is None
            or self._target_dark is None
            or self._target_baryon is None
            or self._potential is None
            or self._halo_seed is None
            or self._filament_seed is None
            or self._cmb_source is None
        ):
            raise RuntimeError("dark ages stage entered without source fields")

        duration = max(stage_screen_duration_s(context.config, self.stage_id, 45.0), 1.0e-6)
        self._elapsed_s = min(duration, self._elapsed_s + max(dt, 0.0))
        progress = self._elapsed_s / duration
        dark_progress = _smoothstep(progress)
        halo_progress = _delayed_progress(progress, start=0.12, end=0.82)
        baryon_progress = _delayed_progress(progress, start=0.34, end=1.0)
        candidate_progress = _delayed_progress(progress, start=0.58, end=1.0)

        growing_dark = _contrast_boost(self._target_dark, 0.60 * halo_progress)
        growing_dark = growing_dark + context.backend.normalize_field(self._halo_seed) * (0.26 * halo_progress)
        context.fields.dark_density = _blend(self._initial_dark, growing_dark, dark_progress)

        baryon_target = _blend(self._target_baryon, context.backend.diffuse(growing_dark, amount=0.68), 0.22)
        context.fields.baryon_density = _blend(
            self._initial_baryon,
            baryon_target,
            baryon_progress,
        )

        # CMB light fades away; hidden matter fields become the story.
        context.fields.radiation = ((1.0 - dark_progress) * self._cmb_source).astype(np.float32)
        context.fields.gravitational_potential = (
            self._potential * (0.55 + 0.45 * dark_progress)
        ).astype(np.float32)
        context.fields.halo_density = np.clip(
            self._halo_seed * (0.20 + 0.80 * halo_progress)
            + self._filament_seed * (0.18 * halo_progress),
            0.0,
            1.0,
        ).astype(np.float32)

        gas01 = _normalize01(context.fields.baryon_density)
        future_sites = np.clip(
            context.fields.halo_density * _soft_threshold(gas01, 0.56, 0.26) * candidate_progress,
            0.0,
            1.0,
        )
        context.fields.future_star_sites = future_sites.astype(np.float32)

        context.state.a = _log_lerp(self._initial_a, 2.2e-2, dark_progress)
        context.state.t_gyr = _log_lerp(self._initial_t_gyr, 0.18, dark_progress)
        context.state.temperature_k = _log_lerp(self._initial_temperature_k, 60.0, dark_progress)
        context.state.stage_progress = progress
        context.state.ionization_fraction = 0.0
        context.state.opacity = 0.0
        self._update_structure_metrics(
            context,
            dark_progress=dark_progress,
            baryon_progress=baryon_progress,
        )
        ExpansionEngine.update_state(context.state, context.config.cosmology)

    def is_complete(self, context: UniverseRunContext) -> bool:
        """Return true after the hidden scaffold and delayed gas are visible."""

        return context.state.stage_progress >= 1.0

    def build_report(self, context: UniverseRunContext) -> StageReport:
        """Build the dark-ages checkpoint report."""

        coupling = _correlation(context.fields.dark_density, context.fields.baryon_density)
        scale = sample_scale(context.state, context.config)
        return StageReport(
            stage_id=self.stage_id,
            title="Тёмные века: скрытый каркас вырос",
            summary_lines=(
                "CMB уже отпечатался, но звёзд ещё нет: Вселенная тёмная.",
                "Тёмная материя усилила гравитационный каркас: видны будущие узлы и нити.",
                "Обычный газ запаздывал, но к концу эпохи начал стекать в те же области.",
                f"Физическое время: {context.state.t_gyr:.3f} Gyr после Big Bang",
                f"Масштаб a(t): {context.state.a:.3e}",
                f"Температура фона: {context.state.temperature_k:.1f} K",
                f"Видимый участок сейчас: {scale.box_now_text}",
                f"Контраст тёмного каркаса: {context.state.dark_matter_contrast:.2f}",
                f"Контраст газа: {context.state.baryon_contrast:.2f}",
                f"Задержка газа: {context.state.gas_lag:.2f}",
                f"Кандидатов будущих гало: {context.state.halo_count}",
                f"Масса в плотных узлах: {context.state.halo_mass_fraction:.1%}",
                f"Будущих звёздных узлов: {context.state.future_star_site_count}",
                f"Связь газ ↔ тёмный каркас: {coupling:.2f}",
                "Следующий логичный этап: газ начнёт остывать и в первых плотных узлах зажгутся звёзды.",
            ),
            metrics={
                "dark_contrast": context.state.dark_matter_contrast,
                "baryon_contrast": context.state.baryon_contrast,
                "gas_lag": context.state.gas_lag,
                "halo_count": float(context.state.halo_count),
                "halo_mass_fraction": context.state.halo_mass_fraction,
                "future_star_site_count": float(context.state.future_star_site_count),
                "dark_baryon_correlation": coupling,
                "temperature_k": context.state.temperature_k,
                "a": context.state.a,
                "t_gyr": context.state.t_gyr,
            },
        )

    def _update_structure_metrics(
        self,
        context: UniverseRunContext,
        *,
        dark_progress: float,
        baryon_progress: float,
    ) -> None:
        dark = np.asarray(context.fields.dark_density, dtype=np.float32)
        baryon = np.asarray(context.fields.baryon_density, dtype=np.float32)
        halo01 = _normalize01(context.fields.halo_density) if _has_signal(context.fields.halo_density) else np.zeros_like(dark)
        future01 = (
            _normalize01(context.fields.future_star_sites)
            if _has_signal(context.fields.future_star_sites)
            else np.zeros_like(dark)
        )

        context.state.dark_matter_contrast = float(np.std(dark))
        context.state.baryon_contrast = float(np.std(baryon))
        context.state.gas_lag = float(max(dark_progress - baryon_progress, 0.0))
        context.state.halo_count = _count_local_peaks(halo01, threshold=0.78)
        context.state.halo_mass_fraction = _mass_fraction_above(halo01, threshold=0.74)
        context.state.future_star_site_count = _count_local_peaks(future01, threshold=0.72)
        context.state.dark_matter_contrast_history.append(context.state.dark_matter_contrast)
        context.state.baryon_contrast_history.append(context.state.baryon_contrast)
        context.state.halo_count_history.append(context.state.halo_count)
        context.state.gas_lag_history.append(context.state.gas_lag)
