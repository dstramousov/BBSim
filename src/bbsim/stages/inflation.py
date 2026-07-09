"""Prototype inflation stage."""

from __future__ import annotations

import math

import numpy as np

from bbsim.core.context import UniverseRunContext
from bbsim.core.expansion import ExpansionEngine
from bbsim.core.report import StageReport


class InflationStage:
    """Apply prototype exponential expansion and curvature smoothing."""

    stage_id = "inflation"
    title = "Инфляция"

    def __init__(self) -> None:
        self._elapsed_s = 0.0
        self._initial_a = 1.0e-32
        self._initial_curvature = 0.0
        self._target_delta = None

    def enter(self, context: UniverseRunContext) -> None:
        """Capture initial state for the inflation stage."""

        context.state.current_stage = self.stage_id
        context.state.era = "inflation"
        context.state.stage_progress = 0.0
        self._elapsed_s = 0.0
        self._initial_a = context.state.a
        self._initial_curvature = context.state.curvature
        self._target_delta = None
        ExpansionEngine.update_state(context.state, context.config.cosmology, update_era=False)

    def step(self, context: UniverseRunContext, dt: float) -> None:
        """Advance the prototype inflation visual stage."""

        duration = max(context.config.inflation.visual_duration_s, 1.0e-6)
        self._elapsed_s = min(duration, self._elapsed_s + max(dt, 0.0))
        progress = self._elapsed_s / duration
        params = context.config.inflation
        n_e_folds = self._compute_e_folds(context)

        context.state.a = self._initial_a * math.exp(n_e_folds * progress)
        context.state.curvature = self._initial_curvature * math.exp(
            -params.smoothing * 10.0 * progress
        )
        context.state.stage_progress = progress
        ExpansionEngine.update_state(context.state, context.config.cosmology, update_era=False)

        context.fields.inflation_delta = self._build_live_field(context, progress)
        context.fields.dark_density = 1.0 + context.fields.inflation_delta
        context.fields.baryon_density = 1.0 + context.backend.diffuse(
            context.fields.inflation_delta,
            amount=0.75,
        )

    def _build_live_field(self, context: UniverseRunContext, progress: float) -> np.ndarray:
        """Return the current live inflation field for a visual progress value."""

        seed = context.fields.seed_delta
        if self._target_delta is None:
            self._target_delta = context.backend.apply_inflation_smoothing(
                seed,
                smoothing=context.config.inflation.smoothing,
            )

        blend = float(np.clip(progress, 0.0, 1.0))
        current = (1.0 - blend) * seed + blend * self._target_delta
        current = context.backend.normalize_field(current)
        current *= float(np.std(seed))
        return current.astype(np.float32)

    def is_complete(self, context: UniverseRunContext) -> bool:
        """Return true when the visual inflation stage is complete."""

        return context.state.stage_progress >= 1.0

    def build_report(self, context: UniverseRunContext) -> StageReport:
        """Build the inflation checkpoint report."""

        params = context.config.inflation
        n_e_folds = self._compute_e_folds(context)
        expansion_factor = math.exp(n_e_folds)
        curvature_suppression = math.exp(-params.smoothing * 10.0)
        seed_std = float(np.std(context.fields.seed_delta))
        inflated_std = float(np.std(context.fields.inflation_delta))
        retained_contrast = inflated_std / max(seed_std, 1.0e-8)
        fine_suppression = self._fine_power_ratio(context)

        return StageReport(
            stage_id=self.stage_id,
            title="Инфляция завершена",
            summary_lines=(
                f"Инфляционное растяжение: e^{n_e_folds:.1f}",
                f"Рост масштаба: ×{expansion_factor:.2e}",
                f"Масштаб a(t): {context.state.a:.3e}",
                f"Остаточная кривизна: {curvature_suppression:.2e} от исходной",
                f"Сохранённый контраст ряби: {retained_contrast:.2f}",
                f"Мелкая рябь подавлена: {fine_suppression:.2f}",
                "Крупные пятна теперь читаются как будущие пустоты, нити и плотные области.",
            ),
            metrics={
                "n_e_folds": n_e_folds,
                "scale_factor": context.state.a,
                "expansion_factor": expansion_factor,
                "curvature_suppression": curvature_suppression,
                "retained_contrast": retained_contrast,
                "fine_suppression": fine_suppression,
            },
        )

    @staticmethod
    def _compute_e_folds(context: UniverseRunContext) -> float:
        params = context.config.inflation
        return params.strength * params.duration * 60.0

    @staticmethod
    def _fine_power(field: np.ndarray) -> float:
        normalized = np.asarray(field, dtype=np.float32)
        normalized = normalized - float(normalized.mean())
        std = float(normalized.std())
        if std > 1.0e-8:
            normalized = normalized / std
        spectrum = np.abs(np.fft.fft2(normalized)) ** 2
        size = normalized.shape[0]
        kx = np.fft.fftfreq(size)
        ky = np.fft.fftfreq(size)
        kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
        k = np.sqrt(kx_grid * kx_grid + ky_grid * ky_grid)
        total = float(spectrum.sum()) + 1.0e-8
        return float(spectrum[k > 0.20].sum()) / total

    def _fine_power_ratio(self, context: UniverseRunContext) -> float:
        before = self._fine_power(context.fields.seed_delta)
        after = self._fine_power(context.fields.inflation_delta)
        if before <= 1.0e-8:
            return 0.0
        return float(np.clip(1.0 - after / before, 0.0, 1.0))
