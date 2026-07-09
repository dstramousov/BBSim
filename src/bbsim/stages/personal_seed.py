"""Pipeline stage that creates the personal primordial seed."""

from __future__ import annotations

import numpy as np

from bbsim.core.context import UniverseRunContext
from bbsim.core.expansion import ExpansionEngine
from bbsim.core.report import StageReport
from bbsim.core.scale import sample_scale
from bbsim.core.seed import SeedMetrics
from bbsim.core.time_director import stage_screen_duration_s


def _describe_seed(metrics: SeedMetrics) -> tuple[str, ...]:
    """Build a short human-readable signature from measured seed metrics."""

    ripple = (
        "рябь слабая, структурам будет трудно зацепиться"
        if metrics.ripple_contrast < 0.25
        else "рябь сильная, возможны ранние плотные пики"
        if metrics.ripple_contrast > 0.65
        else "рябь умеренная, структура должна расти устойчиво"
    )
    voids = (
        "будущие пустоты выражены слабо"
        if metrics.void_potential < 0.35
        else "будущие пустоты выражены хорошо"
    )
    collapse = (
        "риск раннего коллапса низкий"
        if metrics.collapse_risk < 0.45
        else "риск раннего коллапса заметный"
        if metrics.collapse_risk < 0.70
        else "риск раннего коллапса высокий"
    )
    scale = (
        "доминируют крупные пятна будущей космической паутины"
        if metrics.large_scale_power >= metrics.fine_grain_power
        else "мелкая зернистость сильнее крупномасштабного рисунка"
    )
    return (ripple, voids, collapse, scale)


def _smoothstep(value: float) -> float:
    clamped = float(np.clip(value, 0.0, 1.0))
    return clamped * clamped * (3.0 - 2.0 * clamped)


class PersonalSeedStage:
    """Create the deterministic seed field from the user's phrase."""

    stage_id = "personal_seed"
    title = "Личное зерно"

    def __init__(self, visual_duration_s: float = 1.6) -> None:
        self._visual_duration_s = max(visual_duration_s, 1.0e-6)
        self._elapsed_s = 0.0
        self._target_field: np.ndarray | None = None

    def enter(self, context: UniverseRunContext) -> None:
        """Generate the seed metadata and start a visible field reveal."""

        context.state.current_stage = self.stage_id
        context.state.era = "seed"
        context.state.stage_progress = 0.0
        self._elapsed_s = 0.0

        seed, field = context.backend.create_personal_seed(context.config.seed)
        context.seed = seed
        self._target_field = field.astype(np.float32, copy=True)
        context.fields.seed_delta = np.zeros_like(self._target_field, dtype=np.float32)
        context.fields.dark_density = np.ones_like(self._target_field, dtype=np.float32)
        context.fields.baryon_density = np.ones_like(self._target_field, dtype=np.float32)
        ExpansionEngine.update_state(context.state, context.config.cosmology, update_era=False)

    def step(self, context: UniverseRunContext, dt: float) -> None:
        """Reveal the seed field over a short visual interval."""

        if self._target_field is None:
            raise RuntimeError("personal seed stage entered without a target field")

        duration = stage_screen_duration_s(context.config, self.stage_id, self._visual_duration_s)
        self._elapsed_s = min(duration, self._elapsed_s + max(dt, 0.0))
        progress = self._elapsed_s / duration
        reveal = _smoothstep(progress)
        displayed = (self._target_field * reveal).astype(np.float32)

        context.fields.seed_delta = displayed
        context.fields.dark_density = 1.0 + displayed
        context.fields.baryon_density = 1.0 + context.backend.diffuse(displayed, amount=0.65)
        context.state.stage_progress = progress

        if progress >= 1.0:
            context.fields.seed_delta = self._target_field.copy()
            context.fields.dark_density = 1.0 + context.fields.seed_delta
            context.fields.baryon_density = 1.0 + context.backend.diffuse(
                context.fields.seed_delta,
                amount=0.65,
            )

        ExpansionEngine.update_state(context.state, context.config.cosmology, update_era=False)

    def is_complete(self, context: UniverseRunContext) -> bool:
        """Return true after the seed reveal completes."""

        return context.seed is not None and context.state.stage_progress >= 1.0

    def build_report(self, context: UniverseRunContext) -> StageReport:
        """Build the personal seed checkpoint report."""

        if context.seed is None:
            raise RuntimeError("personal seed has not been created")
        metrics = context.seed.metrics
        signature_lines = _describe_seed(metrics)
        scale = sample_scale(context.state, context.config)
        return StageReport(
            stage_id=self.stage_id,
            title="Зерно создано",
            summary_lines=(
                f"Seed code: {context.seed.public_code}",
                f"Контраст ряби: {metrics.ripple_contrast:.2f}",
                f"Крупномасштабный рисунок: {metrics.large_scale_power:.2f}",
                f"Мелкая зернистость: {metrics.fine_grain_power:.2f}",
                f"Потенциал пустот: {metrics.void_potential:.2f}",
                f"Риск раннего коллапса: {metrics.collapse_risk:.2f}",
                f"Видимый участок сейчас: {scale.box_now_text}",
                f"Эквивалент этого же участка сегодня: {scale.box_today_mpc:g} Mpc",
                "Сигнатура зерна:",
                *(f"  {line}" for line in signature_lines),
            ),
            metrics={
                "ripple_contrast": metrics.ripple_contrast,
                "large_scale_power": metrics.large_scale_power,
                "fine_grain_power": metrics.fine_grain_power,
                "void_potential": metrics.void_potential,
                "collapse_risk": metrics.collapse_risk,
            },
        )
