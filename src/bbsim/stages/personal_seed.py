"""Pipeline stage that creates the personal primordial seed."""

from __future__ import annotations

from bbsim.core.context import UniverseRunContext
from bbsim.core.report import StageReport
from bbsim.core.seed import SeedMetrics


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


class PersonalSeedStage:
    """Create the deterministic seed field from the user's phrase."""

    stage_id = "personal_seed"
    title = "Личное зерно"

    def enter(self, context: UniverseRunContext) -> None:
        """Generate the seed and initialize density fields."""

        context.state.current_stage = self.stage_id
        seed, field = context.backend.create_personal_seed(context.config.seed)
        context.seed = seed
        context.fields.seed_delta = field
        context.fields.dark_density = 1.0 + field
        context.fields.baryon_density = 1.0 + context.backend.diffuse(field, amount=0.65)
        context.state.stage_progress = 1.0

    def step(self, context: UniverseRunContext, dt: float) -> None:
        """No-op because seed generation is an immediate checkpoint."""

        _ = (context, dt)

    def is_complete(self, context: UniverseRunContext) -> bool:
        """Return true after seed generation."""

        return context.seed is not None

    def build_report(self, context: UniverseRunContext) -> StageReport:
        """Build the personal seed checkpoint report."""

        if context.seed is None:
            raise RuntimeError("personal seed has not been created")
        metrics = context.seed.metrics
        signature_lines = _describe_seed(metrics)
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
