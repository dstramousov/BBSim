"""Pipeline stage that creates the personal primordial seed."""

from __future__ import annotations

from bbsim.core.context import UniverseRunContext
from bbsim.core.report import StageReport


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
        return StageReport(
            stage_id=self.stage_id,
            title="Зерно создано",
            summary_lines=(
                f"Seed code: {context.seed.public_code}",
                f"Контраст ряби: {metrics.ripple_contrast:.2f}",
                f"Потенциал пустот: {metrics.void_potential:.2f}",
                f"Риск раннего коллапса: {metrics.collapse_risk:.2f}",
            ),
            metrics={
                "ripple_contrast": metrics.ripple_contrast,
                "void_potential": metrics.void_potential,
                "collapse_risk": metrics.collapse_risk,
            },
        )
