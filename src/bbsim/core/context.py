"""Universe run context creation."""

from __future__ import annotations

from dataclasses import dataclass

from bbsim.core.checkpoint import RunHistory
from bbsim.core.config import UniverseConfig
from bbsim.core.fields import UniverseFields, create_empty_fields
from bbsim.core.seed import PersonalSeed
from bbsim.core.state import UniverseState
from bbsim.numeric.backend import NumericBackend


@dataclass(slots=True)
class UniverseRunContext:
    """Mutable context for exactly one universe run.

    This object is intentionally not a singleton. It is passed explicitly to pipeline
    stages so multiple deterministic runs can be tested or compared.
    """

    config: UniverseConfig
    backend: NumericBackend
    state: UniverseState
    fields: UniverseFields
    history: RunHistory
    seed: PersonalSeed | None = None


def create_run_context(config: UniverseConfig, backend: NumericBackend) -> UniverseRunContext:
    """Create an empty context for a new universe run.

    Args:
        config: Immutable run configuration.
        backend: Numeric backend used for field operations.

    Returns:
        New mutable universe run context.
    """

    return UniverseRunContext(
        config=config,
        backend=backend,
        state=UniverseState(curvature=config.cosmology.omega_k),
        fields=create_empty_fields(config.seed.grid_size),
        history=RunHistory(),
    )
