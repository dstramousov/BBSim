"""Physical scale helpers for the displayed comoving universe patch."""

from __future__ import annotations

from dataclasses import dataclass

from bbsim.core.config import ScaleConfig, UniverseConfig
from bbsim.core.state import UniverseState

PARSEC_M = 3.0856775814913673e16
MPC_M = PARSEC_M * 1.0e6
AU_M = 149_597_870_700.0


@dataclass(frozen=True, slots=True)
class ScaleSample:
    """Human-readable size information for the current displayed universe patch."""

    a: float
    box_today_mpc: float
    box_now_m: float
    cell_now_m: float
    box_now_text: str
    cell_now_text: str
    box_today_text: str


def sample_scale(state: UniverseState, config: UniverseConfig) -> ScaleSample:
    """Sample current physical size for the visible comoving box.

    The simulation displays a comoving patch. `ScaleConfig.box_size_today_mpc` says how
    large that same patch would be at a = 1. The physical size at a given epoch is
    therefore `box_size_today * a`.
    """

    return sample_scale_from_values(
        a=state.a,
        grid_size=config.seed.grid_size,
        scale=config.scale,
    )


def sample_scale_from_values(a: float, grid_size: int, scale: ScaleConfig) -> ScaleSample:
    """Sample current physical size from raw scalar inputs."""

    safe_a = max(float(a), 1.0e-80)
    safe_grid = max(int(grid_size), 1)
    box_today_mpc = max(float(scale.box_size_today_mpc), 1.0e-9)
    box_today_m = box_today_mpc * MPC_M
    box_now_m = box_today_m * safe_a
    cell_now_m = box_now_m / safe_grid
    return ScaleSample(
        a=safe_a,
        box_today_mpc=box_today_mpc,
        box_now_m=box_now_m,
        cell_now_m=cell_now_m,
        box_now_text=format_length(box_now_m),
        cell_now_text=format_length(cell_now_m),
        box_today_text=format_length(box_today_m),
    )


def build_scale_overlay_lines(state: UniverseState, config: UniverseConfig) -> tuple[str, ...]:
    """Build concise overlay lines for the central field canvas."""

    sample = sample_scale(state, config)
    return (
        "Масштаб пространства",
        f"a(t): {sample.a:.2e}",
        f"видимый участок сейчас: {sample.box_now_text}",
        f"1 клетка сейчас: {sample.cell_now_text}",
        f"эквивалент сегодня: {sample.box_today_mpc:g} Mpc",
    )


def format_length(meters: float) -> str:
    """Format a length in units that are useful for cosmological storytelling."""

    value = abs(float(meters))
    if value == 0.0:
        return "0 м"
    if value < 1.0e-18:
        return f"{meters:.2e} м"

    units: tuple[tuple[str, float], ...] = (
        ("фм", 1.0e-15),
        ("пм", 1.0e-12),
        ("нм", 1.0e-9),
        ("мкм", 1.0e-6),
        ("мм", 1.0e-3),
        ("м", 1.0),
        ("км", 1.0e3),
        ("AU", AU_M),
        ("pc", PARSEC_M),
        ("kpc", PARSEC_M * 1.0e3),
        ("Mpc", MPC_M),
        ("Gpc", PARSEC_M * 1.0e9),
    )

    selected_unit = units[0]
    for unit in units:
        if value >= unit[1]:
            selected_unit = unit
        else:
            break

    suffix, factor = selected_unit
    scaled = meters / factor
    return f"{_format_scaled(scaled)} {suffix}"


def _format_scaled(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 100.0:
        return f"{value:.0f}"
    if magnitude >= 10.0:
        return f"{value:.1f}"
    if magnitude >= 1.0:
        return f"{value:.2f}"
    return f"{value:.2e}"
