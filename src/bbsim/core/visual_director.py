"""Visual direction helpers for cinematic epoch playback.

The simulation fields are deliberately conservative numeric arrays. This module is
allowed to be theatrical: it converts a normalized field into an RGB frame, applies
stage-specific motion/energy cues, and cross-fades neighboring visual profiles so
stage borders feel like continuous evolution rather than hard slideshow cuts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bbsim.core.config import VisualDirectorConfig


@dataclass(frozen=True, slots=True)
class VisualFrame:
    """Rendered RGB frame and metadata for the current epoch visualization."""

    rgb: np.ndarray
    profile_id: str
    transition_mix: float


_STAGE_ORDER: tuple[str, ...] = (
    "personal_seed",
    "inflation",
    "reheating",
    "nucleosynthesis",
    "recombination",
    "dark_ages",
    "gas_collapse",
    "first_stars",
)

_STAGE_PROFILE: dict[str, str] = {
    "personal_seed": "seed",
    "inflation": "inflation",
    "reheating": "reheating",
    "nucleosynthesis": "nucleosynthesis",
    "recombination": "recombination",
    "dark_ages": "dark_ages",
    "gas_collapse": "gas_collapse",
    "first_stars": "first_stars",
    "stellar_radiation": "stellar_radiation",
    "ionized_bubbles": "ionized_bubbles",
    "cold_gas": "cold_gas",
    "collapse_sites": "collapse_sites",
    "dark_matter": "dark_matter",
    "baryon_gas": "baryon_gas",
    "mixed_matter": "mixed_matter",
    "gravitational_potential": "gravitational_potential",
    "halo_candidates": "halo_candidates",
}

_PALETTES: dict[str, tuple[tuple[float, tuple[float, float, float]], ...]] = {
    "seed": (
        (0.00, (0.01, 0.015, 0.055)),
        (0.28, (0.04, 0.055, 0.22)),
        (0.55, (0.28, 0.42, 0.68)),
        (0.78, (0.73, 0.86, 0.91)),
        (1.00, (1.00, 0.94, 0.70)),
    ),
    "inflation": (
        (0.00, (0.01, 0.012, 0.05)),
        (0.26, (0.035, 0.08, 0.26)),
        (0.56, (0.22, 0.46, 0.78)),
        (0.80, (0.75, 0.91, 0.98)),
        (1.00, (1.00, 0.96, 0.74)),
    ),
    "reheating": (
        (0.00, (0.035, 0.015, 0.075)),
        (0.22, (0.22, 0.035, 0.18)),
        (0.48, (0.68, 0.15, 0.09)),
        (0.76, (1.00, 0.55, 0.16)),
        (1.00, (1.00, 0.96, 0.72)),
    ),
    "nucleosynthesis": (
        (0.00, (0.02, 0.025, 0.075)),
        (0.24, (0.08, 0.12, 0.34)),
        (0.54, (0.16, 0.50, 0.66)),
        (0.78, (0.62, 0.86, 0.78)),
        (1.00, (0.98, 0.84, 0.46)),
    ),
    "recombination": (
        (0.00, (0.012, 0.012, 0.055)),
        (0.30, (0.07, 0.17, 0.42)),
        (0.52, (0.42, 0.66, 0.78)),
        (0.74, (0.92, 0.80, 0.42)),
        (1.00, (1.00, 0.96, 0.78)),
    ),
    "dark_ages": (
        (0.00, (0.003, 0.005, 0.020)),
        (0.24, (0.018, 0.025, 0.070)),
        (0.52, (0.075, 0.115, 0.205)),
        (0.78, (0.265, 0.350, 0.505)),
        (1.00, (0.690, 0.770, 0.860)),
    ),

    "gas_collapse": (
        (0.00, (0.002, 0.006, 0.018)),
        (0.24, (0.012, 0.050, 0.070)),
        (0.52, (0.055, 0.180, 0.185)),
        (0.78, (0.230, 0.470, 0.420)),
        (1.00, (0.820, 0.940, 0.740)),
    ),
    "first_stars": (
        (0.00, (0.002, 0.004, 0.016)),
        (0.22, (0.018, 0.030, 0.070)),
        (0.48, (0.070, 0.125, 0.180)),
        (0.72, (0.620, 0.520, 0.250)),
        (1.00, (1.000, 0.970, 0.760)),
    ),
    "stellar_radiation": (
        (0.00, (0.002, 0.003, 0.015)),
        (0.20, (0.030, 0.022, 0.055)),
        (0.48, (0.210, 0.085, 0.085)),
        (0.76, (0.950, 0.440, 0.160)),
        (1.00, (1.000, 0.950, 0.660)),
    ),
    "ionized_bubbles": (
        (0.00, (0.002, 0.006, 0.018)),
        (0.24, (0.020, 0.040, 0.085)),
        (0.52, (0.090, 0.185, 0.255)),
        (0.76, (0.330, 0.560, 0.650)),
        (1.00, (0.850, 0.960, 1.000)),
    ),
    "cold_gas": (
        (0.00, (0.002, 0.008, 0.018)),
        (0.25, (0.012, 0.060, 0.078)),
        (0.55, (0.050, 0.220, 0.210)),
        (0.78, (0.240, 0.560, 0.470)),
        (1.00, (0.860, 0.980, 0.760)),
    ),
    "collapse_sites": (
        (0.00, (0.003, 0.002, 0.012)),
        (0.22, (0.020, 0.026, 0.060)),
        (0.50, (0.090, 0.160, 0.190)),
        (0.76, (0.450, 0.620, 0.380)),
        (1.00, (1.000, 0.960, 0.660)),
    ),
    "dark_matter": (
        (0.00, (0.002, 0.003, 0.018)),
        (0.24, (0.020, 0.014, 0.070)),
        (0.52, (0.100, 0.060, 0.225)),
        (0.78, (0.285, 0.170, 0.520)),
        (1.00, (0.730, 0.620, 0.900)),
    ),
    "baryon_gas": (
        (0.00, (0.004, 0.010, 0.026)),
        (0.25, (0.020, 0.075, 0.110)),
        (0.55, (0.085, 0.250, 0.300)),
        (0.78, (0.275, 0.520, 0.500)),
        (1.00, (0.780, 0.900, 0.760)),
    ),
    "mixed_matter": (
        (0.00, (0.003, 0.005, 0.020)),
        (0.24, (0.020, 0.025, 0.075)),
        (0.50, (0.070, 0.105, 0.210)),
        (0.74, (0.250, 0.330, 0.490)),
        (1.00, (0.760, 0.835, 0.855)),
    ),
    "gravitational_potential": (
        (0.00, (0.002, 0.003, 0.014)),
        (0.26, (0.016, 0.020, 0.060)),
        (0.52, (0.060, 0.080, 0.165)),
        (0.78, (0.190, 0.260, 0.420)),
        (1.00, (0.660, 0.740, 0.880)),
    ),
    "halo_candidates": (
        (0.00, (0.003, 0.002, 0.012)),
        (0.22, (0.035, 0.018, 0.070)),
        (0.50, (0.190, 0.065, 0.260)),
        (0.76, (0.660, 0.270, 0.580)),
        (1.00, (1.000, 0.790, 0.960)),
    ),
}


def render_visual_frame(
    field: np.ndarray,
    *,
    stage_id: str | None,
    progress: float,
    config: VisualDirectorConfig,
) -> VisualFrame:
    """Convert a field into a smooth RGB frame for the current epoch.

    Args:
        field: Numeric field normalized or raw. It will be normalized defensively.
        stage_id: Current pipeline stage id.
        progress: Local stage progress in the range [0, 1].
        config: Tunable visual-director settings.

    Returns:
        RGB float32 frame in the range [0, 1] plus transition metadata.
    """

    stage = stage_id or "personal_seed"
    profile = _STAGE_PROFILE.get(stage, "seed")
    p = _clamp01(progress)
    values = _normalize(field)
    values = _apply_stage_motion(values, stage, p, config)
    values = _apply_stage_energy(values, stage, p, config)

    current_rgb = _apply_palette(values, _PALETTES[profile])
    previous_profile = _previous_profile(stage)
    if previous_profile is None:
        transition_mix = 1.0
        rgb = current_rgb
    else:
        transition_mix = _transition_mix(p, config.epoch_transition_fraction)
        previous_values = _apply_previous_bridge(values, stage, p)
        previous_rgb = _apply_palette(previous_values, _PALETTES[previous_profile])
        rgb = _lerp(previous_rgb, current_rgb, transition_mix)

    rgb = _apply_epoch_glow(rgb, values, stage, p, config)
    return VisualFrame(
        rgb=np.clip(rgb, 0.0, 1.0).astype(np.float32),
        profile_id=profile,
        transition_mix=float(transition_mix),
    )


def _previous_profile(stage_id: str) -> str | None:
    try:
        index = _STAGE_ORDER.index(stage_id)
    except ValueError:
        return None
    if index <= 0:
        return None
    previous_stage = _STAGE_ORDER[index - 1]
    return _STAGE_PROFILE[previous_stage]


def _apply_stage_motion(
    values: np.ndarray,
    stage_id: str,
    progress: float,
    config: VisualDirectorConfig,
) -> np.ndarray:
    if stage_id != "inflation":
        return values

    p = _smoothstep(progress)
    zoom = 1.0 + max(config.inflation_zoom_strength, 0.0) * p
    zoomed = _zoom_from_center(values, zoom)
    # Keep some source texture early in the stage so the user sees small-scale
    # ruffle becoming large-scale imprint instead of an instant blur.
    return _lerp(values, zoomed, min(1.0, 0.25 + 0.75 * p))


def _apply_stage_energy(
    values: np.ndarray,
    stage_id: str,
    progress: float,
    config: VisualDirectorConfig,
) -> np.ndarray:
    p = _smoothstep(progress)
    clamped = np.clip(values, 0.0, 1.0)
    if stage_id == "personal_seed":
        visibility = 0.12 + 0.88 * p
        return np.clip(clamped * visibility, 0.0, 1.0)
    if stage_id == "inflation":
        # Inflation is not a fireball: it is smoother and quieter while scale races.
        return np.clip((0.82 + 0.18 * p) * np.power(clamped, 1.0 + 0.18 * p), 0.0, 1.0)
    if stage_id == "reheating":
        pulse = config.reheating_pulse_strength * np.sin(progress * np.pi * 10.0)
        return np.clip(np.power(clamped, 0.55) + pulse, 0.0, 1.0)
    if stage_id == "nucleosynthesis":
        return np.clip((1.0 - 0.28 * p) * clamped + 0.10 * (1.0 - p), 0.0, 1.0)
    if stage_id == "recombination":
        clearing = p * max(config.recombination_clearing_strength, 0.0)
        return np.clip((0.70 + clearing) * clamped, 0.0, 1.0)
    if stage_id == "dark_ages":
        # The light fades, but contrast in the hidden scaffold grows.
        return np.clip(np.power(clamped, 0.78) * (0.42 + 0.58 * p), 0.0, 1.0)
    if stage_id == "gas_collapse":
        # Cooling gas should become sharper and less hazy without turning into stars.
        return np.clip(np.power(clamped, 1.05 - 0.28 * p) * (0.58 + 0.42 * p), 0.0, 1.0)
    if stage_id == "first_stars":
        # Rare first lights appear late enough to feel earned after gas collapse.
        late = _smoothstep(max(0.0, (progress - 0.12) / 0.88))
        return np.clip(np.power(clamped, 0.44) * (0.42 + 0.58 * late), 0.0, 1.0)
    if stage_id == "stellar_radiation":
        return np.clip(np.power(clamped, 0.50) * (0.72 + 0.28 * p), 0.0, 1.0)
    if stage_id == "ionized_bubbles":
        return np.clip(np.power(clamped, 0.70) * (0.54 + 0.46 * p), 0.0, 1.0)
    if stage_id == "cold_gas":
        return np.clip(np.power(clamped, 0.92) * 0.92 + 0.035, 0.0, 1.0)
    if stage_id == "collapse_sites":
        return np.clip(np.power(clamped, 0.36) * (0.62 + 0.38 * p), 0.0, 1.0)
    if stage_id == "dark_matter":
        return np.clip(np.power(clamped, 0.62), 0.0, 1.0)
    if stage_id == "baryon_gas":
        return np.clip(np.power(clamped, 1.20) * 0.86 + 0.05, 0.0, 1.0)
    if stage_id == "mixed_matter":
        return np.clip(np.power(clamped, 0.85), 0.0, 1.0)
    if stage_id == "gravitational_potential":
        return np.clip(np.power(clamped, 0.74), 0.0, 1.0)
    if stage_id == "halo_candidates":
        return np.clip(np.power(clamped, 0.42), 0.0, 1.0)
    return clamped


def _apply_previous_bridge(values: np.ndarray, stage_id: str, progress: float) -> np.ndarray:
    # Early in a new stage, keep the previous visual state close so epoch borders
    # are continuous. This is a visual bridge only; the simulation fields still
    # belong to the current stage.
    p = _smoothstep(progress)
    if stage_id == "reheating":
        return np.clip(values * (0.55 + 0.30 * p), 0.0, 1.0)
    if stage_id == "nucleosynthesis":
        return np.clip(np.power(values, 0.65), 0.0, 1.0)
    if stage_id == "recombination":
        return np.clip(values * (0.95 - 0.20 * p), 0.0, 1.0)
    if stage_id == "dark_ages":
        return np.clip(values * (0.74 - 0.28 * p), 0.0, 1.0)
    if stage_id == "gas_collapse":
        return np.clip(np.power(values, 0.84) * (0.64 + 0.24 * p), 0.0, 1.0)
    if stage_id == "first_stars":
        return np.clip(np.power(values, 0.82) * (0.46 + 0.30 * p), 0.0, 1.0)
    return values


def _apply_epoch_glow(
    rgb: np.ndarray,
    values: np.ndarray,
    stage_id: str,
    progress: float,
    config: VisualDirectorConfig,
) -> np.ndarray:
    _ = config
    p = _smoothstep(progress)
    if stage_id == "reheating":
        glow = (values[..., None] ** 2) * (0.18 + 0.12 * np.sin(progress * np.pi * 8.0))
        return rgb + glow * np.array([1.0, 0.48, 0.18], dtype=np.float32)
    if stage_id == "inflation":
        edge = _grid_like_sheen(values.shape, progress)
        return rgb + edge[..., None] * np.array([0.10, 0.16, 0.28], dtype=np.float32)
    if stage_id == "recombination":
        haze = (1.0 - p) * 0.13
        return _lerp(rgb + haze, rgb, p)
    if stage_id in {"dark_ages", "dark_matter", "mixed_matter", "gravitational_potential"}:
        filament = _filament_sheen(values, progress)
        return rgb + filament[..., None] * np.array([0.18, 0.20, 0.34], dtype=np.float32)
    if stage_id in {"gas_collapse", "cold_gas"}:
        cold_flow = _filament_sheen(values, progress)
        return rgb + cold_flow[..., None] * np.array([0.08, 0.24, 0.20], dtype=np.float32)
    if stage_id == "collapse_sites":
        core = (values[..., None] ** 3) * (0.12 + 0.12 * p)
        return rgb + core * np.array([0.85, 0.90, 0.48], dtype=np.float32)
    if stage_id == "first_stars":
        star_core = (values[..., None] ** 4) * (0.30 + 0.30 * p)
        halo = _filament_sheen(values, progress) * 0.65
        return rgb + star_core * np.array([1.00, 0.88, 0.38], dtype=np.float32) + halo[..., None] * np.array([0.18, 0.12, 0.04], dtype=np.float32)
    if stage_id == "stellar_radiation":
        glow = (values[..., None] ** 2) * (0.22 + 0.14 * np.sin(progress * np.pi * 10.0))
        return rgb + glow * np.array([1.0, 0.42, 0.14], dtype=np.float32)
    if stage_id == "ionized_bubbles":
        bubble_edge = _filament_sheen(values, progress)
        return rgb + bubble_edge[..., None] * np.array([0.18, 0.38, 0.45], dtype=np.float32)
    if stage_id == "halo_candidates":
        sparkle = (values[..., None] ** 3) * (0.14 + 0.10 * p)
        return rgb + sparkle * np.array([0.95, 0.52, 0.85], dtype=np.float32)
    if stage_id == "baryon_gas":
        gas_glow = (values[..., None] ** 2) * 0.055
        return rgb + gas_glow * np.array([0.14, 0.26, 0.22], dtype=np.float32)
    return rgb


def _apply_palette(
    values: np.ndarray,
    palette: tuple[tuple[float, tuple[float, float, float]], ...],
) -> np.ndarray:
    clamped = np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)
    positions = np.asarray([item[0] for item in palette], dtype=np.float32)
    colors = np.asarray([item[1] for item in palette], dtype=np.float32)

    flat = clamped.reshape(-1)
    channels = [np.interp(flat, positions, colors[:, channel]) for channel in range(3)]
    return np.stack(channels, axis=1).reshape((*clamped.shape, 3)).astype(np.float32)


def _zoom_from_center(values: np.ndarray, zoom: float) -> np.ndarray:
    if zoom <= 1.0:
        return values
    height, width = values.shape[:2]
    y, x = np.indices((height, width), dtype=np.float32)
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    src_y = center_y + (y - center_y) / zoom
    src_x = center_x + (x - center_x) / zoom
    return _bilinear_sample(values, src_y, src_x)


def _bilinear_sample(values: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    height, width = values.shape[:2]
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    x0 = np.clip(x0, 0, width - 1)

    wy = y - y0
    wx = x - x0
    top = (1.0 - wx) * values[y0, x0] + wx * values[y0, x1]
    bottom = (1.0 - wx) * values[y1, x0] + wx * values[y1, x1]
    return ((1.0 - wy) * top + wy * bottom).astype(np.float32)


def _grid_like_sheen(shape: tuple[int, int], progress: float) -> np.ndarray:
    height, width = shape[:2]
    y, x = np.indices((height, width), dtype=np.float32)
    spacing = 10.0 + 44.0 * _smoothstep(progress)
    phase = progress * spacing * 1.7
    vertical = np.exp(-((np.mod(x + phase, spacing) - spacing / 2.0) ** 2) / 18.0)
    horizontal = np.exp(-((np.mod(y + phase, spacing) - spacing / 2.0) ** 2) / 18.0)
    return (0.018 * (vertical + horizontal)).astype(np.float32)


def _filament_sheen(values: np.ndarray, progress: float) -> np.ndarray:
    p = _smoothstep(progress)
    shifted = (
        np.roll(values, 2, axis=0)
        + np.roll(values, -2, axis=1)
        + np.roll(values, 5, axis=0)
    ) / 3.0
    ridge = np.clip(values - 0.72 * shifted, 0.0, 1.0)
    return (0.10 * (0.35 + 0.65 * p) * ridge).astype(np.float32)


def _normalize(field: np.ndarray) -> np.ndarray:
    data = np.asarray(field, dtype=np.float32)
    if data.size == 0:
        return data
    min_value = float(np.nanmin(data))
    max_value = float(np.nanmax(data))
    span = max(max_value - min_value, 1.0e-8)
    return ((data - min_value) / span).astype(np.float32)


def _transition_mix(progress: float, transition_fraction: float) -> float:
    if transition_fraction <= 0.0:
        return 1.0
    return _smoothstep(_clamp01(progress / transition_fraction))


def _smoothstep(value: float) -> float:
    clamped = _clamp01(value)
    return clamped * clamped * (3.0 - 2.0 * clamped)


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _lerp(a: np.ndarray, b: np.ndarray, amount: float) -> np.ndarray:
    return (1.0 - amount) * a + amount * b
