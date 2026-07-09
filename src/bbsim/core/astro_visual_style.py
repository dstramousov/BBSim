"""Procedural astrophotography-style post-processing for epoch scenes.

This module deliberately works as a visual layer, not as simulation physics. It
keeps the existing numeric epoch fields as the source of truth, then adds
large-scale nebula texture, dust lanes, bloom, sparse stellar cores, and filmic
tone mapping so the central viewport reads like a cosmic scene instead of a flat
heatmap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AstroVisualLayers:
    """Input layers used by the procedural astrophotography renderer.

    Attributes:
        field: Main normalized semantic field for the active epoch.
        warm_haze: Optional warm emission layer such as plasma or stellar light.
        cool_haze: Optional cool gas, dark-matter scaffold, or ionized bubble layer.
        edge_layer: Optional boundary layer such as ionization fronts or ridges.
        dimming_layer: Optional opacity or neutral-gas mask.
        star_points: Optional sparse stellar source layer.
    """

    field: np.ndarray
    warm_haze: np.ndarray | None = None
    cool_haze: np.ndarray | None = None
    edge_layer: np.ndarray | None = None
    dimming_layer: np.ndarray | None = None
    star_points: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class _AstroProfile:
    nebula_strength: float
    dust_strength: float
    bloom_strength: float
    star_strength: float
    fine_grain: float
    warm_color: tuple[float, float, float]
    cool_color: tuple[float, float, float]
    edge_color: tuple[float, float, float]


_PROFILES: dict[str, _AstroProfile] = {
    "personal_seed": _AstroProfile(
        0.42,
        0.18,
        0.18,
        0.00,
        0.18,
        (0.70, 0.62, 1.00),
        (0.22, 0.42, 0.88),
        (0.56, 0.80, 1.00),
    ),
    "inflation": _AstroProfile(
        0.36,
        0.10,
        0.22,
        0.00,
        0.10,
        (0.72, 0.82, 1.00),
        (0.18, 0.36, 0.90),
        (0.70, 0.92, 1.00),
    ),
    "reheating": _AstroProfile(
        0.86,
        0.10,
        0.72,
        0.00,
        0.28,
        (1.00, 0.42, 0.10),
        (0.55, 0.08, 0.26),
        (1.00, 0.78, 0.40),
    ),
    "nucleosynthesis": _AstroProfile(
        0.74,
        0.14,
        0.42,
        0.00,
        0.22,
        (1.00, 0.56, 0.18),
        (0.20, 0.62, 0.82),
        (0.94, 0.86, 0.56),
    ),
    "recombination": _AstroProfile(
        0.48,
        0.06,
        0.16,
        0.00,
        0.34,
        (0.95, 0.76, 0.32),
        (0.16, 0.42, 0.88),
        (0.80, 0.92, 1.00),
    ),
    "dark_ages": _AstroProfile(
        0.30,
        0.46,
        0.12,
        0.00,
        0.12,
        (0.22, 0.24, 0.42),
        (0.06, 0.24, 0.46),
        (0.26, 0.44, 0.72),
    ),
    "gas_collapse": _AstroProfile(
        0.62,
        0.52,
        0.24,
        0.00,
        0.18,
        (0.36, 0.66, 0.54),
        (0.08, 0.42, 0.52),
        (0.64, 0.92, 0.80),
    ),
    "first_stars": _AstroProfile(
        1.00,
        0.08,
        1.35,
        1.40,
        0.28,
        (1.00, 0.36, 0.18),
        (0.18, 0.54, 0.86),
        (1.00, 0.76, 0.36),
    ),
    "reionization": _AstroProfile(
        1.05,
        0.08,
        1.20,
        1.05,
        0.30,
        (1.00, 0.32, 0.18),
        (0.08, 0.72, 0.78),
        (0.70, 0.94, 1.00),
    ),
}

def apply_astro_visual_style(
    rgb: np.ndarray,
    layers: AstroVisualLayers,
    *,
    stage_id: str | None,
    progress: float,
    style_strength: float,
    bloom_strength: float,
    star_density: float,
) -> np.ndarray:
    """Apply deterministic astrophotography styling to an epoch RGB frame.

    Args:
        rgb: Base RGB frame from the visual director in the range [0, 1].
        layers: Semantic scene layers used to drive nebula, dust, and stars.
        stage_id: Active simulation stage identifier.
        progress: Local stage progress in the range [0, 1].
        style_strength: Global blend amount for the styling pass.
        bloom_strength: Global multiplier for diffuse bloom and halos.
        star_density: Global multiplier for stellar source visibility.

    Returns:
        Styled RGB frame in the range [0, 1].
    """

    base_rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    strength = _clamp01(style_strength)
    if strength <= 0.0:
        return base_rgb

    profile = _PROFILES.get(stage_id or "personal_seed", _PROFILES["personal_seed"])
    p = _smoothstep(progress)
    field = _normalize01(layers.field)
    warm = _normalize_optional(layers.warm_haze, field.shape)
    cool = _normalize_optional(layers.cool_haze, field.shape)
    edge = _normalize_optional(layers.edge_layer, field.shape)
    dimming = _normalize_optional(layers.dimming_layer, field.shape)
    stars = _normalize_optional(layers.star_points, field.shape)

    large_cloud = _large_clouds(field)
    fine_cloud = _fine_clouds(field)
    nebula_mask = _normalize01(0.48 * large_cloud + 0.22 * fine_cloud + 0.18 * warm + 0.12 * cool)
    dust_mask = _dust_lanes(field, dimming=dimming, strength=profile.dust_strength)
    front_mask = _normalize01(edge + _ridge(cool) * 0.32)

    styled = base_rgb.copy()
    styled = _screen_blend(
        styled,
        nebula_mask,
        np.asarray(profile.warm_color, dtype=np.float32),
        profile.nebula_strength * (0.36 + 0.64 * p),
    )
    styled = _screen_blend(
        styled,
        np.clip(cool * 0.72 + _blur(cool, passes=2) * 0.28, 0.0, 1.0),
        np.asarray(profile.cool_color, dtype=np.float32),
        profile.nebula_strength * 0.42,
    )
    styled = _screen_blend(
        styled,
        front_mask,
        np.asarray(profile.edge_color, dtype=np.float32),
        0.28 + 0.34 * p,
    )

    if stage_id in {"first_stars", "reionization"}:
        nursery_source = np.maximum.reduce((field, warm, cool, stars))
        nursery_cloud = _normalize01(
            _large_clouds(nursery_source) + 0.45 * _fine_clouds(nursery_source)
        )
        color_split = _normalize01(
            nursery_source
            - np.roll(nursery_source, 9, axis=0)
            + np.roll(nursery_source, -13, axis=1)
        )
        styled = _screen_blend(
            styled,
            nursery_cloud * (1.0 - 0.45 * color_split),
            np.asarray(profile.cool_color, dtype=np.float32),
            0.56 + 0.20 * p,
        )
        styled = _screen_blend(
            styled,
            nursery_cloud * color_split,
            np.asarray(profile.warm_color, dtype=np.float32),
            0.34 + 0.22 * p,
        )
        styled = _screen_blend(
            styled,
            _ridge(nursery_cloud),
            np.asarray(profile.warm_color, dtype=np.float32),
            0.26 + 0.22 * p,
        )

    styled *= np.clip(1.0 - dust_mask[..., None], 0.16, 1.0)
    styled += _grain(field, amount=profile.fine_grain)[..., None]

    bloom = _bloom_source(styled, field, warm, cool, edge)
    styled += _multi_bloom(bloom) * profile.bloom_strength * _clamp01(bloom_strength)

    star_layer = _stellar_sources(stars, field=field, stage_id=stage_id, density=star_density)
    if profile.star_strength > 0.0:
        styled = _add_stars(styled, star_layer, profile=profile, progress=p)

    styled = _filmic_tone_map(styled)
    styled = _boost_saturation(styled, amount=0.16 + 0.12 * profile.nebula_strength)
    styled = np.clip(styled, 0.0, 1.0).astype(np.float32)
    return np.clip((1.0 - strength) * base_rgb + strength * styled, 0.0, 1.0).astype(np.float32)


def _large_clouds(field: np.ndarray) -> np.ndarray:
    soft = _blur(field, passes=6)
    rolled = (
        np.roll(soft, 7, axis=0)
        + np.roll(soft, -11, axis=1)
        + np.roll(np.roll(soft, 5, axis=0), -9, axis=1)
    ) / 3.0
    return _normalize01(0.68 * soft + 0.32 * rolled)


def _fine_clouds(field: np.ndarray) -> np.ndarray:
    coarse = _blur(field, passes=2)
    fine = np.clip(field - 0.46 * coarse, 0.0, 1.0)
    crossed = (
        np.roll(fine, 2, axis=0)
        + np.roll(fine, -3, axis=1)
        + np.roll(np.roll(fine, -2, axis=0), 3, axis=1)
    ) / 3.0
    return _normalize01(0.58 * fine + 0.42 * crossed)


def _dust_lanes(field: np.ndarray, *, dimming: np.ndarray, strength: float) -> np.ndarray:
    clouds = _large_clouds(field)
    ridges = _ridge(1.0 - field)
    shadows = _normalize01(0.50 * ridges + 0.35 * _ridge(clouds) + 0.15 * dimming)
    return np.clip(shadows * float(strength), 0.0, 0.82).astype(np.float32)


def _bloom_source(
    rgb: np.ndarray,
    field: np.ndarray,
    warm: np.ndarray,
    cool: np.ndarray,
    edge: np.ndarray,
) -> np.ndarray:
    luminance = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    hot = np.clip(luminance - 0.52, 0.0, 1.0) * 1.8
    source = np.clip(
        0.45 * hot + 0.28 * warm + 0.16 * cool + 0.11 * edge + field**3 * 0.12,
        0.0,
        1.0,
    )
    return source[..., None] * np.array([1.00, 0.72, 0.42], dtype=np.float32)


def _multi_bloom(source_rgb: np.ndarray) -> np.ndarray:
    bloom = np.asarray(source_rgb, dtype=np.float32)
    return (
        0.52 * _blur_rgb(bloom, passes=2)
        + 0.33 * _blur_rgb(bloom, passes=5)
        + 0.15 * _blur_rgb(bloom, passes=10)
    )


def _stellar_sources(
    stars: np.ndarray,
    *,
    field: np.ndarray,
    stage_id: str | None,
    density: float,
) -> np.ndarray:
    if stage_id not in {"first_stars", "reionization"}:
        return np.zeros_like(field, dtype=np.float32)

    peak_source = np.maximum(stars, field)
    peak_percentile = max(93.0, 98.9 - 1.7 * float(density))
    points = _local_peaks(peak_source, percentile=peak_percentile)
    density_scale = 0.34 if stage_id == "first_stars" else 0.55
    background = _deterministic_star_speckles(
        field.shape, density=max(0.0, float(density)) * density_scale
    )
    background *= np.clip(0.35 + 0.65 * field, 0.0, 1.0)
    return _contrast(_normalize01(np.maximum(points, background)), gamma=0.28)


def _deterministic_star_speckles(shape: tuple[int, int], *, density: float) -> np.ndarray:
    if density <= 0.0:
        return np.zeros(shape, dtype=np.float32)
    height, width = shape[:2]
    y, x = np.indices((height, width), dtype=np.float32)
    noise = np.sin(x * 12.9898 + y * 78.233) * 43758.5453
    noise = noise - np.floor(noise)
    threshold = 1.0 - min(0.030, 0.006 * float(density))
    speckles = np.where(noise >= threshold, noise, 0.0).astype(np.float32)
    return _contrast(_normalize01(speckles), gamma=0.22)


def _add_stars(
    rgb: np.ndarray,
    stars: np.ndarray,
    *,
    profile: _AstroProfile,
    progress: float,
) -> np.ndarray:
    if stars.size == 0 or float(np.nanmax(stars)) <= 1.0e-7:
        return rgb

    cores = _contrast(stars, gamma=0.18)
    close_halo = _normalize01(_blur(cores, passes=1))
    wide_halo = _normalize01(_blur(cores, passes=5))
    nebula_halo = _normalize01(_blur(cores, passes=14))
    spikes = _star_spikes(cores)
    out = rgb.copy()
    strength = profile.star_strength * (0.40 + 0.60 * progress)
    out += nebula_halo[..., None] * np.array([0.95, 0.20, 0.16], dtype=np.float32) * 0.58 * strength
    out += nebula_halo[..., None] * np.array([0.08, 0.48, 0.86], dtype=np.float32) * 0.38 * strength
    out += wide_halo[..., None] * np.array([0.45, 0.68, 1.00], dtype=np.float32) * 0.44 * strength
    out += close_halo[..., None] * np.array([1.00, 0.50, 0.20], dtype=np.float32) * 0.48 * strength
    out += spikes[..., None] * np.array([0.72, 0.88, 1.00], dtype=np.float32) * 0.42 * strength
    out += cores[..., None] * np.array([1.00, 0.96, 0.78], dtype=np.float32) * 1.15 * strength
    return out


def _star_spikes(cores: np.ndarray) -> np.ndarray:
    horizontal = np.maximum(np.roll(cores, 1, axis=1), np.roll(cores, -1, axis=1)) * 0.58
    vertical = np.maximum(np.roll(cores, 1, axis=0), np.roll(cores, -1, axis=0)) * 0.58
    diagonal = np.maximum(
        np.roll(np.roll(cores, 1, axis=0), 1, axis=1),
        np.roll(np.roll(cores, -1, axis=0), -1, axis=1),
    ) * 0.34
    return np.clip(horizontal + vertical + diagonal, 0.0, 1.0).astype(np.float32)


def _local_peaks(field: np.ndarray, *, percentile: float) -> np.ndarray:
    data = np.clip(field, 0.0, 1.0)
    if data.size == 0 or float(np.nanstd(data)) <= 1.0e-7:
        return np.zeros_like(data, dtype=np.float32)
    threshold = float(np.percentile(data, percentile))
    local_max = data >= np.roll(data, 1, axis=0)
    local_max &= data >= np.roll(data, -1, axis=0)
    local_max &= data >= np.roll(data, 1, axis=1)
    local_max &= data >= np.roll(data, -1, axis=1)
    return np.where(local_max & (data >= threshold), data, 0.0).astype(np.float32)


def _screen_blend(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: np.ndarray,
    amount: float,
) -> np.ndarray:
    alpha = np.clip(mask * float(amount), 0.0, 1.0)[..., None]
    layer = np.clip(color.reshape(1, 1, 3), 0.0, 1.0)
    return 1.0 - (1.0 - rgb) * (1.0 - alpha * layer)


def _grain(field: np.ndarray, *, amount: float) -> np.ndarray:
    fine = _fine_clouds(field)
    grain = np.clip((fine - 0.5) * float(amount), -0.08, 0.08)
    return grain.astype(np.float32)


def _boost_saturation(rgb: np.ndarray, *, amount: float) -> np.ndarray:
    luminance = rgb[..., 0:1] * 0.299 + rgb[..., 1:2] * 0.587 + rgb[..., 2:3] * 0.114
    return luminance + (rgb - luminance) * (1.0 + float(amount))


def _filmic_tone_map(rgb: np.ndarray) -> np.ndarray:
    data = np.maximum(np.asarray(rgb, dtype=np.float32), 0.0)
    mapped = data / (1.0 + data)
    return np.power(np.clip(mapped * 1.18, 0.0, 1.0), 0.92).astype(np.float32)


def _normalize_optional(field: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if field is None:
        return np.zeros(shape, dtype=np.float32)
    return _normalize01(field)


def _normalize01(field: np.ndarray) -> np.ndarray:
    data = np.asarray(field, dtype=np.float32)
    if data.size == 0:
        return data
    min_value = float(np.nanmin(data))
    max_value = float(np.nanmax(data))
    span = max(max_value - min_value, 1.0e-8)
    return ((data - min_value) / span).astype(np.float32)


def _contrast(field: np.ndarray, *, gamma: float) -> np.ndarray:
    return np.power(np.clip(field, 0.0, 1.0), max(float(gamma), 1.0e-4)).astype(np.float32)


def _ridge(field: np.ndarray) -> np.ndarray:
    data = np.asarray(field, dtype=np.float32)
    smooth = _blur(data, passes=2)
    return np.clip(data - 0.74 * smooth, 0.0, 1.0).astype(np.float32)


def _blur(field: np.ndarray, *, passes: int) -> np.ndarray:
    data = np.asarray(field, dtype=np.float32)
    out = data.copy()
    for _ in range(max(0, int(passes))):
        out = (
            out
            + np.roll(out, 1, axis=0)
            + np.roll(out, -1, axis=0)
            + np.roll(out, 1, axis=1)
            + np.roll(out, -1, axis=1)
            + 0.50 * np.roll(np.roll(out, 1, axis=0), 1, axis=1)
            + 0.50 * np.roll(np.roll(out, 1, axis=0), -1, axis=1)
            + 0.50 * np.roll(np.roll(out, -1, axis=0), 1, axis=1)
            + 0.50 * np.roll(np.roll(out, -1, axis=0), -1, axis=1)
        ) / 7.0
    return out.astype(np.float32)


def _blur_rgb(rgb: np.ndarray, *, passes: int) -> np.ndarray:
    channels = [_blur(rgb[..., channel], passes=passes) for channel in range(3)]
    return np.stack(channels, axis=-1).astype(np.float32)


def _smoothstep(value: float) -> float:
    clamped = _clamp01(value)
    return clamped * clamped * (3.0 - 2.0 * clamped)


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))
