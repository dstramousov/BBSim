"""Semantic visual composition for live universe epochs.

Simulation fields are numeric diagnostics, not final educational artwork. This
module builds an RGB scene for each current epoch by combining several physical-ish
fields before the frame reaches the UI. The goal is to show what the right-side
explanation describes: plasma, CMB release, hidden dark scaffolds, gas collapse,
first lights, and reionization bubbles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bbsim.core.config import VisualDirectorConfig
from bbsim.core.fields import UniverseFields
from bbsim.core.visual_director import VisualFrame, render_visual_frame


@dataclass(frozen=True, slots=True)
class EraVisualScene:
    """Prepared numeric layers for a semantic epoch visualization.

    Attributes:
        field: Main normalized scalar field used by the visual director.
        visual_stage_id: Visual profile identifier passed to the visual director.
        star_points: Optional sparse point sources that should read as first light.
        warm_haze: Optional diffuse warm emission layer.
        cool_haze: Optional diffuse cool gas or ionized-bubble layer.
        edge_layer: Optional edge/front layer used for boundaries and ionization fronts.
        dimming_layer: Optional mask that darkens neutral or opaque regions.
    """

    field: np.ndarray
    visual_stage_id: str
    star_points: np.ndarray | None = None
    warm_haze: np.ndarray | None = None
    cool_haze: np.ndarray | None = None
    edge_layer: np.ndarray | None = None
    dimming_layer: np.ndarray | None = None


def render_era_visual_frame(
    fields: UniverseFields,
    *,
    stage_id: str | None,
    progress: float,
    config: VisualDirectorConfig,
) -> VisualFrame:
    """Render the automatic semantic scene for the current epoch.

    Args:
        fields: Current mutable universe fields.
        stage_id: Current pipeline stage identifier.
        progress: Local stage progress in the range [0, 1].
        config: Visual director settings.

    Returns:
        RGB frame for the central UI canvas.
    """

    scene = compose_era_visual_scene(fields, stage_id=stage_id, progress=progress)
    frame = render_visual_frame(
        scene.field,
        stage_id=scene.visual_stage_id,
        progress=progress,
        config=config,
    )
    rgb = _apply_scene_overlays(frame.rgb, scene, progress)
    return VisualFrame(
        rgb=np.clip(rgb, 0.0, 1.0).astype(np.float32),
        profile_id=frame.profile_id,
        transition_mix=frame.transition_mix,
    )


def compose_era_visual_scene(
    fields: UniverseFields,
    *,
    stage_id: str | None,
    progress: float,
) -> EraVisualScene:
    """Build semantic scalar layers for the current epoch.

    Args:
        fields: Current universe fields.
        stage_id: Current pipeline stage identifier.
        progress: Local stage progress in the range [0, 1].

    Returns:
        Scene layers in UI image orientation.
    """

    stage = stage_id or "personal_seed"
    p = _smoothstep(progress)

    seed = _field_view(fields.seed_delta)
    inflation = _field_view(fields.inflation_delta)
    radiation = _field_view(fields.radiation)
    cmb = _field_view(fields.cmb)
    dark = _field_view(fields.dark_density)
    baryon = _field_view(fields.baryon_density)
    potential = _field_view(fields.gravitational_potential)
    halo = _field_view(fields.halo_density)
    future_sites = _field_view(fields.future_star_sites)
    cold_gas = _field_view(fields.cold_gas_density)
    molecular = _field_view(fields.molecular_cooling)
    collapse = _field_view(fields.collapse_sites)
    first_stars = _field_view(fields.first_star_density)
    ignition = _field_view(fields.stellar_ignition)
    stellar_radiation = _field_view(fields.stellar_radiation)
    bubbles = _field_view(fields.ionized_bubbles)
    ionization = _field_view(fields.ionization)

    if stage == "personal_seed":
        base = _prefer(seed, dark)
        field = _contrast(_normalize01(base), gamma=0.82)
        return EraVisualScene(
            field=field,
            visual_stage_id="personal_seed",
            edge_layer=_ridge(field) * (0.35 + 0.65 * p),
        )

    if stage == "inflation":
        base = _prefer(inflation, seed)
        field = _contrast(_normalize01(base), gamma=1.08)
        stretched_ridges = _ridge(field) * (0.35 + 0.65 * p)
        return EraVisualScene(
            field=field,
            visual_stage_id="inflation",
            cool_haze=stretched_ridges,
            edge_layer=_grid_front(field.shape, progress) * 0.7,
        )

    if stage == "reheating":
        plasma = _prefer(radiation, inflation, seed)
        field = _contrast(_normalize01(plasma), gamma=0.58)
        turbulent = _normalize01(field + 0.35 * _rolled_difference(field))
        return EraVisualScene(
            field=turbulent,
            visual_stage_id="reheating",
            warm_haze=_blur3(field) * (0.55 + 0.45 * p),
            edge_layer=_ridge(turbulent) * 0.45,
        )

    if stage == "nucleosynthesis":
        plasma = _prefer(radiation, inflation, seed)
        base = _normalize01(plasma)
        islands = _soft_peaks(base, threshold=0.70, width=0.18)
        field = _normalize01(0.62 * _contrast(base, gamma=0.72) + 0.38 * _blur3(base))
        return EraVisualScene(
            field=field,
            visual_stage_id="nucleosynthesis",
            warm_haze=islands * (0.25 + 0.45 * p),
            cool_haze=_blur3(field) * 0.16,
        )

    if stage == "recombination":
        imprint = _prefer(cmb, radiation, inflation, seed)
        field = _contrast(_normalize01(imprint), gamma=0.88)
        haze = _blur3(field) * (1.0 - p)
        clear_edges = _ridge(field) * p
        return EraVisualScene(
            field=field,
            visual_stage_id="recombination",
            warm_haze=haze * 0.24,
            edge_layer=clear_edges * 0.55,
            dimming_layer=haze * 0.22,
        )

    if stage == "dark_ages":
        scaffold = _normalize01(
            0.48 * _normalize01(_prefer(dark, seed))
            + 0.32 * _normalize01(_prefer(potential, halo, dark))
            + 0.20 * _normalize01(_prefer(halo, future_sites, dark))
        )
        gas = _normalize01(_prefer(baryon, cold_gas, scaffold))
        field = _normalize01(0.76 * scaffold + 0.24 * _blur3(gas))
        return EraVisualScene(
            field=_contrast(field, gamma=0.62),
            visual_stage_id="dark_ages",
            cool_haze=_ridge(scaffold) * (0.25 + 0.45 * p),
            dimming_layer=np.clip(0.70 - 0.25 * field, 0.0, 1.0),
        )

    if stage == "gas_collapse":
        scaffold = _normalize01(0.54 * _prefer(halo, potential, dark) + 0.46 * _prefer(dark, seed))
        gas = _normalize01(
            0.44 * _prefer(baryon, scaffold)
            + 0.36 * _prefer(cold_gas, baryon)
            + 0.20 * _prefer(molecular, cold_gas)
        )
        knots = _normalize01(_prefer(collapse, future_sites, halo))
        field = _normalize01(0.30 * scaffold + 0.38 * gas + 0.32 * knots)
        return EraVisualScene(
            field=_contrast(field, gamma=0.56),
            visual_stage_id="gas_collapse",
            cool_haze=_blur3(gas) * (0.22 + 0.48 * p),
            warm_haze=_soft_peaks(knots, threshold=0.58, width=0.22) * 0.22,
            edge_layer=_ridge(knots) * 0.65,
        )

    if stage == "first_stars":
        gas = _normalize01(_prefer(cold_gas, baryon, collapse))
        knots = _normalize01(_prefer(collapse, future_sites, halo))
        stars = _normalize01(_prefer(first_stars, ignition, collapse))
        radiation_glow = _normalize01(_prefer(stellar_radiation, bubbles, stars))
        field = _normalize01(0.24 * gas + 0.22 * knots + 0.34 * radiation_glow + 0.20 * stars)
        point_sources = _star_points(stars, percentile=98.6)
        return EraVisualScene(
            field=_contrast(field, gamma=0.48),
            visual_stage_id="first_stars",
            star_points=point_sources,
            warm_haze=_blur3(radiation_glow) * (0.25 + 0.65 * p),
            cool_haze=_blur3(gas) * 0.20,
            edge_layer=_ridge(radiation_glow) * 0.34,
            dimming_layer=np.clip(0.35 - 0.22 * radiation_glow, 0.0, 1.0),
        )

    if stage == "reionization":
        stars = _normalize01(_prefer(first_stars, ignition, collapse))
        radiation_glow = _normalize01(_prefer(stellar_radiation, stars))
        ionized = _normalize01(_prefer(ionization, bubbles, radiation_glow))
        bubble_layer = _normalize01(_prefer(bubbles, ionized, radiation_glow))
        neutral_gas = _normalize01(_prefer(cold_gas, baryon, dark))
        neutral_mask = np.clip(1.0 - 0.70 * ionized, 0.0, 1.0)
        field = _normalize01(
            0.25 * neutral_gas * neutral_mask
            + 0.38 * bubble_layer
            + 0.24 * radiation_glow
            + 0.13 * stars
        )
        fronts = _bubble_fronts(bubble_layer, ionized)
        return EraVisualScene(
            field=_contrast(field, gamma=0.46),
            visual_stage_id="reionization",
            star_points=_star_points(stars, percentile=98.4),
            warm_haze=_blur3(radiation_glow) * (0.24 + 0.48 * p),
            cool_haze=bubble_layer * (0.34 + 0.66 * p),
            edge_layer=fronts * (0.72 + 0.48 * p),
            dimming_layer=np.clip(neutral_mask * (0.38 - 0.18 * p), 0.0, 1.0),
        )

    fallback = _prefer(cmb, radiation, inflation, seed, dark)
    return EraVisualScene(
        field=_contrast(_normalize01(fallback), gamma=0.85),
        visual_stage_id=stage,
    )


def _apply_scene_overlays(rgb: np.ndarray, scene: EraVisualScene, progress: float) -> np.ndarray:
    p = _smoothstep(progress)
    out = np.asarray(rgb, dtype=np.float32).copy()

    if scene.dimming_layer is not None:
        dim = np.clip(scene.dimming_layer, 0.0, 1.0)[..., None]
        out *= np.clip(1.0 - 0.45 * dim, 0.45, 1.0)

    if scene.cool_haze is not None:
        out += np.clip(scene.cool_haze, 0.0, 1.0)[..., None] * np.array(
            [0.10, 0.30, 0.46], dtype=np.float32
        ) * (0.42 + 0.28 * p)

    if scene.warm_haze is not None:
        out += np.clip(scene.warm_haze, 0.0, 1.0)[..., None] * np.array(
            [0.72, 0.38, 0.12], dtype=np.float32
        ) * (0.32 + 0.36 * p)

    if scene.edge_layer is not None:
        out += np.clip(scene.edge_layer, 0.0, 1.0)[..., None] * np.array(
            [0.34, 0.62, 0.82], dtype=np.float32
        ) * (0.34 + 0.28 * p)

    if scene.star_points is not None:
        stars = np.clip(scene.star_points, 0.0, 1.0)
        halos = _blur3(stars)
        out += halos[..., None] * np.array([1.00, 0.70, 0.24], dtype=np.float32) * (0.28 + 0.34 * p)
        out += stars[..., None] * np.array([1.00, 0.96, 0.72], dtype=np.float32) * (0.82 + 0.42 * p)

    return out


def _field_view(field: np.ndarray) -> np.ndarray:
    data = np.asarray(field, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("universe visual fields must be 2D arrays")
    return data.T.astype(np.float32, copy=False)


def _prefer(*fields: np.ndarray) -> np.ndarray:
    for field in fields:
        data = np.asarray(field, dtype=np.float32)
        if data.size and float(np.nanstd(data)) > 1.0e-7:
            return data
    if not fields:
        raise ValueError("at least one field is required")
    return np.asarray(fields[0], dtype=np.float32)


def _normalize01(field: np.ndarray) -> np.ndarray:
    data = np.asarray(field, dtype=np.float32)
    if data.size == 0:
        return data
    min_value = float(np.nanmin(data))
    max_value = float(np.nanmax(data))
    span = max(max_value - min_value, 1.0e-8)
    return ((data - min_value) / span).astype(np.float32)


def _contrast(field: np.ndarray, *, gamma: float) -> np.ndarray:
    safe_gamma = max(float(gamma), 1.0e-4)
    return np.power(np.clip(field, 0.0, 1.0), safe_gamma).astype(np.float32)


def _blur3(field: np.ndarray) -> np.ndarray:
    data = np.asarray(field, dtype=np.float32)
    blurred = (
        data
        + np.roll(data, 1, axis=0)
        + np.roll(data, -1, axis=0)
        + np.roll(data, 1, axis=1)
        + np.roll(data, -1, axis=1)
        + np.roll(np.roll(data, 1, axis=0), 1, axis=1)
        + np.roll(np.roll(data, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(data, -1, axis=0), 1, axis=1)
        + np.roll(np.roll(data, -1, axis=0), -1, axis=1)
    ) / 9.0
    return blurred.astype(np.float32)


def _ridge(field: np.ndarray) -> np.ndarray:
    data = np.asarray(field, dtype=np.float32)
    smooth = _blur3(_blur3(data))
    return np.clip(data - 0.82 * smooth, 0.0, 1.0).astype(np.float32)


def _rolled_difference(field: np.ndarray) -> np.ndarray:
    data = np.asarray(field, dtype=np.float32)
    return (
        np.roll(data, 3, axis=0)
        - np.roll(data, -4, axis=1)
        + 0.5 * np.roll(data, 5, axis=0)
    ).astype(np.float32)


def _soft_peaks(field: np.ndarray, *, threshold: float, width: float) -> np.ndarray:
    data = np.clip(np.asarray(field, dtype=np.float32), 0.0, 1.0)
    safe_width = max(float(width), 1.0e-5)
    return np.clip((data - float(threshold)) / safe_width, 0.0, 1.0).astype(np.float32)


def _star_points(field: np.ndarray, *, percentile: float) -> np.ndarray:
    data = np.clip(np.asarray(field, dtype=np.float32), 0.0, 1.0)
    if data.size == 0 or float(np.nanstd(data)) <= 1.0e-7:
        return np.zeros_like(data, dtype=np.float32)
    threshold = float(np.percentile(data, percentile))
    peaks = np.where(data >= threshold, data, 0.0).astype(np.float32)
    return _contrast(_normalize01(peaks), gamma=0.34)


def _bubble_fronts(bubbles: np.ndarray, ionization: np.ndarray) -> np.ndarray:
    bubble = np.clip(np.asarray(bubbles, dtype=np.float32), 0.0, 1.0)
    ion = np.clip(np.asarray(ionization, dtype=np.float32), 0.0, 1.0)
    smooth = _blur3(bubble)
    gradient = np.abs(bubble - smooth)
    front = np.clip(gradient * 2.8 + _ridge(ion) * 0.9, 0.0, 1.0)
    return front.astype(np.float32)


def _grid_front(shape: tuple[int, int], progress: float) -> np.ndarray:
    height, width = shape[:2]
    y, x = np.indices((height, width), dtype=np.float32)
    spacing = 8.0 + 36.0 * _smoothstep(progress)
    phase = progress * spacing * 1.9
    vertical = np.exp(-((np.mod(x + phase, spacing) - spacing / 2.0) ** 2) / 10.0)
    horizontal = np.exp(-((np.mod(y - phase, spacing) - spacing / 2.0) ** 2) / 10.0)
    return (0.030 * (vertical + horizontal)).astype(np.float32)


def _smoothstep(value: float) -> float:
    clamped = float(np.clip(value, 0.0, 1.0))
    return clamped * clamped * (3.0 - 2.0 * clamped)
