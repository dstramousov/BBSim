from __future__ import annotations

import numpy as np

from bbsim.core.astro_visual_style import AstroVisualLayers, apply_astro_visual_style


def _sample_layers(size: int = 32) -> AstroVisualLayers:
    y, x = np.indices((size, size), dtype=np.float32)
    base = np.sin(x * 0.35) + np.cos(y * 0.27) + (x + y) / float(size * 2)
    base = base.astype(np.float32)
    normalized = (base - base.min()) / max(float(base.max() - base.min()), 1.0e-8)
    stars = np.where(normalized > 0.92, normalized, 0.0).astype(np.float32)
    return AstroVisualLayers(
        field=normalized,
        warm_haze=np.roll(normalized, 3, axis=0),
        cool_haze=np.roll(normalized, -4, axis=1),
        edge_layer=np.abs(normalized - np.roll(normalized, 1, axis=0)),
        dimming_layer=1.0 - normalized,
        star_points=stars,
    )


def test_astro_visual_style_keeps_rgb_bounds_and_shape() -> None:
    layers = _sample_layers()
    rgb = np.dstack([layers.field * 0.2, layers.field * 0.3, layers.field * 0.5]).astype(
        np.float32
    )

    styled = apply_astro_visual_style(
        rgb,
        layers,
        stage_id="reionization",
        progress=0.75,
        style_strength=1.0,
        bloom_strength=1.0,
        star_density=1.0,
    )

    assert styled.shape == rgb.shape
    assert styled.dtype == np.float32
    assert float(styled.min()) >= 0.0
    assert float(styled.max()) <= 1.0
    assert not np.allclose(styled, rgb)


def test_astro_visual_style_can_be_disabled() -> None:
    layers = _sample_layers()
    rgb = np.dstack([layers.field, layers.field * 0.5, layers.field * 0.25]).astype(np.float32)

    styled = apply_astro_visual_style(
        rgb,
        layers,
        stage_id="first_stars",
        progress=0.9,
        style_strength=0.0,
        bloom_strength=1.0,
        star_density=1.0,
    )

    assert np.allclose(styled, rgb)


def test_astro_visual_style_does_not_add_stars_before_first_light() -> None:
    layers = _sample_layers()
    rgb = np.zeros((*layers.field.shape, 3), dtype=np.float32)

    early = apply_astro_visual_style(
        rgb,
        layers,
        stage_id="dark_ages",
        progress=0.8,
        style_strength=1.0,
        bloom_strength=0.0,
        star_density=3.0,
    )
    first_light = apply_astro_visual_style(
        rgb,
        layers,
        stage_id="first_stars",
        progress=0.8,
        style_strength=1.0,
        bloom_strength=0.0,
        star_density=3.0,
    )

    assert float(first_light.max()) > float(early.max())
