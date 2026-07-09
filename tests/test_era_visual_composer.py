from __future__ import annotations

import numpy as np

from bbsim.core.config import UniverseConfig
from bbsim.core.era_visual_composer import compose_era_visual_scene, render_era_visual_frame
from bbsim.core.fields import create_empty_fields


def _sample_fields(size: int = 24):
    fields = create_empty_fields(size)
    y, x = np.indices((size, size), dtype=np.float32)
    gradient = (x + y) / float(max(1, 2 * size - 2))
    ripple = np.sin(x * 0.7) + np.cos(y * 0.55) + gradient
    ripple = ripple.astype(np.float32)

    fields.seed_delta = ripple
    fields.inflation_delta = np.roll(ripple, 3, axis=0) * 0.7
    fields.radiation = np.sin(ripple * 2.1).astype(np.float32)
    fields.cmb = (0.6 * ripple + 0.4 * np.roll(ripple, 4, axis=1)).astype(np.float32)
    fields.dark_density = (1.0 + ripple).astype(np.float32)
    fields.baryon_density = (0.7 + 0.5 * np.roll(ripple, 2, axis=0)).astype(np.float32)
    fields.gravitational_potential = (ripple + np.roll(ripple, 1, axis=1)).astype(np.float32)
    fields.halo_density = np.maximum(ripple, np.percentile(ripple, 74)).astype(np.float32)
    fields.future_star_sites = np.maximum(ripple, np.percentile(ripple, 82)).astype(np.float32)
    fields.cold_gas_density = np.maximum(ripple, np.percentile(ripple, 62)).astype(np.float32)
    fields.molecular_cooling = np.maximum(ripple, np.percentile(ripple, 69)).astype(np.float32)
    fields.collapse_sites = np.maximum(ripple, np.percentile(ripple, 88)).astype(np.float32)
    fields.stellar_ignition = np.maximum(ripple, np.percentile(ripple, 94)).astype(np.float32)
    fields.first_star_density = np.maximum(ripple, np.percentile(ripple, 96)).astype(np.float32)
    fields.stellar_radiation = np.maximum(ripple, np.percentile(ripple, 86)).astype(np.float32)
    fields.ionized_bubbles = np.maximum(ripple, np.percentile(ripple, 70)).astype(np.float32)
    fields.ionization = np.maximum(ripple, np.percentile(ripple, 67)).astype(np.float32)
    return fields


def test_era_visual_composer_returns_rgb_for_all_current_epochs() -> None:
    fields = _sample_fields()
    config = UniverseConfig.default().visual_director

    for stage_id in (
        "personal_seed",
        "inflation",
        "reheating",
        "nucleosynthesis",
        "recombination",
        "dark_ages",
        "gas_collapse",
        "first_stars",
        "reionization",
    ):
        frame = render_era_visual_frame(fields, stage_id=stage_id, progress=0.7, config=config)

        assert frame.rgb.shape == (24, 24, 3)
        assert frame.rgb.dtype == np.float32
        assert float(frame.rgb.min()) >= 0.0
        assert float(frame.rgb.max()) <= 1.0


def test_era_visual_composer_uses_semantic_stage_profiles() -> None:
    fields = _sample_fields()

    first_stars = compose_era_visual_scene(fields, stage_id="first_stars", progress=0.8)
    reionization = compose_era_visual_scene(fields, stage_id="reionization", progress=0.8)
    dark_ages = compose_era_visual_scene(fields, stage_id="dark_ages", progress=0.8)

    assert first_stars.visual_stage_id == "first_stars"
    assert reionization.visual_stage_id == "reionization"
    assert dark_ages.visual_stage_id == "dark_ages"
    assert first_stars.star_points is not None
    assert reionization.edge_layer is not None
    assert dark_ages.dimming_layer is not None
    assert not np.allclose(first_stars.field, reionization.field)
    assert not np.allclose(dark_ages.field, first_stars.field)
