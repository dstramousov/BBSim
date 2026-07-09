from __future__ import annotations

import numpy as np

from bbsim.core.config import UniverseConfig
from bbsim.core.visual_director import render_visual_frame


def test_visual_director_returns_rgb_frame() -> None:
    config = UniverseConfig.default().visual_director
    field = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)

    frame = render_visual_frame(field, stage_id="reheating", progress=0.5, config=config)

    assert frame.rgb.shape == (4, 4, 3)
    assert frame.rgb.dtype == np.float32
    assert float(frame.rgb.min()) >= 0.0
    assert float(frame.rgb.max()) <= 1.0


def test_visual_director_crossfades_start_of_new_epoch() -> None:
    config = UniverseConfig.default().visual_director
    field = np.linspace(0.0, 1.0, 25, dtype=np.float32).reshape(5, 5)

    early = render_visual_frame(field, stage_id="reheating", progress=0.01, config=config)
    late = render_visual_frame(field, stage_id="reheating", progress=0.50, config=config)

    assert early.transition_mix < late.transition_mix
    assert not np.allclose(early.rgb, late.rgb)


def test_visual_director_supports_dark_age_matter_profiles() -> None:
    config = UniverseConfig.default().visual_director
    field = np.linspace(-1.0, 1.0, 36, dtype=np.float32).reshape(6, 6)

    dark = render_visual_frame(field, stage_id="dark_matter", progress=0.7, config=config)
    gas = render_visual_frame(field, stage_id="baryon_gas", progress=0.7, config=config)
    mixed = render_visual_frame(field, stage_id="dark_ages", progress=0.7, config=config)
    potential = render_visual_frame(field, stage_id="gravitational_potential", progress=0.7, config=config)
    halos = render_visual_frame(field, stage_id="halo_candidates", progress=0.7, config=config)
    gas_collapse = render_visual_frame(field, stage_id="gas_collapse", progress=0.7, config=config)
    cold_gas = render_visual_frame(field, stage_id="cold_gas", progress=0.7, config=config)
    collapse_sites = render_visual_frame(field, stage_id="collapse_sites", progress=0.7, config=config)

    assert dark.profile_id == "dark_matter"
    assert gas.profile_id == "baryon_gas"
    assert mixed.profile_id == "dark_ages"
    assert potential.profile_id == "gravitational_potential"
    assert halos.profile_id == "halo_candidates"
    assert gas_collapse.profile_id == "gas_collapse"
    assert cold_gas.profile_id == "cold_gas"
    assert collapse_sites.profile_id == "collapse_sites"
    assert (
        dark.rgb.shape
        == gas.rgb.shape
        == mixed.rgb.shape
        == potential.rgb.shape
        == halos.rgb.shape
        == gas_collapse.rgb.shape
        == cold_gas.rgb.shape
        == collapse_sites.rgb.shape
    )
    assert not np.allclose(dark.rgb, gas.rgb)
    assert not np.allclose(potential.rgb, halos.rgb)
    assert not np.allclose(cold_gas.rgb, collapse_sites.rgb)
