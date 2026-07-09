from __future__ import annotations

from bbsim.core.config import ScaleConfig
from bbsim.core.scale import format_length, sample_scale_from_values


def test_format_length_uses_human_units() -> None:
    assert format_length(1.0e-9).endswith("нм")
    assert format_length(149_597_870_700.0).endswith("AU")
    assert format_length(3.0856775814913673e22).endswith("Mpc")


def test_sample_scale_uses_comoving_box_and_scale_factor() -> None:
    sample = sample_scale_from_values(
        a=1.0e-6,
        grid_size=100,
        scale=ScaleConfig(box_size_today_mpc=1000.0),
    )

    assert sample.box_today_mpc == 1000.0
    assert sample.box_now_m > 0.0
    assert sample.cell_now_m == sample.box_now_m / 100
    assert "kpc" in sample.box_now_text or "Mpc" in sample.box_now_text
