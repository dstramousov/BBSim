from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from bbsim.core.config import UniverseConfig
from bbsim.core.epoch_gallery import render_epoch_gallery, upscale_rgb


def test_epoch_gallery_writes_pngs_manifest_and_markdown(tmp_path) -> None:
    config = UniverseConfig.default(player_seed_phrase="Gallery Seed")
    config = replace(config, seed=replace(config.seed, grid_size=32))

    result = render_epoch_gallery(config=config, output_dir=tmp_path / "gallery", image_scale=2)

    assert len(result.entries) == 9
    assert result.manifest_path.exists()
    assert result.markdown_path.exists()
    assert [entry.stage_id for entry in result.entries] == [
        "personal_seed",
        "inflation",
        "reheating",
        "nucleosynthesis",
        "recombination",
        "dark_ages",
        "gas_collapse",
        "first_stars",
        "reionization",
    ]

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["phrase"] == "Gallery Seed"
    assert manifest["grid_size"] == 32
    assert manifest["image_scale"] == 2
    assert len(manifest["entries"]) == 9

    for entry in result.entries:
        data = entry.png_path.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert entry.png_path.name in result.markdown_path.read_text(encoding="utf-8")


def test_upscale_rgb_keeps_values_in_bounds() -> None:
    rgb = np.array(
        [
            [[0.0, 0.2, 0.4], [1.0, 0.8, 0.6]],
            [[0.3, 0.5, 0.7], [0.9, 1.0, 0.1]],
        ],
        dtype=np.float32,
    )

    scaled = upscale_rgb(rgb, 3)

    assert scaled.shape == (6, 6, 3)
    assert scaled.dtype == np.float32
    assert float(scaled.min()) >= 0.0
    assert float(scaled.max()) <= 1.0


def test_epoch_gallery_rejects_invalid_scale(tmp_path) -> None:
    with pytest.raises(ValueError, match="image_scale"):
        render_epoch_gallery(
            config=UniverseConfig.default(),
            output_dir=tmp_path / "gallery",
            image_scale=0,
        )
