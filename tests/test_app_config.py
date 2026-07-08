from __future__ import annotations

from pathlib import Path

from bbsim.core.app_config import AppConfig, load_app_config


def test_missing_app_config_uses_defaults(tmp_path: Path) -> None:
    config = load_app_config(tmp_path / "missing.toml")

    assert config == AppConfig()
    assert config.window.width == 1400
    assert config.window.height == 900
    assert config.window.mode == "normal"
    assert config.view.field_fill_canvas is True


def test_app_config_loads_window_and_view_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "app.toml"
    config_path.write_text(
        """
[window]
mode = "maximized"
width = 1600
height = 1000

[view]
field_fill_canvas = false
""".strip(),
        encoding="utf-8",
    )

    config = load_app_config(config_path)

    assert config.window.mode == "maximized"
    assert config.window.width == 1600
    assert config.window.height == 1000
    assert config.view.field_fill_canvas is False


def test_invalid_app_config_values_fall_back_to_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "app.toml"
    config_path.write_text(
        """
[window]
mode = "giant"
width = -1
height = "big"

[view]
field_fill_canvas = "yes"
""".strip(),
        encoding="utf-8",
    )

    config = load_app_config(config_path)

    assert config == AppConfig()
