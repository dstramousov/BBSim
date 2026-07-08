"""Application-level configuration loaded from TOML."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_VALID_WINDOW_MODES = {"normal", "maximized", "fullscreen"}


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Configuration for the main application window."""

    mode: str = "normal"
    width: int = 1400
    height: int = 900


@dataclass(frozen=True, slots=True)
class ViewConfig:
    """Configuration for field visualization widgets."""

    field_fill_canvas: bool = True


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Mutable-free application settings loaded before the Qt window is shown."""

    window: WindowConfig = WindowConfig()
    view: ViewConfig = ViewConfig()


def load_app_config(path: Path | str | None = None) -> AppConfig:
    """Load application settings from TOML, falling back to defaults.

    The default lookup order is:
    1. `BBSIM_APP_CONFIG` environment variable, if set;
    2. `config/app.toml` relative to the current working directory;
    3. built-in defaults.

    Args:
        path: Optional explicit path to a TOML config file.

    Returns:
        Application configuration with validated default fallbacks.
    """

    config_path = _resolve_config_path(path)
    if config_path is None or not config_path.exists():
        return AppConfig()

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    return _parse_app_config(raw)


def _resolve_config_path(path: Path | str | None) -> Path | None:
    if path is not None:
        return Path(path)

    env_path = os.environ.get("BBSIM_APP_CONFIG")
    if env_path:
        return Path(env_path)

    cwd_config = Path.cwd() / "config" / "app.toml"
    if cwd_config.exists():
        return cwd_config

    return None


def _parse_app_config(raw: dict[str, Any]) -> AppConfig:
    defaults = AppConfig()
    window = _parse_window_config(raw.get("window", {}), defaults.window)
    view = _parse_view_config(raw.get("view", {}), defaults.view)
    return AppConfig(window=window, view=view)


def _parse_window_config(raw: Any, defaults: WindowConfig) -> WindowConfig:
    if not isinstance(raw, dict):
        return defaults

    mode = raw.get("mode", defaults.mode)
    if not isinstance(mode, str) or mode not in _VALID_WINDOW_MODES:
        mode = defaults.mode

    width = _positive_int(raw.get("width"), defaults.width)
    height = _positive_int(raw.get("height"), defaults.height)
    return replace(defaults, mode=mode, width=width, height=height)


def _parse_view_config(raw: Any, defaults: ViewConfig) -> ViewConfig:
    if not isinstance(raw, dict):
        return defaults

    field_fill_canvas = raw.get("field_fill_canvas", defaults.field_fill_canvas)
    if not isinstance(field_fill_canvas, bool):
        field_fill_canvas = defaults.field_fill_canvas
    return replace(defaults, field_fill_canvas=field_fill_canvas)


def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int) and value > 0:
        return value
    return default
