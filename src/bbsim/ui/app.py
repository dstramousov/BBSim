"""Qt application bootstrap."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from bbsim.core.app_config import load_app_config
from bbsim.ui.main_window import MainWindow


def run_app() -> int:
    """Run the BBSim Qt application."""

    app = QApplication(sys.argv)
    app_config = load_app_config()
    window = MainWindow(app_config=app_config)

    if app_config.window.mode == "maximized":
        window.showMaximized()
    elif app_config.window.mode == "fullscreen":
        window.showFullScreen()
    else:
        window.resize(app_config.window.width, app_config.window.height)
        window.show()

    return int(app.exec())
