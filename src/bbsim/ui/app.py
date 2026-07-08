"""Qt application bootstrap."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from bbsim.ui.main_window import MainWindow


def run_app() -> int:
    """Run the BBSim Qt application."""

    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1280, 820)
    window.show()
    return int(app.exec())
