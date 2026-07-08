"""Main Qt window for the BBSim prototype."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from bbsim.core.config import UniverseConfig
from bbsim.core.context import UniverseRunContext, create_run_context
from bbsim.core.pipeline import UniversePipeline, create_default_pipeline
from bbsim.numeric.numpy_backend import NumpyBackend
from bbsim.render.field_renderer import field_to_display


class MainWindow(QMainWindow):
    """Minimal prototype window for seed, timeline, field, and report views."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BBSim v0.0.1")
        self._context: UniverseRunContext | None = None
        self._pipeline: UniversePipeline | None = None

        self._phrase_edit = QLineEdit("Dimas")
        self._new_run_button = QPushButton("Создать запуск")
        self._next_button = QPushButton("Следующий checkpoint")
        self._stage_label = QLabel("Stage: —")
        self._report = QTextEdit()
        self._report.setReadOnly(True)

        self._image = pg.ImageView()
        self._image.ui.roiBtn.hide()
        self._image.ui.menuBtn.hide()

        self._plot = pg.PlotWidget(title="a(t)")
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.setLabel("left", "scale factor a")
        self._plot.setLabel("bottom", "sample")

        self._new_run_button.clicked.connect(self._create_run)
        self._next_button.clicked.connect(self._next_checkpoint)

        self.setCentralWidget(self._build_layout())
        self._create_run()

    def _build_layout(self) -> QWidget:
        root = QWidget()
        layout = QHBoxLayout(root)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Имя / фраза зерна"))
        left_layout.addWidget(self._phrase_edit)
        left_layout.addWidget(self._new_run_button)
        left_layout.addWidget(self._next_button)
        left_layout.addWidget(self._stage_label)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self._plot, stretch=1)
        right_layout.addWidget(QLabel("Отчёт checkpoint-а"))
        right_layout.addWidget(self._report, stretch=2)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self._image)
        splitter.addWidget(right)
        splitter.setSizes([260, 680, 340])
        layout.addWidget(splitter)
        return root

    def _create_run(self) -> None:
        config = UniverseConfig.default(player_seed_phrase=self._phrase_edit.text())
        self._context = create_run_context(config=config, backend=NumpyBackend())
        self._pipeline = create_default_pipeline()
        self._report.clear()
        self._plot.clear()
        self._next_checkpoint()

    def _next_checkpoint(self) -> None:
        if self._context is None or self._pipeline is None:
            return
        if self._pipeline.is_finished:
            self._append_report_text("Pipeline завершён.")
            return

        self._pipeline.step_to_checkpoint(self._context)
        report = self._context.history.reports[-1]
        self._stage_label.setText(f"Stage: {report.stage_id}")
        self._show_current_field(report.stage_id)
        self._show_report(report)
        self._update_plot()
        self._pipeline.advance(self._context)

    def _show_current_field(self, stage_id: str) -> None:
        if self._context is None:
            return
        fields = self._context.fields
        if stage_id == "recombination_preview":
            field = fields.cmb
        elif stage_id == "inflation":
            field = fields.seed_delta
        else:
            field = fields.seed_delta
        display = field_to_display(field)
        self._image.setImage(display.T, autoLevels=True, autoRange=True)

    def _show_report(self, report) -> None:
        lines = [report.title, ""]
        lines.extend(f"• {line}" for line in report.summary_lines)
        self._report.setPlainText("\n".join(lines))

    def _append_report_text(self, text: str) -> None:
        self._report.setPlainText(self._report.toPlainText() + "\n" + text)

    def _update_plot(self) -> None:
        if self._context is None:
            return
        values = np.asarray(self._context.state.a_history, dtype=float)
        self._plot.clear()
        if values.size:
            self._plot.plot(np.arange(values.size), values, pen=pg.mkPen(width=2))
