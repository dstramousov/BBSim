"""Main Qt window for the BBSim prototype."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from bbsim import __version__
from bbsim.core.app_config import AppConfig
from bbsim.core.config import UniverseConfig
from bbsim.core.context import UniverseRunContext, create_run_context
from bbsim.core.pipeline import UniversePipeline, create_default_pipeline
from bbsim.numeric.numpy_backend import NumpyBackend
from bbsim.render.field_renderer import field_to_display

_TIMELINE = (
    ("personal_seed", "Зерно"),
    ("inflation", "Инфляция"),
    ("recombination_preview", "CMB"),
)


def _create_seed_colormap() -> pg.ColorMap:
    """Create a calm cosmic colormap for primordial fields."""

    positions = np.array([0.0, 0.25, 0.50, 0.72, 1.0], dtype=float)
    colors = np.array(
        [
            [5, 8, 28, 255],
            [24, 26, 83, 255],
            [73, 106, 171, 255],
            [185, 216, 230, 255],
            [255, 239, 184, 255],
        ],
        dtype=np.ubyte,
    )
    return pg.ColorMap(positions, colors)


class MainWindow(QMainWindow):
    """Minimal prototype window for seed, timeline, field, and report views."""

    def __init__(self, app_config: AppConfig | None = None) -> None:
        super().__init__()
        self.setWindowTitle(f"BBSim v{__version__}")
        self._app_config = app_config or AppConfig()
        self._field_fill_canvas = self._app_config.view.field_fill_canvas
        self._context: UniverseRunContext | None = None
        self._pipeline: UniversePipeline | None = None

        self._phrase_edit = QLineEdit("Dimas")
        self._new_run_button = QPushButton("Создать запуск")
        self._next_button = QPushButton("Следующий checkpoint")
        self._stage_label = QLabel("Stage: —")
        self._timeline_label = QLabel(self._format_timeline(None))
        self._timeline_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._report = QTextEdit()
        self._report.setReadOnly(True)

        self._image = pg.ImageView()
        self._image.ui.roiBtn.hide()
        self._image.ui.menuBtn.hide()
        self._image.ui.histogram.hide()
        self._image.setColorMap(_create_seed_colormap())
        self._image.getView().setDefaultPadding(0.0)
        self._image.getView().setMouseEnabled(x=False, y=False)
        self._image.getView().setAspectLocked(not self._field_fill_canvas)

        self._plot = pg.PlotWidget(title="log10 a(t)")
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.setLabel("left", "log10 scale factor a")
        self._plot.setLabel("bottom", "sample")

        self._new_run_button.clicked.connect(self._create_run)
        self._next_button.clicked.connect(self._next_checkpoint)

        self.setCentralWidget(self._build_layout())
        self._create_run()

    def _build_layout(self) -> QWidget:
        root = QWidget()
        root_layout = QVBoxLayout(root)

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

        root_layout.addWidget(splitter, stretch=1)
        root_layout.addWidget(self._timeline_label)
        return root

    def _create_run(self) -> None:
        config = UniverseConfig.default(player_seed_phrase=self._phrase_edit.text())
        self._context = create_run_context(config=config, backend=NumpyBackend())
        self._pipeline = create_default_pipeline()
        self._report.clear()
        self._plot.clear()
        self._timeline_label.setText(self._format_timeline(None))
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
        self._timeline_label.setText(self._format_timeline(report.stage_id))
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
        elif stage_id == "inflation" and np.any(fields.inflation_delta):
            field = fields.inflation_delta
        else:
            field = fields.seed_delta
        display = field_to_display(field).T
        self._image.setImage(display, levels=(0.0, 1.0), autoRange=False)
        self._fit_field_to_canvas(display.shape)

    def _fit_field_to_canvas(self, image_shape: tuple[int, ...]) -> None:
        if len(image_shape) < 2:
            return
        width = int(image_shape[0])
        height = int(image_shape[1])
        view = self._image.getView()
        view.setAspectLocked(not self._field_fill_canvas)
        view.setRange(xRange=(0, width), yRange=(0, height), padding=0.0)

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
        positive_values = values[values > 0.0]
        if positive_values.size:
            self._plot.plot(
                np.arange(positive_values.size),
                np.log10(positive_values),
                pen=pg.mkPen(width=2),
            )

    @staticmethod
    def _format_timeline(active_stage_id: str | None) -> str:
        parts = []
        for stage_id, title in _TIMELINE:
            marker = "●" if stage_id == active_stage_id else "○"
            parts.append(f"{marker} {title}")
        return "   →   ".join(parts)
