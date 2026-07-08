"""Main Qt window for the BBSim prototype."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from bbsim import __version__
from bbsim.core.app_config import AppConfig
from bbsim.core.config import CosmologyConfig, InflationConfig, SeedConfig, UniverseConfig
from bbsim.core.context import UniverseRunContext, create_run_context
from bbsim.core.pipeline import UniversePipeline, create_default_pipeline
from bbsim.numeric.numpy_backend import NumpyBackend
from bbsim.render.field_renderer import field_to_display
from bbsim.ui.timeline_panel import TimelinePanel, TimelineViewState


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

    def __init__(
        self,
        app_config: AppConfig | None = None,
        simulation_config: UniverseConfig | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"BBSim v{__version__}")
        self._app_config = app_config or AppConfig()
        self._base_config = simulation_config or UniverseConfig.default()
        self._field_fill_canvas = self._app_config.view.field_fill_canvas
        self._context: UniverseRunContext | None = None
        self._pipeline: UniversePipeline | None = None

        self._phrase_edit = QLineEdit(self._base_config.seed.phrase)
        self._grid_size_spin = self._create_int_spin(16, 1024, self._base_config.seed.grid_size, 16)
        self._ripple_amp_spin = self._create_float_spin(
            0.0, 2.0, self._base_config.seed.fluctuation_amplitude, 0.01, 3
        )
        self._ripple_scale_spin = self._create_float_spin(
            0.05, 1.0, self._base_config.seed.fluctuation_scale, 0.01, 3
        )
        self._spectral_tilt_spin = self._create_float_spin(
            0.1, 2.0, self._base_config.seed.spectral_tilt, 0.01, 3
        )

        self._inflation_strength_spin = self._create_float_spin(
            0.0, 3.0, self._base_config.inflation.strength, 0.01, 3
        )
        self._inflation_duration_spin = self._create_float_spin(
            0.0, 3.0, self._base_config.inflation.duration, 0.01, 3
        )
        self._inflation_smoothing_spin = self._create_float_spin(
            0.0, 1.0, self._base_config.inflation.smoothing, 0.01, 3
        )

        self._h0_spin = self._create_float_spin(
            0.001, 1.0, self._base_config.cosmology.h0_gyr_inv, 0.001, 4
        )
        self._omega_b_spin = self._create_float_spin(
            0.0, 2.0, self._base_config.cosmology.omega_b, 0.001, 4
        )
        self._omega_dm_spin = self._create_float_spin(
            0.0, 2.0, self._base_config.cosmology.omega_dm, 0.001, 4
        )
        self._omega_lambda_spin = self._create_float_spin(
            -2.0, 3.0, self._base_config.cosmology.omega_lambda, 0.001, 4
        )
        self._omega_r_spin = self._create_float_spin(
            0.0, 1.0, self._base_config.cosmology.omega_r, 0.0001, 6
        )
        self._omega_k_spin = self._create_float_spin(
            -3.0, 3.0, self._base_config.cosmology.omega_k, 0.001, 4
        )

        self._new_run_button = QPushButton("Создать запуск")
        self._next_button = QPushButton("Следующий checkpoint")
        self._stage_label = QLabel("Stage: —")
        self._timeline_panel = TimelinePanel()
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

        left = self._build_parameter_panel()

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self._plot, stretch=1)
        right_layout.addWidget(QLabel("Отчёт checkpoint-а"))
        right_layout.addWidget(self._report, stretch=2)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self._image)
        splitter.addWidget(right)
        splitter.setSizes([330, 820, 360])

        root_layout.addWidget(splitter, stretch=1)
        root_layout.addWidget(self._timeline_panel)
        return root

    def _build_parameter_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        layout.addWidget(self._create_seed_group())
        layout.addWidget(self._create_inflation_group())
        layout.addWidget(self._create_cosmology_group())
        layout.addWidget(self._new_run_button)
        layout.addWidget(self._next_button)
        layout.addWidget(self._stage_label)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(300)
        return scroll

    def _create_seed_group(self) -> QGroupBox:
        group = QGroupBox("Seed / первичная рябь")
        form = QFormLayout(group)
        form.addRow("Фраза", self._phrase_edit)
        form.addRow("Размер сетки", self._grid_size_spin)
        form.addRow("Сила ряби", self._ripple_amp_spin)
        form.addRow("Масштаб ряби", self._ripple_scale_spin)
        form.addRow("Спектральный наклон", self._spectral_tilt_spin)
        return group

    def _create_inflation_group(self) -> QGroupBox:
        group = QGroupBox("Инфляция")
        form = QFormLayout(group)
        form.addRow("Сила", self._inflation_strength_spin)
        form.addRow("Длительность", self._inflation_duration_spin)
        form.addRow("Сглаживание", self._inflation_smoothing_spin)
        return group

    def _create_cosmology_group(self) -> QGroupBox:
        group = QGroupBox("Космология")
        form = QFormLayout(group)
        form.addRow("Начальный разлёт H0", self._h0_spin)
        form.addRow("Обычная материя Ωb", self._omega_b_spin)
        form.addRow("Тёмный каркас Ωdm", self._omega_dm_spin)
        form.addRow("Космический разгон ΩΛ", self._omega_lambda_spin)
        form.addRow("Горячее излучение Ωr", self._omega_r_spin)
        form.addRow("Кривизна Ωk", self._omega_k_spin)
        return group

    def _create_run(self) -> None:
        config = self._build_config_from_ui()
        self._base_config = config
        self._context = create_run_context(config=config, backend=NumpyBackend())
        self._pipeline = create_default_pipeline()
        self._report.clear()
        self._plot.clear()
        self._timeline_panel.set_timeline_state(TimelineViewState())
        self._next_checkpoint()

    def _build_config_from_ui(self) -> UniverseConfig:
        return UniverseConfig(
            seed=SeedConfig(
                phrase=self._phrase_edit.text(),
                grid_size=self._grid_size_spin.value(),
                fluctuation_amplitude=self._ripple_amp_spin.value(),
                fluctuation_scale=self._ripple_scale_spin.value(),
                spectral_tilt=self._spectral_tilt_spin.value(),
            ),
            inflation=InflationConfig(
                strength=self._inflation_strength_spin.value(),
                duration=self._inflation_duration_spin.value(),
                smoothing=self._inflation_smoothing_spin.value(),
                visual_duration_s=self._base_config.inflation.visual_duration_s,
            ),
            cosmology=CosmologyConfig(
                h0_gyr_inv=self._h0_spin.value(),
                omega_b=self._omega_b_spin.value(),
                omega_dm=self._omega_dm_spin.value(),
                omega_lambda=self._omega_lambda_spin.value(),
                omega_r=self._omega_r_spin.value(),
                omega_k=self._omega_k_spin.value(),
            ),
            structure=self._base_config.structure,
        )

    def _next_checkpoint(self) -> None:
        if self._context is None or self._pipeline is None:
            return
        if self._pipeline.is_finished:
            self._append_report_text("Pipeline завершён.")
            return

        self._pipeline.step_to_checkpoint(self._context)
        report = self._context.history.reports[-1]
        self._stage_label.setText(f"Stage: {report.stage_id}")
        self._update_timeline(report.stage_id)
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
    def _create_int_spin(minimum: int, maximum: int, value: int, step: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    @staticmethod
    def _create_float_spin(
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _update_timeline(self, active_stage_id: str | None) -> None:
        if self._context is None:
            self._timeline_panel.set_timeline_state(TimelineViewState())
            return

        completed_stage_ids = tuple(report.stage_id for report in self._context.history.reports)
        self._timeline_panel.set_timeline_state(
            TimelineViewState(
                active_stage_id=active_stage_id,
                completed_stage_ids=completed_stage_ids,
                local_stage_progress=self._context.state.stage_progress,
            )
        )
