"""Main Qt window for the BBSim prototype."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
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
    QStackedWidget,
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

_RUN_IDLE = "idle"
_RUN_RUNNING = "running"
_RUN_PAUSED = "paused"
_RUN_CHECKPOINT_PAUSE = "checkpoint_pause"
_RUN_FINISHED = "finished"


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
        self._run_state = _RUN_IDLE

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

        self._main_button = QPushButton("BIG BANG")
        self._main_button.setMinimumHeight(42)
        self._pause_on_epochs_check = QCheckBox("Останавливаться на эпохах")
        self._pause_on_epochs_check.setChecked(self._app_config.timeline.pause_on_epochs)
        self._stage_label = QLabel("Состояние: ожидание параметров")
        self._timeline_panel = TimelinePanel()
        self._report = QTextEdit()
        self._report.setReadOnly(True)
        self._pipeline_finished_reported = False

        self._field_placeholder = QLabel(
            "Вселенная ещё не создана\n\n"
            "Введите фразу зерна и параметры слева.\n"
            "Затем нажмите BIG BANG."
        )
        self._field_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._field_placeholder.setObjectName("fieldPlaceholder")
        self._field_placeholder.setStyleSheet(
            "QLabel#fieldPlaceholder {"
            "background-color: #050608;"
            "color: #bfc7d5;"
            "font-size: 18px;"
            "border: 1px solid #24262c;"
            "}"
        )

        self._image = pg.ImageView()
        self._image.ui.roiBtn.hide()
        self._image.ui.menuBtn.hide()
        self._image.ui.histogram.hide()
        self._image.setColorMap(_create_seed_colormap())
        self._image.getView().setDefaultPadding(0.0)
        self._image.getView().setMouseEnabled(x=False, y=False)
        self._image.getView().setAspectLocked(not self._field_fill_canvas)

        self._field_stack = QStackedWidget()
        self._field_stack.addWidget(self._field_placeholder)
        self._field_stack.addWidget(self._image)

        self._plot_placeholder = QLabel(
            "График появится после запуска BIG BANG."
        )
        self._plot_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._plot_placeholder.setObjectName("plotPlaceholder")
        self._plot_placeholder.setStyleSheet(
            "QLabel#plotPlaceholder {"
            "background-color: #050608;"
            "color: #9ba3b1;"
            "border: 1px solid #24262c;"
            "}"
        )
        self._scale_plot = pg.PlotWidget(title="log10 a(t)")
        self._scale_plot.showGrid(x=True, y=True, alpha=0.25)
        self._scale_plot.setLabel("left", "log10 a")
        self._scale_plot.setLabel("bottom", "sample")

        self._components_plot = pg.PlotWidget(title="Компоненты плотности")
        self._components_plot.showGrid(x=True, y=True, alpha=0.25)
        self._components_plot.setLabel("left", "fraction")
        self._components_plot.setLabel("bottom", "sample")
        self._components_plot.setYRange(0.0, 1.0)
        self._components_plot.addLegend(offset=(10, 10))

        self._recombination_plot = pg.PlotWidget(title="Рекомбинация")
        self._recombination_plot.showGrid(x=True, y=True, alpha=0.25)
        self._recombination_plot.setLabel("left", "fraction")
        self._recombination_plot.setLabel("bottom", "sample")
        self._recombination_plot.setYRange(0.0, 1.0)
        self._recombination_plot.addLegend(offset=(10, 10))

        self._plot_panel = QWidget()
        plot_panel_layout = QVBoxLayout(self._plot_panel)
        plot_panel_layout.setContentsMargins(0, 0, 0, 0)
        plot_panel_layout.addWidget(self._scale_plot, stretch=1)
        plot_panel_layout.addWidget(self._components_plot, stretch=1)
        plot_panel_layout.addWidget(self._recombination_plot, stretch=1)

        self._scale_curve = self._scale_plot.plot([], [], pen=pg.mkPen(width=2))
        self._radiation_curve = self._components_plot.plot(
            [], [], pen=pg.mkPen("#66d9ff", width=2), name="radiation"
        )
        self._matter_curve = self._components_plot.plot(
            [], [], pen=pg.mkPen("#f0c36d", width=2), name="matter"
        )
        self._dark_energy_curve = self._components_plot.plot(
            [], [], pen=pg.mkPen("#b48cff", width=2), name="dark energy"
        )
        self._curvature_curve = self._components_plot.plot(
            [], [], pen=pg.mkPen("#aaaaaa", width=1), name="curvature"
        )
        self._ionization_curve = self._recombination_plot.plot(
            [], [], pen=pg.mkPen("#8fd3ff", width=2), name="ionization"
        )
        self._opacity_curve = self._recombination_plot.plot(
            [], [], pen=pg.mkPen("#ffb86c", width=2), name="opacity"
        )

        self._plot_stack = QStackedWidget()
        self._plot_stack.addWidget(self._plot_placeholder)
        self._plot_stack.addWidget(self._plot_panel)

        self._parameter_widgets: tuple[QWidget, ...] = (
            self._phrase_edit,
            self._grid_size_spin,
            self._ripple_amp_spin,
            self._ripple_scale_spin,
            self._spectral_tilt_spin,
            self._inflation_strength_spin,
            self._inflation_duration_spin,
            self._inflation_smoothing_spin,
            self._h0_spin,
            self._omega_b_spin,
            self._omega_dm_spin,
            self._omega_lambda_spin,
            self._omega_r_spin,
            self._omega_k_spin,
        )

        self._timer = QTimer(self)
        self._timer.setInterval(self._app_config.timeline.tick_interval_ms)
        self._timer.timeout.connect(self._tick_simulation)

        self._main_button.clicked.connect(self._handle_main_button)

        self.setCentralWidget(self._build_layout())
        self._show_idle_state()

    def _build_layout(self) -> QWidget:
        root = QWidget()
        root_layout = QVBoxLayout(root)

        left = self._build_parameter_panel()

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self._plot_stack, stretch=1)
        right_layout.addWidget(QLabel("Отчёт эпохи"))
        right_layout.addWidget(self._report, stretch=2)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self._field_stack)
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
        layout.addWidget(self._pause_on_epochs_check)
        layout.addWidget(self._main_button)
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

    def _show_idle_state(self) -> None:
        self._timer.stop()
        self._context = None
        self._pipeline = None
        self._run_state = _RUN_IDLE
        self._pipeline_finished_reported = False
        self._set_parameter_inputs_enabled(True)
        self._field_stack.setCurrentWidget(self._field_placeholder)
        self._plot_stack.setCurrentWidget(self._plot_placeholder)
        self._clear_plot_data()
        self._timeline_panel.set_timeline_state(TimelineViewState())
        self._report.setPlainText(
            "Ожидание BIG BANG\n\n"
            "Заполните параметры слева.\n"
            "Кнопка BIG BANG зафиксирует параметры, создаст зерно "
            "и запустит живую эволюцию эпох."
        )
        self._stage_label.setText("Состояние: ожидание параметров")
        self._main_button.setText("BIG BANG")
        self._main_button.setEnabled(True)

    def _handle_main_button(self) -> None:
        if self._run_state in {_RUN_IDLE, _RUN_FINISHED}:
            self._start_big_bang()
        elif self._run_state == _RUN_RUNNING:
            self._pause_live_run()
        elif self._run_state in {_RUN_PAUSED, _RUN_CHECKPOINT_PAUSE}:
            self._resume_live_run()

    def _start_big_bang(self) -> None:
        config = self._build_config_from_ui()
        self._base_config = config
        self._context = create_run_context(config=config, backend=NumpyBackend())
        self._pipeline = create_default_pipeline()
        self._pipeline_finished_reported = False
        self._report.clear()
        self._clear_plot_data()
        self._field_stack.setCurrentWidget(self._image)
        self._plot_stack.setCurrentWidget(self._plot_panel)
        self._timeline_panel.set_timeline_state(TimelineViewState())
        self._set_parameter_inputs_enabled(False)
        self._run_state = _RUN_RUNNING
        self._main_button.setText("ПАУЗА")
        self._stage_label.setText("Состояние: эволюция запущена")
        self._timer.start()

    def _pause_live_run(self) -> None:
        self._timer.stop()
        self._run_state = _RUN_PAUSED
        self._main_button.setText("ПРОДОЛЖИТЬ")
        self._stage_label.setText("Состояние: пауза")

    def _resume_live_run(self) -> None:
        if self._context is None or self._pipeline is None or self._pipeline.is_finished:
            return
        self._run_state = _RUN_RUNNING
        self._main_button.setText("ПАУЗА")
        self._stage_label.setText("Состояние: эволюция идёт")
        self._timer.start()

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
            early_universe=self._base_config.early_universe,
            structure=self._base_config.structure,
        )

    def _tick_simulation(self) -> None:
        if self._context is None or self._pipeline is None:
            return
        if self._pipeline.is_finished:
            self._finish_live_run()
            return

        dt = self._timer.interval() / 1000.0
        report = self._pipeline.step_live(self._context, dt=dt)
        active_stage_id = self._context.state.current_stage
        self._show_current_field(active_stage_id)
        self._update_timeline(active_stage_id)
        self._update_plot()

        if report is None:
            self._stage_label.setText(f"Эпоха: {active_stage_id}")
            return

        self._show_report(report)
        self._pipeline.advance(self._context)
        self._update_timeline(report.stage_id)

        if self._pipeline.is_finished:
            self._finish_live_run()
            return

        if self._pause_on_epochs_check.isChecked():
            self._timer.stop()
            self._run_state = _RUN_CHECKPOINT_PAUSE
            self._main_button.setText("ПРОДОЛЖИТЬ")
            self._stage_label.setText(f"Checkpoint: {report.title}")
        else:
            self._stage_label.setText(f"Checkpoint пройден: {report.title}")

    def _finish_live_run(self) -> None:
        self._timer.stop()
        self._run_state = _RUN_FINISHED
        self._main_button.setText("НОВАЯ ВСЕЛЕННАЯ")
        self._set_parameter_inputs_enabled(True)
        if not self._pipeline_finished_reported:
            self._append_report_text("\nPipeline завершён.")
            self._pipeline_finished_reported = True
        self._stage_label.setText("Состояние: pipeline завершён")

    def _show_current_field(self, stage_id: str | None) -> None:
        if self._context is None:
            return
        fields = self._context.fields
        if stage_id == "recombination" and np.any(fields.cmb):
            field = fields.cmb
        elif stage_id in {"reheating", "nucleosynthesis"} and np.any(fields.radiation):
            field = fields.radiation
        elif stage_id == "inflation" and np.any(fields.inflation_delta):
            field = fields.inflation_delta
        elif np.any(fields.seed_delta):
            field = fields.seed_delta
        else:
            return
        display = field_to_display(field).T
        self._field_stack.setCurrentWidget(self._image)
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
            self._plot_stack.setCurrentWidget(self._plot_placeholder)
            return

        state = self._context.state
        scale_values = np.asarray(state.a_history, dtype=float)
        positive_values = scale_values[scale_values > 0.0]

        if not positive_values.size:
            self._clear_plot_data()
            self._plot_stack.setCurrentWidget(self._plot_placeholder)
            return

        self._plot_stack.setCurrentWidget(self._plot_panel)
        scale_x = np.arange(positive_values.size)
        self._scale_curve.setData(scale_x, np.log10(positive_values))

        radiation = np.asarray(state.radiation_fraction_history, dtype=float)
        matter = np.asarray(state.matter_fraction_history, dtype=float)
        dark_energy = np.asarray(state.dark_energy_fraction_history, dtype=float)
        curvature = np.asarray(state.curvature_fraction_history, dtype=float)
        sample_count = min(radiation.size, matter.size, dark_energy.size, curvature.size)
        if sample_count <= 0:
            self._radiation_curve.setData([], [])
            self._matter_curve.setData([], [])
            self._dark_energy_curve.setData([], [])
            self._curvature_curve.setData([], [])
            return

        x_values = np.arange(sample_count)
        self._radiation_curve.setData(x_values, radiation[-sample_count:])
        self._matter_curve.setData(x_values, matter[-sample_count:])
        self._dark_energy_curve.setData(x_values, dark_energy[-sample_count:])
        if np.max(curvature[-sample_count:]) > 0.02:
            self._curvature_curve.setData(x_values, curvature[-sample_count:])
        else:
            self._curvature_curve.setData([], [])

        ionization = np.asarray(state.ionization_fraction_history, dtype=float)
        opacity = np.asarray(state.opacity_history, dtype=float)
        recombination_count = min(ionization.size, opacity.size)
        if recombination_count <= 0:
            self._ionization_curve.setData([], [])
            self._opacity_curve.setData([], [])
            return

        recombination_x = np.arange(recombination_count)
        self._ionization_curve.setData(recombination_x, ionization[-recombination_count:])
        self._opacity_curve.setData(recombination_x, opacity[-recombination_count:])

    def _clear_plot_data(self) -> None:
        self._scale_curve.setData([], [])
        self._radiation_curve.setData([], [])
        self._matter_curve.setData([], [])
        self._dark_energy_curve.setData([], [])
        self._curvature_curve.setData([], [])
        self._ionization_curve.setData([], [])
        self._opacity_curve.setData([], [])

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
        progress = self._context.state.stage_progress
        if active_stage_id in completed_stage_ids:
            progress = 1.0
        self._timeline_panel.set_timeline_state(
            TimelineViewState(
                active_stage_id=active_stage_id,
                completed_stage_ids=completed_stage_ids,
                local_stage_progress=progress,
            )
        )

    def _set_parameter_inputs_enabled(self, enabled: bool) -> None:
        for widget in self._parameter_widgets:
            widget.setEnabled(enabled)
