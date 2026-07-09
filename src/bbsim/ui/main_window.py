"""Main Qt window for the BBSim prototype."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
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
from bbsim.core.epoch_notes import get_epoch_note
from bbsim.core.context import UniverseRunContext, create_run_context
from bbsim.core.pipeline import UniversePipeline, create_default_pipeline
from bbsim.numeric.numpy_backend import NumpyBackend
from bbsim.core.scale import build_scale_overlay_lines
from bbsim.core.time_director import sample_time_scale
from bbsim.core.visual_director import render_visual_frame
from bbsim.ui.space_overlay import SpaceScaleOverlay
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


def _create_hot_plasma_colormap() -> pg.ColorMap:
    """Create a hot plasma colormap for reheating."""

    positions = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=float)
    colors = np.array(
        [
            [10, 4, 20, 255],
            [75, 12, 60, 255],
            [180, 55, 32, 255],
            [255, 150, 54, 255],
            [255, 244, 190, 255],
        ],
        dtype=np.ubyte,
    )
    return pg.ColorMap(positions, colors)


def _create_cooling_colormap() -> pg.ColorMap:
    """Create a cooler plasma colormap for nucleosynthesis."""

    positions = np.array([0.0, 0.25, 0.55, 0.78, 1.0], dtype=float)
    colors = np.array(
        [
            [8, 10, 33, 255],
            [35, 37, 104, 255],
            [54, 127, 168, 255],
            [163, 220, 206, 255],
            [255, 232, 150, 255],
        ],
        dtype=np.ubyte,
    )
    return pg.ColorMap(positions, colors)


def _create_cmb_colormap() -> pg.ColorMap:
    """Create a CMB-like blue/yellow colormap."""

    positions = np.array([0.0, 0.28, 0.50, 0.72, 1.0], dtype=float)
    colors = np.array(
        [
            [9, 8, 38, 255],
            [32, 63, 128, 255],
            [135, 183, 210, 255],
            [243, 220, 138, 255],
            [255, 246, 204, 255],
        ],
        dtype=np.ubyte,
    )
    return pg.ColorMap(positions, colors)


def _smooth_visual_progress(value: float) -> float:
    """Return smooth visual 0..1 progress for UI-only fades."""

    clamped = float(np.clip(value, 0.0, 1.0))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def _seed_reveal_visibility(progress: float) -> float:
    """Return visible opacity for the personal-seed reveal.

    The first live frame must not look frozen-black. The numeric field can start
    close to zero, but the UI should immediately show a faint primordial pattern
    and then brighten it while the timeline moves to the first checkpoint.
    """

    return 0.18 + 0.82 * _smooth_visual_progress(progress)


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
        self._display_layer_combo = QComboBox()
        self._display_layer_combo.addItem("Авто по эпохе", "auto")
        self._display_layer_combo.addItem("CMB / реликтовый отпечаток", "cmb")
        self._display_layer_combo.addItem("Тёмная материя", "dark_density")
        self._display_layer_combo.addItem("Гравитационный каркас", "gravitational_potential")
        self._display_layer_combo.addItem("Будущие гало / звёздные узлы", "future_star_sites")
        self._display_layer_combo.addItem("Холодный газ", "cold_gas_density")
        self._display_layer_combo.addItem("Зоны газового коллапса", "collapse_sites")
        self._display_layer_combo.addItem("Первые звёзды", "first_star_density")
        self._display_layer_combo.addItem("Излучение первых звёзд", "stellar_radiation")
        self._display_layer_combo.addItem("Ионизированные пузыри", "ionized_bubbles")
        self._display_layer_combo.addItem("Ионизация / реионизация", "ionization")
        self._display_layer_combo.addItem("Обычная материя / газ", "baryon_density")
        self._display_layer_combo.addItem("Смешанный вид", "mixed_matter")
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
        # RGB frames are produced by VisualDirector. The legacy pyqtgraph
        # colormaps remain available above for future debugging, but runtime
        # rendering now uses smooth palette transitions across epoch borders.
        self._image.getView().setDefaultPadding(0.0)
        self._image.getView().setMouseEnabled(x=False, y=False)
        self._image.getView().setAspectLocked(not self._field_fill_canvas)

        self._space_overlay = SpaceScaleOverlay()
        self._image_container = QWidget()
        image_layout = QGridLayout(self._image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(0)
        image_layout.addWidget(self._image, 0, 0)
        image_layout.addWidget(self._space_overlay, 0, 0)
        self._space_overlay.raise_()

        self._field_stack = QStackedWidget()
        self._field_stack.addWidget(self._field_placeholder)
        self._field_stack.addWidget(self._image_container)

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

        self._recombination_plot = pg.PlotWidget(title="Ионизация / прозрачность")
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
            [], [], pen=pg.mkPen("#ffb86c", width=2), name="opacity / neutral"
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
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(self._app_config.timeline.tick_interval_ms)
        self._timer.timeout.connect(self._tick_simulation)

        self._main_button.clicked.connect(self._handle_main_button)
        self._display_layer_combo.currentIndexChanged.connect(lambda _index: self._redraw_current_field())

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
        layout.addWidget(self._create_display_group())
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

    def _create_display_group(self) -> QGroupBox:
        group = QGroupBox("Отображение")
        form = QFormLayout(group)
        form.addRow("Слой поля", self._display_layer_combo)
        return group

    def _show_idle_state(self) -> None:
        self._timer.stop()
        self._context = None
        self._pipeline = None
        self._run_state = _RUN_IDLE
        self._pipeline_finished_reported = False
        self._set_parameter_inputs_enabled(True)
        self._field_stack.setCurrentWidget(self._field_placeholder)
        self._space_overlay.set_overlay_state(
            lines=(), stage_id=None, stage_progress=0.0, visible=False
        )
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
        self._field_stack.setCurrentWidget(self._image_container)
        self._plot_stack.setCurrentWidget(self._plot_panel)
        self._timeline_panel.set_timeline_state(TimelineViewState())
        self._set_parameter_inputs_enabled(False)
        self._run_state = _RUN_RUNNING
        self._main_button.setText("ПАУЗА")
        self._stage_label.setText("Состояние: эволюция запущена")
        # Start the heartbeat first and draw a live frame immediately. This makes
        # BIG BANG visibly enter the live loop instead of showing a dark canvas
        # until the next event-cycle tick.
        self._timer.start()
        self._tick_simulation()
        if self._run_state != _RUN_RUNNING:
            self._timer.stop()

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
            time_director=self._base_config.time_director,
            scale=self._base_config.scale,
            visual_director=self._base_config.visual_director,
            structure=self._base_config.structure,
        )

    def _tick_simulation(self) -> None:
        if self._context is None or self._pipeline is None:
            return
        if self._pipeline.is_finished:
            self._finish_live_run()
            return

        dt = max(self._timer.interval() / 1000.0, 1.0e-3)
        report = self._pipeline.step_live(self._context, dt=dt)
        active_stage_id = self._context.state.current_stage
        self._show_current_field(active_stage_id)
        self._update_timeline(active_stage_id)
        self._update_plot()

        if report is None:
            self._stage_label.setText(self._live_stage_status(active_stage_id))
            self._show_live_status(active_stage_id)
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


    def _live_stage_status(self, stage_id: str | None) -> str:
        note = get_epoch_note(stage_id)
        return f"Эпоха: {note.title} — {note.visual_hint}"

    def _show_live_status(self, stage_id: str | None) -> None:
        if self._context is None:
            return
        note = get_epoch_note(stage_id)
        time_sample = sample_time_scale(
            self._context.config,
            stage_id,
            self._context.state.stage_progress,
        )
        lines = [note.title, "", note.summary, "", "Что происходит:"]
        lines.extend(f"• {bullet}" for bullet in note.bullets)
        lines.append("")
        lines.append(f"Визуально: {note.visual_hint}.")
        if stage_id == "dark_ages":
            lines.extend(
                (
                    "",
                    "Структура:",
                    f"• контраст тёмного каркаса: {self._context.state.dark_matter_contrast:.2f}",
                    f"• контраст газа: {self._context.state.baryon_contrast:.2f}",
                    f"• задержка газа: {self._context.state.gas_lag:.2f}",
                    f"• кандидатов будущих гало: {self._context.state.halo_count}",
                    f"• будущих звёздных узлов: {self._context.state.future_star_site_count}",
                )
            )
        if stage_id == "gas_collapse":
            lines.extend(
                (
                    "",
                    "Газ:",
                    f"• доля охлаждённого газа: {self._context.state.gas_cooling_fraction:.1%}",
                    f"• доля газа в коллапсе: {self._context.state.collapsed_gas_fraction:.1%}",
                    f"• облаков коллапса: {self._context.state.collapse_site_count}",
                    f"• температура холодного газа: {self._context.state.gas_temperature_k:.0f} K",
                    f"• готовность к первым звёздам: {self._context.state.star_formation_readiness:.2f}",
                )
            )
        if stage_id == "first_stars":
            lines.extend(
                (
                    "",
                    "Первый свет:",
                    f"• очагов первых звёзд: {self._context.state.first_star_count}",
                    f"• доля газа в звездообразовании: {self._context.state.star_formation_fraction:.1%}",
                    f"• интенсивность излучения: {self._context.state.stellar_radiation_intensity:.2f}",
                    f"• ионизированных пузырей: {self._context.state.ionized_bubble_fraction:.1%}",
                    f"• прогресс к реионизации: {self._context.state.reionization_progress:.2f}",
                )
            )
        if stage_id == "reionization":
            lines.extend(
                (
                    "",
                    "Реионизация:",
                    f"• ионизированная доля: {self._context.state.ionized_fraction:.1%}",
                    f"• нейтральная доля: {self._context.state.neutral_fraction:.1%}",
                    f"• перекрытие пузырей: {self._context.state.bubble_overlap_fraction:.1%}",
                    f"• photoheating feedback: {self._context.state.photoheating_feedback:.2f}",
                    f"• температура нагретого газа: {self._context.state.gas_temperature_k:.0f} K",
                )
            )
        if time_sample is not None:
            lines.extend(
                (
                    "",
                    "Шкала времени:",
                    f"• {time_sample.physical_time_text}",
                    f"• {time_sample.screen_duration_text}",
                    f"• {time_sample.time_scale_text}",
                    f"• отображение: {time_sample.mapping_text}",
                )
            )
        self._report.setPlainText("\n".join(lines))

    def _select_display_field(self, stage_id: str | None) -> tuple[np.ndarray, str | None] | None:
        if self._context is None:
            return None
        fields = self._context.fields
        layer = str(self._display_layer_combo.currentData() or "auto")

        if layer == "cmb" and self._has_field_signal(fields.cmb):
            return fields.cmb, "recombination"
        if layer == "dark_density" and self._has_field_signal(fields.dark_density):
            return fields.dark_density, "dark_matter"
        if layer == "gravitational_potential" and self._has_field_signal(fields.gravitational_potential):
            return fields.gravitational_potential, "gravitational_potential"
        if layer == "future_star_sites" and self._has_field_signal(fields.future_star_sites):
            return fields.future_star_sites, "halo_candidates"
        if layer == "future_star_sites" and self._has_field_signal(fields.halo_density):
            return fields.halo_density, "halo_candidates"
        if layer == "cold_gas_density" and self._has_field_signal(fields.cold_gas_density):
            return fields.cold_gas_density, "cold_gas"
        if layer == "collapse_sites" and self._has_field_signal(fields.collapse_sites):
            return fields.collapse_sites, "collapse_sites"
        if layer == "first_star_density" and self._has_field_signal(fields.first_star_density):
            return fields.first_star_density, "first_stars"
        if layer == "stellar_radiation" and self._has_field_signal(fields.stellar_radiation):
            return fields.stellar_radiation, "stellar_radiation"
        if layer == "ionized_bubbles" and self._has_field_signal(fields.ionized_bubbles):
            return fields.ionized_bubbles, "ionized_bubbles"
        if layer == "ionization" and self._has_field_signal(fields.ionization):
            return fields.ionization, "reionization"
        if layer == "baryon_density" and self._has_field_signal(fields.baryon_density):
            return fields.baryon_density, "baryon_gas"
        if layer == "mixed_matter":
            mixed = self._mixed_matter_field()
            if mixed is not None:
                return mixed, "mixed_matter"

        if stage_id == "reionization":
            if self._has_field_signal(fields.ionization):
                return self._reionization_field(), "reionization"
            if self._has_field_signal(fields.ionized_bubbles):
                return fields.ionized_bubbles, "reionization"
        if stage_id == "first_stars":
            if self._has_field_signal(fields.first_star_density):
                return self._first_stars_field(), "first_stars"
            if self._has_field_signal(fields.stellar_ignition):
                return fields.stellar_ignition, "first_stars"
            if self._has_field_signal(fields.collapse_sites):
                return fields.collapse_sites, "first_stars"
        if stage_id == "gas_collapse":
            if self._has_field_signal(fields.collapse_sites):
                return self._gas_collapse_field(), "gas_collapse"
            if self._has_field_signal(fields.cold_gas_density):
                return fields.cold_gas_density, "gas_collapse"
            if self._has_field_signal(fields.baryon_density):
                return fields.baryon_density, "gas_collapse"
        if stage_id == "dark_ages":
            mixed = self._mixed_matter_field()
            if mixed is not None:
                return mixed, "dark_ages"
            if self._has_field_signal(fields.halo_density):
                return fields.halo_density, "dark_ages"
            if self._has_field_signal(fields.dark_density):
                return fields.dark_density, "dark_ages"
        if stage_id == "recombination" and self._has_field_signal(fields.cmb):
            return fields.cmb, stage_id
        if stage_id in {"reheating", "nucleosynthesis"} and self._has_field_signal(fields.radiation):
            return fields.radiation, stage_id
        if stage_id == "inflation" and self._has_field_signal(fields.inflation_delta):
            return fields.inflation_delta, stage_id
        if self._has_field_signal(fields.seed_delta):
            return fields.seed_delta, stage_id
        return None

    def _reionization_field(self) -> np.ndarray:
        if self._context is None:
            raise RuntimeError("reionization field requested without context")
        fields = self._context.fields
        base = fields.ionization
        ionization = (
            self._normalize_for_display(fields.ionization)
            if self._has_field_signal(fields.ionization)
            else np.zeros_like(base)
        )
        bubbles = (
            self._normalize_for_display(fields.ionized_bubbles)
            if self._has_field_signal(fields.ionized_bubbles)
            else np.zeros_like(base)
        )
        radiation = (
            self._normalize_for_display(fields.stellar_radiation)
            if self._has_field_signal(fields.stellar_radiation)
            else np.zeros_like(base)
        )
        stars = (
            self._normalize_for_display(fields.first_star_density)
            if self._has_field_signal(fields.first_star_density)
            else np.zeros_like(base)
        )
        cold = (
            self._normalize_for_display(fields.cold_gas_density)
            if self._has_field_signal(fields.cold_gas_density)
            else np.zeros_like(base)
        )
        neutral_gaps = np.clip(1.0 - cold, 0.0, 1.0)
        return (
            0.42 * ionization
            + 0.22 * bubbles
            + 0.18 * radiation
            + 0.10 * stars
            + 0.08 * neutral_gaps
        ).astype(np.float32)

    def _first_stars_field(self) -> np.ndarray:
        if self._context is None:
            raise RuntimeError("first stars field requested without context")
        fields = self._context.fields
        base = fields.first_star_density
        cold = self._normalize_for_display(fields.cold_gas_density) if self._has_field_signal(fields.cold_gas_density) else np.zeros_like(base)
        stars = self._normalize_for_display(fields.first_star_density) if self._has_field_signal(fields.first_star_density) else np.zeros_like(base)
        radiation = self._normalize_for_display(fields.stellar_radiation) if self._has_field_signal(fields.stellar_radiation) else np.zeros_like(base)
        bubbles = self._normalize_for_display(fields.ionized_bubbles) if self._has_field_signal(fields.ionized_bubbles) else np.zeros_like(base)
        collapse = self._normalize_for_display(fields.collapse_sites) if self._has_field_signal(fields.collapse_sites) else np.zeros_like(base)
        return (0.20 * cold + 0.18 * collapse + 0.38 * stars + 0.16 * radiation + 0.08 * bubbles).astype(np.float32)

    def _gas_collapse_field(self) -> np.ndarray:
        if self._context is None:
            raise RuntimeError("gas collapse field requested without context")
        fields = self._context.fields
        baryon = self._normalize_for_display(fields.baryon_density) if self._has_field_signal(fields.baryon_density) else np.zeros_like(fields.collapse_sites)
        cold = self._normalize_for_display(fields.cold_gas_density) if self._has_field_signal(fields.cold_gas_density) else np.zeros_like(baryon)
        collapse = self._normalize_for_display(fields.collapse_sites) if self._has_field_signal(fields.collapse_sites) else np.zeros_like(baryon)
        halo = self._normalize_for_display(fields.halo_density) if self._has_field_signal(fields.halo_density) else np.zeros_like(baryon)
        return (0.30 * baryon + 0.30 * cold + 0.28 * collapse + 0.12 * halo).astype(np.float32)

    def _mixed_matter_field(self) -> np.ndarray | None:
        if self._context is None:
            return None
        fields = self._context.fields
        if not self._has_field_signal(fields.dark_density):
            return None
        dark = self._normalize_for_display(fields.dark_density)
        if self._has_field_signal(fields.baryon_density):
            baryon = self._normalize_for_display(fields.baryon_density)
        else:
            baryon = np.zeros_like(dark)
        if self._has_field_signal(fields.halo_density):
            halo = self._normalize_for_display(fields.halo_density)
        else:
            halo = np.zeros_like(dark)
        return (0.58 * dark + 0.28 * baryon + 0.14 * halo).astype(np.float32)

    @staticmethod
    def _has_field_signal(field: np.ndarray) -> bool:
        data = np.asarray(field)
        return bool(data.size and float(np.nanstd(data)) > 1.0e-7)

    @staticmethod
    def _normalize_for_display(field: np.ndarray) -> np.ndarray:
        data = np.asarray(field, dtype=np.float32)
        mean = float(np.nanmean(data))
        std = float(np.nanstd(data))
        if std <= 1.0e-8:
            return np.zeros_like(data, dtype=np.float32)
        return ((data - mean) / std).astype(np.float32)

    def _redraw_current_field(self) -> None:
        if self._context is None:
            return
        self._show_current_field(self._context.state.current_stage)

    def _show_current_field(self, stage_id: str | None) -> None:
        if self._context is None:
            return
        selected = self._select_display_field(stage_id)
        if selected is None:
            return
        field, visual_stage_id = selected
        frame = render_visual_frame(
            field.T,
            stage_id=visual_stage_id,
            progress=self._context.state.stage_progress,
            config=self._context.config.visual_director,
        )
        self._field_stack.setCurrentWidget(self._image_container)
        self._image.setImage(frame.rgb, autoRange=False, autoLevels=False)
        self._fit_field_to_canvas(frame.rgb.shape)
        self._update_scale_overlay(stage_id)

    def _fit_field_to_canvas(self, image_shape: tuple[int, ...]) -> None:
        if len(image_shape) < 2:
            return
        width = int(image_shape[0])
        height = int(image_shape[1])
        view = self._image.getView()
        view.setAspectLocked(not self._field_fill_canvas)
        view.setRange(xRange=(0, width), yRange=(0, height), padding=0.0)

    def _update_scale_overlay(self, stage_id: str | None) -> None:
        if self._context is None:
            self._space_overlay.set_overlay_state(
                lines=(), stage_id=None, stage_progress=0.0, visible=False
            )
            return

        visible = self._context.config.scale.show_scale_overlay
        lines = build_scale_overlay_lines(self._context.state, self._context.config) if visible else ()
        self._space_overlay.set_overlay_state(
            lines=lines,
            stage_id=stage_id,
            stage_progress=self._context.state.stage_progress,
            visible=visible,
        )

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
