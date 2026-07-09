# BBSim

BBSim — прототип 2D-симулятора эволюции Вселенной: личное первичное зерно, инфляция, расширение через `a(t)`, контрольные точки эпох и визуализация 2D-полей.

Это стартовый костяк репозитория. Здесь ещё нет полной игровой модели, но уже есть правильные архитектурные границы:

- `core` — конфиги, состояние, поля, контекст запуска, pipeline, checkpoint-и;
- `numeric` — заменяемый backend для тяжёлой математики, сейчас NumPy;
- `stages` — этапы pipeline;
- `ui` — Qt/PySide6 окно;
- `render` — преобразование полей в данные для отображения;
- `tests` — базовые тесты детерминизма и расчётов.

## Стек

- Python 3.12+
- NumPy
- PySide6 / Qt
- PyQtGraph
- pytest
- ruff

## Быстрый старт

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
bbsim --headless --phrase "Dimas"
bbsim
```

Если `python3.12` не установлен, используй доступный Python 3.12+.


## Конфиг приложения

UI-настройки читаются из `config/app.toml` при запуске из корня проекта. Без файла используются встроенные значения по умолчанию.

```toml
[window]
mode = "normal"      # normal / maximized / fullscreen
width = 1400
height = 900

[view]
field_fill_canvas = true

[timeline]
pause_on_epochs = true
tick_interval_ms = 33
```

- `normal` открывает окно заданного размера.
- `maximized` разворачивает окно на рабочую область десктопа.
- `fullscreen` включает полноэкранный режим.
- `field_fill_canvas = true` растягивает текущую карту поля на всю центральную область.
- `pause_on_epochs = true` останавливает live-эволюцию на checkpoint-ах эпох.
- `tick_interval_ms` задаёт частоту обновления UI live-симуляции.

Можно указать альтернативный файл через переменную окружения:

```bash
BBSIM_APP_CONFIG=/path/to/app.toml bbsim
```


## Конфиг симуляции

Параметры одного запуска читаются из `config/simulation.toml`. Без файла используются встроенные значения по умолчанию. Эти значения также подставляются в левую UI-панель при старте.

```toml
[seed]
phrase = "Dimas"
grid_size = 192
fluctuation_amplitude = 0.35
fluctuation_scale = 0.50
spectral_tilt = 0.965

[inflation]
strength = 0.72
duration = 0.68
smoothing = 0.85
visual_duration_s = 40.0

[cosmology]
h0_gyr_inv = 0.069
omega_b = 0.049
omega_dm = 0.265
omega_lambda = 0.686
omega_r = 0.0001
omega_k = 0.0

[early_universe]
reheating_visual_duration_s = 6.0
nucleosynthesis_visual_duration_s = 6.0
recombination_visual_duration_s = 7.0
```

В UI эти параметры можно менять перед нажатием `BIG BANG`. Во время одного запуска `UniverseConfig` остаётся immutable: изменение полей на панели влияет только на новый run.

Можно указать альтернативный файл через переменную окружения или CLI:

```bash
BBSIM_SIMULATION_CONFIG=/path/to/simulation.toml bbsim
python -m bbsim --headless --simulation-config /path/to/simulation.toml
python -m bbsim --headless --phrase "Other Seed"
```

## Первый запуск UI

В UI пока есть минимальная лаборатория:

- поле фразы зерна;
- одна главная кнопка `BIG BANG` / `ПАУЗА` / `ПРОДОЛЖИТЬ` / `НОВАЯ ВСЕЛЕННАЯ`;
- галочка `Останавливаться на эпохах`;
- live-карта первичного поля, инфляции, разогрева, нуклеосинтеза и рекомбинации/CMB;
- графики `a(t)`, долей radiation / matter / dark energy и прозрачности рекомбинации, обновляющиеся во время live-эволюции;
- текстовый отчёт checkpoint-а;
- полноширинный нижний timeline с визуальным бегущим progress-bar эпох.

## Архитектурное решение

Один запуск хранится в `UniverseRunContext`. Это не singleton. `UniverseConfig` и `PersonalSeed` immutable, `UniverseState` и `UniverseFields` изменяются stage-ами pipeline.

Тяжёлая математика вызывается через `NumericBackend`. Сейчас есть `NumpyBackend`; позже можно добавить C++ backend через pybind11, не переписывая UI и pipeline.

## Команды проверки

```bash
python -m compileall src tests
pytest
ruff check .
```
