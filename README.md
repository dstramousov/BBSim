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
# Legacy fallback; cinematic playback uses [time_director].
visual_duration_s = 8.0

[cosmology]
h0_gyr_inv = 0.069
omega_b = 0.049
omega_dm = 0.265
omega_lambda = 0.686
omega_r = 0.0001
omega_k = 0.0

[early_universe]
# Legacy fallbacks; cinematic playback uses [time_director].
reheating_visual_duration_s = 6.0
nucleosynthesis_visual_duration_s = 6.0
recombination_visual_duration_s = 7.0

[time_director]
mode = "cinematic" # quick / cinematic / deep
duration_scale = 1.0
personal_seed_visual_duration_s = 12.0
inflation_visual_duration_s = 38.0
reheating_visual_duration_s = 28.0
nucleosynthesis_visual_duration_s = 28.0
recombination_visual_duration_s = 38.0

[scale]
box_size_today_mpc = 1000.0
show_scale_overlay = true
```

В UI эти параметры можно менять перед нажатием `BIG BANG`. Во время одного запуска `UniverseConfig` остаётся immutable: изменение полей на панели влияет только на новый run.

`[time_director]` отвечает не за физику, а за режиссуру времени на экране: эпохи могут занимать сравнимое экранное время, хотя физически одна длится доли секунды, а другая сотни тысяч лет. Во время live-run справа показывается текущая шкала времени: например `1 сек экрана ≈ ... физического времени`. `duration_scale` умножает все экранные длительности и позволяет быстро получить quick/deep-поведение без изменения физики.

Во время live-эволюции правая панель показывает короткую аннотацию текущей эпохи: что происходит, что меняется визуально и какая шкала времени используется. Тексты локальные и короткие, чтобы не зависеть от сети и не превращать UI в статью.

`[scale]` задаёт человеческую линейку для отображаемого comoving-участка. Например `box_size_today_mpc = 1000.0` означает: весь центральный canvas — это кусок пространства, который при `a = 1` соответствовал бы 1000 Mpc. В ранних эпохах оверлей показывает физический размер этого же участка при текущем `a(t)`.

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
- оверлей масштаба на центральном canvas: `a(t)`, текущий физический размер видимого участка, размер клетки и сегодняшний Mpc-эквивалент;
- полупрозрачная сетка пространства поверх поля, которая во время инфляции визуально растягивается;
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

## v0.0.17 -> v0.0.18

- Добавлен `TimeDirector`: экранная длительность эпох теперь вынесена в конфиг и отделена от физического времени эпохи.
- Добавлена секция `[time_director]` в `config/simulation.toml`: режим, общий множитель длительности и длительность каждого раннего этапа.
- Live-аннотация справа теперь показывает короткое объяснение текущей эпохи, визуальный смысл происходящего и шкалу времени (`1 сек экрана ≈ ... физического времени`).
- Добавлены локальные аннотации эпох для зерна, инфляции, разогрева, нуклеосинтеза и рекомбинации.
- Центральная визуализация получила разные цветовые профили для ранних эпох: спокойное зерно/инфляция, горячий разогрев, охлаждение нуклеосинтеза и CMB-переход.
- Stage-длительности теперь читаются через `TimeDirector`, при этом старые `visual_duration_s` остаются legacy fallback-ами.
- Синхронизирована версия проекта до 0.0.18.
