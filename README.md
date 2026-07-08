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
```

- `normal` открывает окно заданного размера.
- `maximized` разворачивает окно на рабочую область десктопа.
- `fullscreen` включает полноэкранный режим.
- `field_fill_canvas = true` растягивает текущую карту поля на всю центральную область.

Можно указать альтернативный файл через переменную окружения:

```bash
BBSIM_APP_CONFIG=/path/to/app.toml bbsim
```

## Первый запуск UI

В UI пока есть минимальная лаборатория:

- поле фразы зерна;
- кнопка создания нового запуска;
- кнопка перехода к следующей контрольной точке;
- карта первичного поля / CMB-like поля;
- график `a(t)`;
- текстовый отчёт stage-а.

## Архитектурное решение

Один запуск хранится в `UniverseRunContext`. Это не singleton. `UniverseConfig` и `PersonalSeed` immutable, `UniverseState` и `UniverseFields` изменяются stage-ами pipeline.

Тяжёлая математика вызывается через `NumericBackend`. Сейчас есть `NumpyBackend`; позже можно добавить C++ backend через pybind11, не переписывая UI и pipeline.

## Команды проверки

```bash
python -m compileall src tests
pytest
ruff check .
```
