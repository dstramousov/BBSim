"""Cinematic time direction for epoch playback and time-scale readouts."""

from __future__ import annotations

from dataclasses import dataclass

from bbsim.core.config import UniverseConfig

_SECONDS_PER_YEAR = 31_557_600.0
_SECONDS_PER_GYR = _SECONDS_PER_YEAR * 1.0e9


@dataclass(frozen=True, slots=True)
class StagePhysicalRange:
    """Approximate physical time interval represented by one visual stage."""

    stage_id: str
    physical_start_s: float | None
    physical_end_s: float | None
    mapping: str


@dataclass(frozen=True, slots=True)
class TimeScaleSample:
    """Human-readable stage timing sample for UI annotations."""

    stage_id: str
    stage_screen_duration_s: float
    physical_time_s: float | None
    physical_start_s: float | None
    physical_end_s: float | None
    seconds_per_screen_second: float | None
    physical_time_text: str
    screen_duration_text: str
    time_scale_text: str
    mapping_text: str


_STAGE_RANGES: dict[str, StagePhysicalRange] = {
    # The seed is a user action before the modeled physical clock starts.
    "personal_seed": StagePhysicalRange("personal_seed", None, None, "подготовка начальных условий"),
    # These values are intentionally educational approximations, not precision cosmology.
    "inflation": StagePhysicalRange("inflation", 1.0e-36, 1.0e-32, "логарифмически растянуто"),
    "reheating": StagePhysicalRange("reheating", 1.0e-32, 1.0e-12, "логарифмически растянуто"),
    "nucleosynthesis": StagePhysicalRange("nucleosynthesis", 1.0, 20.0 * 60.0, "логарифмически растянуто"),
    "recombination": StagePhysicalRange(
        "recombination",
        20.0 * 60.0,
        380_000.0 * _SECONDS_PER_YEAR,
        "логарифмически сжато",
    ),
    "dark_ages": StagePhysicalRange(
        "dark_ages",
        380_000.0 * _SECONDS_PER_YEAR,
        180_000_000.0 * _SECONDS_PER_YEAR,
        "логарифмически сжато",
    ),
    "gas_collapse": StagePhysicalRange(
        "gas_collapse",
        180_000_000.0 * _SECONDS_PER_YEAR,
        260_000_000.0 * _SECONDS_PER_YEAR,
        "почти линейно сжато",
    ),
}


def stage_screen_duration_s(config: UniverseConfig, stage_id: str, fallback: float = 1.0) -> float:
    """Return configured screen duration for a stage in seconds."""

    director = config.time_director
    durations = {
        "personal_seed": director.personal_seed_visual_duration_s,
        "inflation": director.inflation_visual_duration_s,
        "reheating": director.reheating_visual_duration_s,
        "nucleosynthesis": director.nucleosynthesis_visual_duration_s,
        "recombination": director.recombination_visual_duration_s,
        "dark_ages": director.dark_ages_visual_duration_s,
        "gas_collapse": director.gas_collapse_visual_duration_s,
    }
    raw_duration = durations.get(stage_id, fallback)
    return max(float(raw_duration) * max(director.duration_scale, 0.01), 0.1)


def sample_time_scale(
    config: UniverseConfig,
    stage_id: str | None,
    local_stage_progress: float,
) -> TimeScaleSample | None:
    """Return a UI-friendly sample of the current stage time compression."""

    if stage_id is None:
        return None
    duration = stage_screen_duration_s(config, stage_id)
    stage_range = _STAGE_RANGES.get(stage_id)
    if stage_range is None:
        return TimeScaleSample(
            stage_id=stage_id,
            stage_screen_duration_s=duration,
            physical_time_s=None,
            physical_start_s=None,
            physical_end_s=None,
            seconds_per_screen_second=None,
            physical_time_text="физическое время не задано",
            screen_duration_text=format_screen_duration(duration),
            time_scale_text="шкала времени: игровая",
            mapping_text="визуальный этап",
        )

    screen_text = format_screen_duration(duration)
    if stage_range.physical_start_s is None or stage_range.physical_end_s is None:
        return TimeScaleSample(
            stage_id=stage_id,
            stage_screen_duration_s=duration,
            physical_time_s=None,
            physical_start_s=None,
            physical_end_s=None,
            seconds_per_screen_second=None,
            physical_time_text="физическое время ещё не запущено",
            screen_duration_text=screen_text,
            time_scale_text="1 сек экрана = подготовка зерна",
            mapping_text=stage_range.mapping,
        )

    progress = max(0.0, min(1.0, float(local_stage_progress)))
    physical_time = _log_lerp(stage_range.physical_start_s, stage_range.physical_end_s, progress)
    physical_span = max(stage_range.physical_end_s - stage_range.physical_start_s, 0.0)
    seconds_per_screen_second = physical_span / max(duration, 1.0e-9)
    return TimeScaleSample(
        stage_id=stage_id,
        stage_screen_duration_s=duration,
        physical_time_s=physical_time,
        physical_start_s=stage_range.physical_start_s,
        physical_end_s=stage_range.physical_end_s,
        seconds_per_screen_second=seconds_per_screen_second,
        physical_time_text=f"физическое время: {format_physical_duration(physical_time)}",
        screen_duration_text=screen_text,
        time_scale_text=f"1 сек экрана ≈ {format_physical_duration(seconds_per_screen_second)}",
        mapping_text=stage_range.mapping,
    )


def format_screen_duration(seconds: float) -> str:
    """Format screen playback duration."""

    return f"экранная длительность: {format_physical_duration(seconds)}"


def format_physical_duration(seconds: float) -> str:
    """Format a physical duration in compact Russian labels."""

    value = abs(float(seconds))
    if value == 0.0:
        return "0 с"
    if value < 1.0e-9:
        return f"{seconds:.1e} с"
    if value < 1.0e-6:
        return f"{seconds * 1.0e9:.2g} нс"
    if value < 1.0e-3:
        return f"{seconds * 1.0e6:.2g} мкс"
    if value < 1.0:
        return f"{seconds * 1.0e3:.2g} мс"
    if value < 60.0:
        return f"{seconds:.2g} с"
    if value < 3600.0:
        return f"{seconds / 60.0:.2g} мин"
    if value < 86_400.0:
        return f"{seconds / 3600.0:.2g} ч"
    years = seconds / _SECONDS_PER_YEAR
    if years < 1.0e3:
        return f"{years:.2g} лет"
    if years < 1.0e6:
        return f"{years / 1.0e3:.2g} тыс. лет"
    if years < 1.0e9:
        return f"{years / 1.0e6:.2g} млн лет"
    return f"{years / 1.0e9:.2g} млрд лет"


def _log_lerp(start: float, end: float, progress: float) -> float:
    safe_start = max(float(start), 1.0e-99)
    safe_end = max(float(end), safe_start)
    return safe_start * ((safe_end / safe_start) ** progress)
