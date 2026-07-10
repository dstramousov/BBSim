"""Headless epoch gallery rendering for visual regression review."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from bbsim.core.config import UniverseConfig
from bbsim.core.context import create_run_context
from bbsim.core.era_visual_composer import render_era_visual_frame
from bbsim.core.pipeline import create_default_pipeline
from bbsim.core.report import StageReport
from bbsim.numeric.numpy_backend import NumpyBackend
from bbsim.render.png_writer import save_rgb_png


@dataclass(frozen=True, slots=True)
class EpochGalleryEntry:
    """Metadata for one saved epoch gallery frame.

    Attributes:
        index: One-based epoch index in pipeline order.
        stage_id: Stable stage identifier.
        title: Human-readable checkpoint title.
        png_path: Path to the rendered PNG file.
        progress: Local stage progress used for the render.
        metrics: Numeric metrics included in the checkpoint report.
    """

    index: int
    stage_id: str
    title: str
    png_path: Path
    progress: float
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class EpochGalleryResult:
    """Result of a gallery rendering run.

    Attributes:
        output_dir: Directory containing gallery files.
        entries: Saved frame metadata in pipeline order.
        manifest_path: JSON manifest path.
        markdown_path: Human-readable Markdown index path.
    """

    output_dir: Path
    entries: tuple[EpochGalleryEntry, ...]
    manifest_path: Path
    markdown_path: Path


def render_epoch_gallery(
    *,
    config: UniverseConfig,
    output_dir: Path,
    image_scale: int = 4,
) -> EpochGalleryResult:
    """Run the pipeline and save one styled PNG snapshot for each epoch.

    Args:
        config: Immutable simulation configuration.
        output_dir: Destination directory for PNG frames and indexes.
        image_scale: Positive integer output multiplier for easier visual inspection.

    Returns:
        Metadata describing the saved gallery.

    Raises:
        ValueError: If ``image_scale`` is outside the supported range.
        OSError: If gallery files cannot be written.
    """

    if image_scale < 1 or image_scale > 16:
        raise ValueError("image_scale must be in the range [1, 16]")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    context = create_run_context(config=config, backend=NumpyBackend())
    pipeline = create_default_pipeline()
    entries: list[EpochGalleryEntry] = []

    while not pipeline.is_finished:
        pipeline.step_to_checkpoint(context)
        report = context.history.reports[-1]
        stage_id = report.stage_id
        frame = render_era_visual_frame(
            context.fields,
            stage_id=stage_id,
            progress=1.0,
            config=config.visual_director,
        )
        image = upscale_rgb(frame.rgb, image_scale)
        png_name = f"{len(entries) + 1:02d}_{_safe_stage_filename(stage_id)}.png"
        png_path = output / png_name
        save_rgb_png(png_path, image)
        entries.append(
            EpochGalleryEntry(
                index=len(entries) + 1,
                stage_id=stage_id,
                title=report.title,
                png_path=png_path,
                progress=1.0,
                metrics=dict(report.metrics),
            )
        )
        pipeline.advance(context)

    manifest_path = output / "manifest.json"
    markdown_path = output / "index.md"
    _write_manifest(manifest_path, config=config, entries=entries, image_scale=image_scale)
    _write_markdown(markdown_path, config=config, entries=entries, image_scale=image_scale)
    return EpochGalleryResult(
        output_dir=output,
        entries=tuple(entries),
        manifest_path=manifest_path,
        markdown_path=markdown_path,
    )


def upscale_rgb(rgb: np.ndarray, image_scale: int) -> np.ndarray:
    """Upscale an RGB frame by an integer factor using deterministic bilinear sampling.

    Args:
        rgb: RGB image in the range ``[0, 1]``.
        image_scale: Positive integer scale factor. ``1`` returns a copy.

    Returns:
        Upscaled RGB float32 image.
    """

    source = np.asarray(rgb, dtype=np.float32)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"expected RGB image with shape (height, width, 3), got {source.shape}")
    if image_scale < 1:
        raise ValueError("image_scale must be positive")
    if image_scale == 1:
        return np.clip(source.copy(), 0.0, 1.0).astype(np.float32)

    height, width, _ = source.shape
    out_height = height * image_scale
    out_width = width * image_scale

    y = np.linspace(0.0, max(height - 1, 0), out_height, dtype=np.float32)
    x = np.linspace(0.0, max(width - 1, 0), out_width, dtype=np.float32)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    wy = (y - y0).reshape(out_height, 1, 1)
    wx = (x - x0).reshape(1, out_width, 1)

    top = source[y0[:, None], x0[None, :]] * (1.0 - wx) + source[y0[:, None], x1[None, :]] * wx
    bottom = source[y1[:, None], x0[None, :]] * (1.0 - wx) + source[y1[:, None], x1[None, :]] * wx
    return np.clip(top * (1.0 - wy) + bottom * wy, 0.0, 1.0).astype(np.float32)


def _write_manifest(
    path: Path,
    *,
    config: UniverseConfig,
    entries: Iterable[EpochGalleryEntry],
    image_scale: int,
) -> None:
    manifest = {
        "phrase": config.seed.phrase,
        "grid_size": config.seed.grid_size,
        "image_scale": image_scale,
        "entries": [
            {
                "index": entry.index,
                "stage_id": entry.stage_id,
                "title": entry.title,
                "file": entry.png_path.name,
                "progress": entry.progress,
                "metrics": entry.metrics,
            }
            for entry in entries
        ],
    }
    _atomic_write_text(path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def _write_markdown(
    path: Path,
    *,
    config: UniverseConfig,
    entries: Iterable[EpochGalleryEntry],
    image_scale: int,
) -> None:
    lines = [
        "# BBSim epoch gallery",
        "",
        f"Seed phrase: `{config.seed.phrase}`",
        f"Grid: `{config.seed.grid_size}` · image scale: `{image_scale}x`",
        "",
    ]
    for entry in entries:
        lines.extend(
            (
                f"## {entry.index:02d}. {entry.title}",
                "",
                f"Stage: `{entry.stage_id}`",
                "",
                f"![{entry.stage_id}]({entry.png_path.name})",
                "",
            )
        )
    _atomic_write_text(path, "\n".join(lines).rstrip() + "\n")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _safe_stage_filename(stage_id: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", stage_id.strip().lower())
    return sanitized.strip("_") or "epoch"
