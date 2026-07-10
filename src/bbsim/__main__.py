"""Command line entry point for BBSim."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from bbsim.core.context import create_run_context
from bbsim.core.epoch_gallery import render_epoch_gallery
from bbsim.core.pipeline import create_default_pipeline
from bbsim.core.simulation_config import load_simulation_config
from bbsim.numeric.numpy_backend import NumpyBackend


def _run_headless(phrase: str | None, simulation_config_path: str | None) -> int:
    config = load_simulation_config(simulation_config_path, phrase_override=phrase)
    context = create_run_context(config=config, backend=NumpyBackend())
    pipeline = create_default_pipeline()

    reports = []
    pipeline.enter_current(context)
    while not pipeline.is_finished:
        pipeline.step_to_checkpoint(context)
        if context.history.reports:
            reports.append(context.history.reports[-1])
        pipeline.advance(context)

    print(f"BBSim {config.seed.phrase!r}")
    print(
        "Config: "
        f"grid={config.seed.grid_size}, "
        f"ripple={config.seed.fluctuation_amplitude:.3f}, "
        f"inflation={config.inflation.strength:.3f}×{config.inflation.duration:.3f}, "
        f"Ωb={config.cosmology.omega_b:.4f}, "
        f"Ωdm={config.cosmology.omega_dm:.4f}, "
        f"ΩΛ={config.cosmology.omega_lambda:.4f}"
    )
    for report in reports:
        print(f"[{report.stage_id}] {report.title}")
        for line in report.summary_lines:
            print(f"  - {line}")
    return 0


def _run_epoch_gallery(
    phrase: str | None,
    simulation_config_path: str | None,
    output_dir: str,
    image_scale: int,
) -> int:
    config = load_simulation_config(simulation_config_path, phrase_override=phrase)
    result = render_epoch_gallery(
        config=config,
        output_dir=Path(output_dir),
        image_scale=image_scale,
    )

    print(f"BBSim epoch gallery written to {result.output_dir}")
    print(f"Frames: {len(result.entries)}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Index: {result.markdown_path}")
    for entry in result.entries:
        print(f"[{entry.index:02d}] {entry.stage_id}: {entry.png_path.name}")
    return 0


def _run_ui() -> int:
    try:
        from bbsim.ui.app import run_app
    except ImportError as exc:
        print(
            "UI dependencies are not installed. Run: pip install -e '.[dev]'",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 2
    return run_app()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BBSim prototype runner.")
    parser.add_argument("--headless", action="store_true", help="Run the pipeline without Qt UI.")
    parser.add_argument(
        "--epoch-gallery",
        default=None,
        metavar="DIR",
        help="Run headless and save one PNG snapshot per epoch into DIR.",
    )
    parser.add_argument(
        "--gallery-scale",
        type=int,
        default=4,
        help="Integer PNG upscale factor for --epoch-gallery. Defaults to 4.",
    )
    parser.add_argument("--phrase", default=None, help="Personal seed phrase override.")
    parser.add_argument(
        "--simulation-config",
        default=None,
        help="Path to simulation TOML config. Defaults to config/simulation.toml.",
    )
    args = parser.parse_args(argv)

    if args.epoch_gallery is not None:
        return _run_epoch_gallery(
            args.phrase,
            args.simulation_config,
            args.epoch_gallery,
            args.gallery_scale,
        )

    if args.headless:
        return _run_headless(args.phrase, args.simulation_config)
    return _run_ui()


if __name__ == "__main__":
    raise SystemExit(main())
