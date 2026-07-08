"""Command line entry point for BBSim."""

from __future__ import annotations

import argparse
import sys

from bbsim.core.context import create_run_context
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
    parser.add_argument("--phrase", default=None, help="Personal seed phrase override.")
    parser.add_argument(
        "--simulation-config",
        default=None,
        help="Path to simulation TOML config. Defaults to config/simulation.toml.",
    )
    args = parser.parse_args(argv)

    if args.headless:
        return _run_headless(args.phrase, args.simulation_config)
    return _run_ui()


if __name__ == "__main__":
    raise SystemExit(main())
