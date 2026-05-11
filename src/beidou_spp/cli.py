"""Command line entry point for the Beidou SPP full pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

from .config import PipelineConfig
from .pipeline import run_pipeline


def _parse_datetime(text: str) -> datetime:
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid datetime '{text}', use YYYY-MM-DDTHH:MM:SS") from exc


def _parse_ecef(text: str):
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--receiver-ecef must be x,y,z")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--receiver-ecef values must be numeric") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Beidou RINEX NAV SPP full-flow solver")
    parser.add_argument("--nav", required=True, type=Path, help="RINEX NAV file path")
    parser.add_argument("--output", default=Path("outputs"), type=Path, help="output directory")
    parser.add_argument("--receiver-ecef", default="-2267800.0,5009340.0,3221000.0", type=_parse_ecef)
    parser.add_argument("--start", default=datetime(2026, 4, 1, 0, 0, 0), type=_parse_datetime)
    parser.add_argument("--end", default=datetime(2026, 4, 1, 1, 0, 0), type=_parse_datetime)
    parser.add_argument("--interval", default=300, type=int, help="epoch interval in seconds")
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--max-iter", default=10, type=int)
    parser.add_argument("--threshold", default=1e-4, type=float)
    parser.add_argument("--elevation-mask", default=15.0, type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = PipelineConfig(
        nav=args.nav,
        output=args.output,
        receiver_ecef=args.receiver_ecef,
        start=args.start,
        end=args.end,
        interval=args.interval,
        seed=args.seed,
        max_iter=args.max_iter,
        threshold=args.threshold,
        elevation_mask=args.elevation_mask,
    )
    try:
        result = run_pipeline(config)
    except Exception as exc:
        print(f"运行失败: {exc}", file=sys.stderr)
        return 1
    print(f"输出目录: {result.output_dir.resolve()}")
    print(f"成功历元: {result.success_epochs}/{result.total_epochs}")
    print(f"RMS误差: {result.rms_error_m:.3f} m")
    for name, path in sorted(result.files.items()):
        print(f"{name}: {path}")
    return 0 if result.success_epochs > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

