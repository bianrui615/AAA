"""北斗 SPP 全流程命令行入口。"""

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
        raise argparse.ArgumentTypeError(f"日期时间格式无效：'{text}'，请使用 YYYY-MM-DDTHH:MM:SS") from exc


def _parse_ecef(text: str):
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--receiver-ecef 必须写成 x,y,z")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--receiver-ecef 的三个值必须是数字") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="北斗 RINEX NAV SPP 全流程解算程序")
    parser.add_argument("--nav", required=True, type=Path, help="RINEX NAV 文件路径")
    parser.add_argument("--output", default=Path("outputs"), type=Path, help="输出目录")
    parser.add_argument("--receiver-ecef", default="-2267800.0,5009340.0,3221000.0", type=_parse_ecef)
    parser.add_argument("--start", default=datetime(2026, 4, 1, 0, 0, 0), type=_parse_datetime)
    parser.add_argument("--end", default=datetime(2026, 4, 1, 1, 0, 0), type=_parse_datetime)
    parser.add_argument("--interval", default=300, type=int, help="历元间隔，单位秒")
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
