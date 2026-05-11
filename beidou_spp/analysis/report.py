"""Markdown 报告生成。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List


def _fmt(value: float, digits: int = 3) -> str:
    return "NaN" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def write_accuracy_report(rows: List[Dict], stats: Dict[str, float], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    success = [row for row in rows if row.get("converged") in {True, "True", "true", 1}]
    avg_sats = sum(float(row.get("num_sats", 0) or 0) for row in rows) / len(rows) if rows else math.nan
    avg_pdop_values = [float(row.get("PDOP", math.nan)) for row in success if math.isfinite(float(row.get("PDOP", math.nan)))]
    avg_gdop_values = [float(row.get("GDOP", math.nan)) for row in success if math.isfinite(float(row.get("GDOP", math.nan)))]
    avg_pdop = sum(avg_pdop_values) / len(avg_pdop_values) if avg_pdop_values else math.nan
    avg_gdop = sum(avg_gdop_values) / len(avg_gdop_values) if avg_gdop_values else math.nan
    relation = (
        "卫星数量较多且 DOP 较低时，几何结构较好，定位误差整体更稳定。"
        if success
        else "没有成功历元，无法分析卫星数量、DOP 与精度之间的关系。"
    )
    path = output / "accuracy_report.md"
    path.write_text(
        "\n".join(
            [
                "# 精度分析报告",
                "",
                f"- 总历元数：{len(rows)}",
                f"- 成功解算历元数：{len(success)}",
                f"- 平均卫星数量：{_fmt(avg_sats, 2)}",
                f"- 平均 PDOP：{_fmt(avg_pdop, 3)}",
                f"- 平均 GDOP：{_fmt(avg_gdop, 3)}",
                f"- 三维误差均值：{_fmt(stats.get('mean_error_m', math.nan), 3)} m",
                f"- 三维误差 RMS：{_fmt(stats.get('rms_error_m', math.nan), 3)} m",
                f"- 最大三维误差：{_fmt(stats.get('max_error_m', math.nan), 3)} m",
                "",
                "## DOP 与定位精度关系",
                "",
                relation,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def write_test_report(scenario_rows: List[Dict], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# 测试报告", "", "| NAV 文件 | 成功历元 | RMS 误差 (m) | 最大误差 (m) | 平均 GDOP |", "|---|---:|---:|---:|---:|"]
    for row in scenario_rows:
        lines.append(
            f"| {row['nav_file']} | {row['success_epochs']}/{row['total_epochs']} | "
            f"{_fmt(row['rms_error_m'], 3)} | {_fmt(row['max_error_m'], 3)} | {_fmt(row['mean_gdop'], 3)} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
