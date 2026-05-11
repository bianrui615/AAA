"""
module2.py

模块二：卫星位置与钟差计算模块。

本模块为兼容层，核心计算已迁移至 module1.py。
保留 calculate_all_satellite_positions 和 save_satellite_position_outputs
供 module4/module5 调用，底层实现统一使用 module1 的函数。
"""

from __future__ import annotations

import sys
from pathlib import Path
# 确保项目根目录在 sys.path 中，支持直接运行和作为模块导入
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from datetime import datetime
import csv
import math
from pathlib import Path
from typing import Dict, List

from basic.module1 import (
    BroadcastEphemeris,
    compute_satellite_clock_bias,
    compute_satellite_position,
    select_ephemeris,
)


def calculate_all_satellite_positions(
    nav_data: Dict[str, List[BroadcastEphemeris]],
    epoch_time: datetime,
) -> List[dict]:
    """计算指定测试历元下健康北斗卫星的位置和钟差。

    底层实现调用 module1 的 compute_satellite_position 和
    compute_satellite_clock_bias。
    """

    records: List[dict] = []
    for sat_id in sorted(nav_data):
        eph = select_ephemeris(nav_data, sat_id, epoch_time, healthy_only=True)
        if eph is None:
            records.append(
                {
                    "epoch_time": epoch_time.isoformat(sep=" "),
                    "sat_id": sat_id,
                    "X": "",
                    "Y": "",
                    "Z": "",
                    "satellite_clock_bias": "",
                    "relativistic_correction": "",
                    "position_norm": "",
                    "health": "",
                    "status": "跳过：未找到健康星历",
                }
            )
            continue
        try:
            x, y, z = compute_satellite_position(eph, epoch_time)
            clock_bias, relativity = compute_satellite_clock_bias(eph, epoch_time)
            position_norm = math.sqrt(x * x + y * y + z * z)
            status = "计算成功"
            if not 1.0e7 <= position_norm <= 6.0e7:
                status = "卫星坐标数量级异常"
            records.append(
                {
                    "epoch_time": epoch_time.isoformat(sep=" "),
                    "sat_id": sat_id,
                    "X": x,
                    "Y": y,
                    "Z": z,
                    "satellite_clock_bias": clock_bias,
                    "relativistic_correction": relativity,
                    "position_norm": position_norm,
                    "health": eph.health,
                    "status": status,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "epoch_time": epoch_time.isoformat(sep=" "),
                    "sat_id": sat_id,
                    "X": "",
                    "Y": "",
                    "Z": "",
                    "satellite_clock_bias": "",
                    "relativistic_correction": "",
                    "position_norm": "",
                    "health": getattr(eph, "health", ""),
                    "status": f"计算失败：{exc}",
                }
            )
    return records


def save_satellite_position_outputs(
    position_records: List[dict],
    output_dir: str | Path,
    epoch_time: datetime,
) -> Dict[str, Path]:
    """保存模块二输出：卫星位置钟差 CSV 和摘要 TXT。"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "module2_satellite_position_clock.csv"
    summary_path = output_path / "module2_satellite_position_summary.txt"

    fieldnames = [
        "epoch_time",
        "sat_id",
        "X",
        "Y",
        "Z",
        "satellite_clock_bias",
        "relativistic_correction",
        "position_norm",
        "health",
        "status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(position_records)

    skipped_records = [row for row in position_records if str(row["status"]).startswith("跳过")]
    calculated_records = [row for row in position_records if row not in skipped_records]
    unhealthy_records = [
        row
        for row in position_records
        if row.get("health") not in ("", None) and int(round(float(row["health"]))) != 0
    ]
    success_records = [row for row in position_records if row["status"] == "计算成功"]
    abnormal_records = [row for row in position_records if row["status"] == "卫星坐标数量级异常"]
    failed_records = [row for row in position_records if str(row["status"]).startswith("计算失败")]
    norms = [float(row["position_norm"]) for row in success_records + abnormal_records if row["position_norm"] != ""]
    clock_biases = [
        float(row["satellite_clock_bias"])
        for row in success_records + abnormal_records
        if row["satellite_clock_bias"] != ""
    ]

    with summary_path.open("w", encoding="utf-8-sig") as file:
        file.write("模块二：卫星位置与钟差计算结果\n")
        file.write("=" * 50 + "\n")
        file.write(f"测试历元时间：{epoch_time.isoformat(sep=' ')}\n")
        file.write(f"NAV 中候选北斗卫星数量：{len(position_records)}\n")
        file.write(f"参与计算的健康卫星数量：{len(calculated_records)}\n")
        file.write(f"因健康状态异常或无健康星历跳过的卫星数量：{len(skipped_records)}\n")
        file.write(f"通过模块二健康筛选后仍异常的记录数量：{len(unhealthy_records)}\n")
        file.write(f"成功计算位置的卫星数量：{len(success_records)}\n")
        file.write(f"位置数量级异常的卫星数量：{len(abnormal_records)}\n")
        file.write(f"计算失败的卫星数量：{len(failed_records)}\n")
        if norms:
            min_norm = min(norms)
            max_norm = max(norms)
            quantity_ok = all(1.0e7 <= value <= 6.0e7 for value in norms)
            file.write(f"卫星位置模长范围：{min_norm:.3f} m 至 {max_norm:.3f} m\n")
            file.write(f"坐标数量级检查结果：{'通过，约为 10^7 m 量级' if quantity_ok else '警告，存在异常数量级'}\n")
        else:
            file.write("卫星位置模长范围：无可用结果\n")
            file.write("坐标数量级检查结果：无法判断\n")
        if clock_biases:
            file.write(
                f"卫星钟差范围：{min(clock_biases):.12e} s 至 {max(clock_biases):.12e} s\n"
            )
        else:
            file.write("卫星钟差范围：无可用结果\n")
        file.write("模块运行状态：卫星位置与钟差计算完成。\n")

    return {"csv": csv_path, "summary": summary_path}


if __name__ == "__main__":
    from basic.module1 import parse_rinex_nav

    nav = parse_rinex_nav("nav/tarc0910.26b_cnav")
    test_epoch = datetime(2026, 4, 1, 0, 0, 0)
    rows = calculate_all_satellite_positions(nav, test_epoch)
    paths = save_satellite_position_outputs(rows, "output", test_epoch)
    print(f"模块二输出文件：{paths['csv']}，{paths['summary']}")
