"""
module2.py

模块二：卫星位置、钟差与传播延迟修正模块。

功能：
1. 根据广播星历计算北斗卫星在 ECEF 坐标系下的位置，单位为米；
2. 实现轨道摄动修正（delta_u、delta_r、delta_i）；
3. 实现卫星钟差计算，包括多项式钟差修正和相对论效应修正；
4. 输出 module2_satellite_debug.csv（含中间调试变量）和卫星位置钟差 CSV；
5. 输出 module2_pseudorange_correction_debug.csv（伪距修正调试文件，仅用于调试和实验报告展示，不参与 SPP 解算）。

注：核心计算逻辑统一调用 module1.py 的函数，模块二本身为组织与输出层。
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
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from basic.module1 import (
    BroadcastEphemeris,
    compute_satellite_clock_bias,
    compute_satellite_position,
    compute_satellite_position_with_debug,
    select_ephemeris,
)
from basic.module3 import (
    generate_simulated_pseudorange_record,
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


def calculate_satellite_debug_data(
    nav_data: Dict[str, List[BroadcastEphemeris]],
    epoch_time: datetime,
) -> List[dict]:
    """计算指定测试历元下健康北斗卫星的位置中间调试变量。

    返回记录包含 Kepler 轨道参数、摄动修正量、轨道平面坐标等，
    用于输出 module2_satellite_debug.csv。

    底层调用 module1 的 compute_satellite_position_with_debug。
    """
    records: List[dict] = []
    for sat_id in sorted(nav_data):
        eph = select_ephemeris(nav_data, sat_id, epoch_time, healthy_only=True)
        if eph is None:
            continue
        try:
            debug = compute_satellite_position_with_debug(eph, epoch_time)
            debug["epoch_time"] = epoch_time.isoformat(sep=" ")
            debug["sat_id"] = sat_id
            debug["health"] = eph.health
            records.append(debug)
        except Exception:
            continue
    return records


C = 299_792_458.0


def generate_pseudorange_correction_debug_records(
    nav_data: Dict[str, List[BroadcastEphemeris]],
    epoch_time: datetime,
    receiver_approx: Tuple[float, float, float],
    rng: Optional[random.Random] = None,
) -> List[dict]:
    """生成伪距修正调试记录，用于输出 module2_pseudorange_correction_debug.csv。

    说明：
    - 该文件仅用于模块二调试和实验报告展示；
    - 用于展示 rho、卫星钟差、SISRE、电离层误差、对流层误差、接收机钟差、
      噪声与模拟伪距之间的关系；
    - 该文件不作为模块三 SPP 解算输入；
    - 模块三和模块四仍使用模拟伪距字典进行定位解算。

    参数：
        nav_data: 广播星历字典
        epoch_time: 观测历元
        receiver_approx: 接收机概略 ECEF 坐标 (x, y, z)，单位 m
        rng: 随机数生成器，保证可复现性

    返回：
        调试记录列表，每个记录包含几何距离、卫星钟差、各项误差及模拟伪距。
    """
    source = rng or random.Random()
    records: List[dict] = []
    for sat_id in sorted(nav_data):
        eph = select_ephemeris(nav_data, sat_id, epoch_time, healthy_only=True)
        if eph is None:
            continue
        try:
            sat_pos = compute_satellite_position(eph, epoch_time)
            clock_bias_s, _ = compute_satellite_clock_bias(eph, epoch_time)
            rec = generate_simulated_pseudorange_record(
                epoch_time=epoch_time,
                sat_id=sat_id,
                sat_position=sat_pos,
                receiver_true_position=receiver_approx,
                health=eph.health,
                rng=source,
            )
            rec["satellite_clock_bias_s"] = clock_bias_s
            rec["satellite_clock_correction_m"] = clock_bias_s * C
            rec["debug_pseudorange"] = rec["simulated_pseudorange"]
            rec["debug_note"] = "模拟伪距调试记录，不参与SPP解算"
            records.append(rec)
        except Exception:
            continue
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


def save_satellite_debug_csv(
    debug_records: List[dict],
    output_dir: str | Path,
) -> Path:
    """保存 module2_satellite_debug.csv，包含卫星位置计算的中间调试变量。

    字段说明（所有角度单位为 rad，距离单位为 m，时间单位为 s）：
    - tk: 相对 toe 的时间差
    - semi_major_axis: 轨道长半轴 a
    - mean_motion_0: 参考平均角速度 n0
    - mean_motion: 改正后平均角速度 n
    - mean_anomaly: 平近点角 M
    - eccentric_anomaly: 偏近点角 E
    - true_anomaly: 真近点角 f
    - phi: 纬度幅角
    - delta_u/delta_r/delta_i: 摄动修正量
    - u/r/i: 改正后的纬度幅角、轨道半径、轨道倾角
    - omega_k: 升交点赤经
    - x_orb/y_orb: 轨道平面坐标
    - x/y/z: ECEF 坐标
    - is_geo: 是否为 GEO 卫星
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "module2_satellite_debug.csv"

    fieldnames = [
        "epoch_time", "sat_id", "health", "tk", "semi_major_axis",
        "mean_motion_0", "mean_motion", "mean_anomaly", "eccentric_anomaly",
        "true_anomaly", "phi", "delta_u", "delta_r", "delta_i",
        "u", "r", "i", "omega_k", "x_orb", "y_orb",
        "x", "y", "z", "is_geo",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rec in debug_records:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})
    return csv_path


def save_pseudorange_correction_debug_csv(
    debug_records: List[dict],
    output_dir: str | Path,
) -> Path:
    """保存 module2_pseudorange_correction_debug.csv。

    该文件为模块二调试输出，仅用于展示 rho、卫星钟差、各项误差与模拟伪距
    之间的关系，不作为模块三 SPP 解算输入。

    字段说明：
    - epoch_time: 观测历元
    - sat_id: 卫星编号
    - sat_X, sat_Y, sat_Z: 卫星 ECEF 坐标（m）
    - health: 卫星健康状态
    - elevation_deg: 高度角（deg）
    - rho: 几何距离（m）
    - satellite_clock_bias_s: 卫星钟差（s）
    - satellite_clock_correction_m: 卫星钟差距离修正（m）
    - sisre_error: SISRE（m）
    - iono_error: 电离层延迟（m）
    - tropo_error: 对流层延迟（m）
    - receiver_clock_error: 接收机钟差（m）
    - noise_error: 观测噪声（m）
    - simulated_pseudorange: 模拟伪距（m）
    - debug_note: 调试说明
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "module2_pseudorange_correction_debug.csv"

    fieldnames = [
        "epoch_time", "sat_id", "sat_X", "sat_Y", "sat_Z", "health",
        "elevation_deg", "rho", "satellite_clock_bias_s",
        "satellite_clock_correction_m", "sisre_error", "iono_error",
        "tropo_error", "receiver_clock_error", "noise_error",
        "simulated_pseudorange", "debug_note",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rec in debug_records:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})
    return csv_path


if __name__ == "__main__":
    from basic.module1 import parse_nav_file

    nav = parse_nav_file("nav/tarc0910.26b_cnav")[0]
    test_epoch = datetime(2026, 4, 1, 0, 0, 0)
    receiver_approx = (-2267800.0, 5009340.0, 3221000.0)
    rng = random.Random(2026)

    # 1. 计算卫星位置与钟差，输出原有 CSV 和摘要
    rows = calculate_all_satellite_positions(nav, test_epoch)
    paths = save_satellite_position_outputs(rows, "outputs/basic/module", test_epoch)
    print(f"module2_satellite_position_clock.csv and summary saved.")

    # 2. 输出 module2_satellite_debug.csv（中间调试变量）
    debug_rows = calculate_satellite_debug_data(nav, test_epoch)
    debug_path = save_satellite_debug_csv(debug_rows, "outputs/basic/module")
    print(f"module2_satellite_debug.csv saved: {debug_path}")

    # 3. 输出 module2_pseudorange_correction_debug.csv（伪距修正调试文件）
    debug_rows = generate_pseudorange_correction_debug_records(nav, test_epoch, receiver_approx, rng=rng)
    debug_path = save_pseudorange_correction_debug_csv(debug_rows, "outputs/basic/module")
    print(f"module2_pseudorange_correction_debug.csv saved: {debug_path}")
