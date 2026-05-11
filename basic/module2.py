"""
module2_satellite_position_clock.py

模块二：卫星位置与钟差计算模块。

输入模块一解析得到的 BroadcastEphemeris 和指定历元时间，输出卫星 ECEF
坐标 X/Y/Z 和卫星钟差。本模块按照广播星历标准公式手写实现，不调用任何
第三方 GNSS 定位库。
"""

from __future__ import annotations

import sys
from pathlib import Path
# 确保项目根目录在 sys.path 中，支持直接运行和作为模块导入
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dataclasses import dataclass
from datetime import datetime
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

from basic.module1 import BroadcastEphemeris, select_ephemeris


# 常量定义，所有公式均使用 SI 单位。
MU = 3.986004418e14  # 地球引力常数，单位 m^3/s^2
OMEGA_E = 7.2921150e-5  # 地球自转角速度，单位 rad/s
C = 299_792_458.0  # 光速，单位 m/s

BDT_EPOCH = datetime(2006, 1, 1, 0, 0, 0)  # 北斗时起点近似
SECONDS_IN_WEEK = 604_800.0
HALF_WEEK = 302_400.0
RELATIVITY_F = -2.0 * math.sqrt(MU) / (C * C)


def _is_ephemeris_healthy(eph: BroadcastEphemeris) -> bool:
    """按广播星历健康标志判断卫星是否可参与定位。"""

    return int(round(eph.health)) == 0


@dataclass
class SatelliteState:
    """单颗卫星在指定历元的计算状态。"""

    sat_id: str
    position: Tuple[float, float, float]  # ECEF 坐标，单位 m
    clock_bias: float  # 卫星钟差，单位 s
    eccentric_anomaly: float  # 偏近点角，单位 rad
    relativistic_correction: float  # 相对论效应修正，单位 s
    position_norm: float  # 卫星位置向量模长，单位 m
    status: str = "success"


def _bds_seconds_of_week(epoch_time: datetime) -> float:
    """将 datetime 转换为 BDS 周内秒。

    本课程仿真中，NAV 文件时间和仿真时间视为同一时间系统，因此不引入
    UTC/BDT 闰秒修正。
    """

    total_seconds = (epoch_time - BDT_EPOCH).total_seconds()
    return total_seconds % SECONDS_IN_WEEK


def _normalize_time(seconds: float) -> float:
    """将时间差归化到半周范围，避免跨周时出现过大的时间差。"""

    if seconds > HALF_WEEK:
        seconds -= SECONDS_IN_WEEK
    elif seconds < -HALF_WEEK:
        seconds += SECONDS_IN_WEEK
    return seconds


def _solve_kepler(mean_anomaly: float, eccentricity: float, max_iter: int, tol: float) -> float:
    """迭代求解 Kepler 方程 E = M + e * sin(E)。"""

    eccentric_anomaly = mean_anomaly
    for _ in range(max_iter):
        next_value = mean_anomaly + eccentricity * math.sin(eccentric_anomaly)
        if abs(next_value - eccentric_anomaly) < tol:
            return next_value
        eccentric_anomaly = next_value
    return eccentric_anomaly


def _is_bds_geo(sat_id: str) -> bool:
    """粗略判断北斗 GEO 卫星。

    BDS-2 中 C01-C05 通常为 GEO 卫星；部分 BDS-3 GEO 编号也可能存在。
    对未知编号，本项目按普通 MEO/IGSO 公式处理。
    """

    try:
        prn = int(sat_id[1:])
    except ValueError:
        return False
    return 1 <= prn <= 5 or 59 <= prn <= 63


def calculate_satellite_position_clock(
    eph: BroadcastEphemeris,
    epoch_time: datetime,
    max_kepler_iter: int = 30,
    kepler_tol: float = 1e-13,
    apply_tgd: bool = False,
    raise_on_abnormal: bool = True,
    require_healthy: bool = True,
) -> SatelliteState:
    """根据广播星历计算卫星 ECEF 坐标和卫星钟差。"""

    if require_healthy and not _is_ephemeris_healthy(eph):
        raise ValueError(f"{eph.sat_id} 健康状态异常，health={eph.health}")

    if eph.sqrt_a <= 0.0:
        raise ValueError(f"{eph.sat_id} 的 sqrtA 非法：{eph.sqrt_a}")

    # tk 是当前历元相对星历参考时刻 toe 的时间差。
    t = _bds_seconds_of_week(epoch_time)
    tk = _normalize_time(t - eph.toe)

    # 轨道长半轴 A = sqrtA^2；平均角速度 n = n0 + delta_n。
    semi_major_axis = eph.sqrt_a * eph.sqrt_a
    mean_motion_0 = math.sqrt(MU / (semi_major_axis ** 3))
    mean_motion = mean_motion_0 + eph.delta_n

    # 平近点角 M = M0 + n * tk。
    mean_anomaly = math.fmod(eph.m0 + mean_motion * tk, 2.0 * math.pi)

    # 求偏近点角 E，这是后续计算真近点角和径向距离的基础。
    eccentric_anomaly = _solve_kepler(
        mean_anomaly,
        eph.eccentricity,
        max_kepler_iter,
        kepler_tol,
    )

    sin_e = math.sin(eccentric_anomaly)
    cos_e = math.cos(eccentric_anomaly)

    # 真近点角 v，以及未经改正的升交距角 phi = v + omega。
    true_anomaly = math.atan2(
        math.sqrt(1.0 - eph.eccentricity * eph.eccentricity) * sin_e,
        cos_e - eph.eccentricity,
    )
    phi = true_anomaly + eph.omega

    # 二倍升交距角摄动修正：改正纬度幅角 u、轨道半径 r、轨道倾角 i。
    sin_2phi = math.sin(2.0 * phi)
    cos_2phi = math.cos(2.0 * phi)
    delta_u = eph.cus * sin_2phi + eph.cuc * cos_2phi
    delta_r = eph.crs * sin_2phi + eph.crc * cos_2phi
    delta_i = eph.cis * sin_2phi + eph.cic * cos_2phi

    u = phi + delta_u
    r = semi_major_axis * (1.0 - eph.eccentricity * cos_e) + delta_r
    i = eph.i0 + eph.idot * tk + delta_i

    # 轨道平面坐标。
    x_orb = r * math.cos(u)
    y_orb = r * math.sin(u)

    # 将轨道平面坐标旋转到地固 ECEF 坐标系。
    if _is_bds_geo(eph.sat_id):
        # 北斗 GEO 星历通常需要额外考虑约 5 度倾角坐标旋转。
        omega_k = eph.omega0 + eph.omega_dot * tk - OMEGA_E * eph.toe
        x_g = x_orb * math.cos(omega_k) - y_orb * math.cos(i) * math.sin(omega_k)
        y_g = x_orb * math.sin(omega_k) + y_orb * math.cos(i) * math.cos(omega_k)
        z_g = y_orb * math.sin(i)

        geo_tilt = math.radians(-5.0)
        x_t = x_g
        y_t = y_g * math.cos(geo_tilt) + z_g * math.sin(geo_tilt)
        z_t = -y_g * math.sin(geo_tilt) + z_g * math.cos(geo_tilt)

        cos_rot = math.cos(OMEGA_E * tk)
        sin_rot = math.sin(OMEGA_E * tk)
        x = x_t * cos_rot + y_t * sin_rot
        y = -x_t * sin_rot + y_t * cos_rot
        z = z_t
    else:
        omega_k = eph.omega0 + (eph.omega_dot - OMEGA_E) * tk - OMEGA_E * eph.toe
        x = x_orb * math.cos(omega_k) - y_orb * math.cos(i) * math.sin(omega_k)
        y = x_orb * math.sin(omega_k) + y_orb * math.cos(i) * math.cos(omega_k)
        z = y_orb * math.sin(i)

    position_norm = math.sqrt(x * x + y * y + z * z)
    status = "计算成功"
    if not 1.0e7 <= position_norm <= 6.0e7:
        status = "卫星坐标数量级异常"
        if raise_on_abnormal:
            raise ValueError(f"{eph.sat_id} 卫星位置模长异常：{position_norm:.3f} m")

    # 卫星钟差：多项式项 + 相对论效应修正。
    dt_clock = _normalize_time((epoch_time - eph.toc).total_seconds())
    relativity = RELATIVITY_F * eph.eccentricity * eph.sqrt_a * sin_e
    clock_bias = eph.af0 + eph.af1 * dt_clock + eph.af2 * dt_clock * dt_clock + relativity
    if apply_tgd:
        clock_bias -= eph.tgd1

    return SatelliteState(
        sat_id=eph.sat_id,
        position=(x, y, z),
        clock_bias=clock_bias,
        eccentric_anomaly=eccentric_anomaly,
        relativistic_correction=relativity,
        position_norm=position_norm,
        status=status,
    )


def satellite_position_and_clock(
    eph: BroadcastEphemeris,
    epoch_time: datetime,
) -> Tuple[Tuple[float, float, float], float]:
    """便捷接口：返回 ((X, Y, Z), 卫星钟差)。"""

    state = calculate_satellite_position_clock(eph, epoch_time)
    return state.position, state.clock_bias


def calculate_all_satellite_positions(
    nav_data: Dict[str, List[BroadcastEphemeris]],
    epoch_time: datetime,
) -> List[dict]:
    """计算指定测试历元下健康北斗卫星的位置和钟差。"""

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
            state = calculate_satellite_position_clock(
                eph,
                epoch_time,
                raise_on_abnormal=False,
            )
            x, y, z = state.position
            records.append(
                {
                    "epoch_time": epoch_time.isoformat(sep=" "),
                    "sat_id": sat_id,
                    "X": x,
                    "Y": y,
                    "Z": z,
                    "satellite_clock_bias": state.clock_bias,
                    "relativistic_correction": state.relativistic_correction,
                    "position_norm": state.position_norm,
                    "health": eph.health,
                    "status": state.status,
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
    from module1_nav_parser import parse_rinex_nav

    nav = parse_rinex_nav("nav/tarc0910.26b_cnav")
    test_epoch = datetime(2026, 4, 1, 0, 0, 0)
    rows = calculate_all_satellite_positions(nav, test_epoch)
    paths = save_satellite_position_outputs(rows, "output", test_epoch)
    print(f"模块二输出文件：{paths['csv']}，{paths['summary']}")
