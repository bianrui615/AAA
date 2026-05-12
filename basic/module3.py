"""
module3.py

模块三：伪距生成与单点定位解算核心算法模块。

本模块根据卫星 ECEF 坐标和预设接收机真实坐标生成模拟伪距，然后使用
伪距观测方程和迭代最小二乘算法解算接收机 ECEF 坐标和接收机钟差。
最小二乘流程手写实现，仅使用 numpy 进行矩阵运算。
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
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from basic.module1 import (
    compute_elevation as satellite_elevation_deg,
    compute_geometric_range as geometric_distance,
    compute_satellite_position,
    ecef_to_blh,
    parse_nav_file,
    select_ephemeris,
    simulate_pseudorange,
)


ECEF = Tuple[float, float, float]
C = 299_792_458.0


# ============================================================================
# 伪距修正函数（任务1：Saastamoinen 对流层 + 简化电离层 + 卫星钟差）
# ============================================================================

def saastamoinen_tropospheric_delay(
    elevation_deg: float,
    receiver_height_m: float = 0.0,
    pressure_hpa: float = 1013.25,
    temperature_k: float = 288.15,
    humidity: float = 0.5,
) -> float:
    """Saastamoinen 对流层延迟模型（简化版）。

    返回某高度角下的对流层延迟，单位米。高度角 < 5° 时强制按 5° 计算。

    参数:
        elevation_deg: 卫星高度角，单位度
        receiver_height_m: 接收机高程，单位米（当前未用于主路径，保留接口）
        pressure_hpa: 地面气压，默认标准大气 1013.25 hPa
        temperature_k: 地面温度，默认 288.15 K
        humidity: 相对湿度，默认 0.5
    返回:
        对流层延迟，单位米
    """
    el_rad = math.radians(max(elevation_deg, 5.0))
    e_vapor = humidity * 6.108 * math.exp(
        (17.15 * (temperature_k - 273.15) - 4684.0) / (temperature_k - 38.45)
    )
    zenith_rad = math.pi / 2.0 - el_rad
    delay = (0.002277 / math.sin(el_rad)) * (
        pressure_hpa
        + (1255.0 / temperature_k + 0.05) * e_vapor
        - math.tan(zenith_rad) ** 2
    )
    return delay


def simple_ionospheric_delay(
    elevation_deg: float,
    zenith_delay_m: float = 10.0,
) -> float:
    """简化电离层延迟模型：基于高度角的余割映射函数。

    I(el) = zenith_delay / sin(elevation)。高度角 < 5° 时按 5° 计算。

    参数:
        elevation_deg: 卫星高度角，单位度
        zenith_delay_m: 天顶方向电离层延迟，单位米（默认 10 m）
    返回:
        电离层延迟，单位米
    """
    el_rad = math.radians(max(elevation_deg, 5.0))
    return zenith_delay_m / math.sin(el_rad)


def apply_pseudorange_corrections(
    raw_pseudorange: float,
    satellite_clock_bias_s: float,
    elevation_deg: float,
    receiver_height_m: float = 0.0,
    enable_satellite_clock: bool = True,
    enable_tropospheric: bool = True,
    enable_ionospheric: bool = True,
    iono_zenith_delay_m: float = 10.0,
) -> Tuple[float, dict]:
    """对单颗卫星的伪距进行确定性修正。

    P_corrected = P_raw + c·dt_sat - T_iono - T_tropo

    参数:
        raw_pseudorange: 原始伪距，单位米
        satellite_clock_bias_s: 卫星钟差，单位秒
        elevation_deg: 卫星高度角，单位度
        receiver_height_m: 接收机高程，单位米
        enable_satellite_clock: 是否启用卫星钟差修正
        enable_tropospheric: 是否启用对流层修正
        enable_ionospheric: 是否启用电离层修正
        iono_zenith_delay_m: 天顶方向电离层延迟，单位米
    返回:
        (修正后伪距, 修正量明细字典)
    """
    sat_clock_m = C * satellite_clock_bias_s if enable_satellite_clock else 0.0
    tropo_m = (
        saastamoinen_tropospheric_delay(elevation_deg, receiver_height_m)
        if enable_tropospheric
        else 0.0
    )
    iono_m = (
        simple_ionospheric_delay(elevation_deg, iono_zenith_delay_m)
        if enable_ionospheric
        else 0.0
    )
    corrected = raw_pseudorange + sat_clock_m - tropo_m - iono_m
    return corrected, {
        "satellite_clock_correction_m": sat_clock_m,
        "tropospheric_correction_m": tropo_m,
        "ionospheric_correction_m": iono_m,
    }

def _is_satellite_healthy(
    sat_id: str,
    satellite_health: Optional[Dict[str, float]] = None,
) -> bool:
    """健康状态字典缺省时兼容旧接口；给定时仅 health=0 的卫星参与计算。"""

    if satellite_health is None:
        return True
    health = satellite_health.get(sat_id)
    if health is None:
        return False
    return int(round(float(health))) == 0


def filter_healthy_satellites(
    satellite_positions: Dict[str, ECEF],
    satellite_health: Optional[Dict[str, float]] = None,
) -> Dict[str, ECEF]:
    """按健康状态过滤卫星坐标，确保模块三只使用健康卫星。"""

    return {
        sat_id: position
        for sat_id, position in satellite_positions.items()
        if _is_satellite_healthy(sat_id, satellite_health)
    }


@dataclass
class SppSolution:
    """单历元 SPP 解算结果。

    clock_bias 使用米表示，与伪距观测方程 P = rho + clock_bias 保持一致。
    """

    status: str
    converged: bool
    x: float = math.nan
    y: float = math.nan
    z: float = math.nan
    lat: float = math.nan
    lon: float = math.nan
    height: float = math.nan
    clock_bias: float = math.nan
    pdop: float = math.nan
    gdop: float = math.nan
    iterations: int = 0
    satellite_count: int = 0
    rejected_outliers: int = 0
    elevation_mask_deg: float = 0.0
    message: str = ""


def generate_simulated_pseudorange_record(
    epoch_time: datetime,
    sat_id: str,
    sat_position: ECEF,
    receiver_true_position: ECEF,
    health: float = 0.0,
    rng: Optional[random.Random] = None,
    receiver_clock_error: Optional[float] = None,
) -> dict:
    """生成单颗卫星模拟伪距记录。

    薄包装：底层调用 module1.simulate_pseudorange，在返回字典中附加
    epoch_time、sat_id、sat_position 等元数据。字段命名统一使用
    iono_error / tropo_error（与 module1 一致）。

    参数:
        epoch_time: 历元时间
        sat_id: 卫星编号
        sat_position: 卫星 ECEF 坐标
        receiver_true_position: 接收机真实 ECEF 坐标
        health: 卫星健康状态，0 为正常
        rng: 随机数生成器，保证可复现性
        receiver_clock_error: 接收机钟差（米），为 None 时由 rng 生成
    返回:
        包含伪距、误差分量和元数据的字典
    """

    source = rng or random.Random()
    rho = geometric_distance(sat_position, receiver_true_position)
    sim = simulate_pseudorange(rho, source)

    # 若指定了接收机钟差，覆盖模拟值并更新伪距
    if receiver_clock_error is not None:
        old_clk = sim["receiver_clock_error"]
        sim["receiver_clock_error"] = receiver_clock_error
        sim["pseudorange"] = sim["pseudorange"] - old_clk + receiver_clock_error

    elevation_deg = satellite_elevation_deg(receiver_true_position, sat_position)

    return {
        "epoch_time": epoch_time.isoformat(sep=" "),
        "sat_id": sat_id,
        "sat_X": sat_position[0],
        "sat_Y": sat_position[1],
        "sat_Z": sat_position[2],
        "health": health,
        "satellite_clock_bias": 0.0,
        "satellite_clock_correction_m": 0.0,
        "elevation_deg": elevation_deg,
        "true_receiver_X": receiver_true_position[0],
        "true_receiver_Y": receiver_true_position[1],
        "true_receiver_Z": receiver_true_position[2],
        "rho": sim["rho"],
        "sisre_error": sim["sisre_error"],
        "iono_error": sim["iono_error"],
        "tropo_error": sim["tropo_error"],
        "receiver_clock_error": sim["receiver_clock_error"],
        "noise_error": sim["noise_error"],
        "raw_simulated_pseudorange": sim["pseudorange"],
        "simulated_pseudorange": sim["pseudorange"],
    }


def generate_simulated_pseudorange(
    sat_position: ECEF,
    receiver_true_position: ECEF,
    rng: Optional[random.Random] = None,
) -> float:
    """生成单颗卫星模拟伪距，只返回最终 P 值。"""

    record = generate_simulated_pseudorange_record(
        datetime(1970, 1, 1),
        "",
        sat_position,
        receiver_true_position,
        0.0,
        rng,
    )
    return float(record["simulated_pseudorange"])


def generate_simulated_pseudorange_records(
    satellite_positions: Dict[str, ECEF],
    receiver_true_position: ECEF,
    epoch_time: datetime,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
    satellite_health: Optional[Dict[str, float]] = None,
) -> List[dict]:
    """为一个历元的多颗卫星生成伪距明细记录，支持随机种子复现。"""

    source = rng or random.Random(seed)
    receiver_clock_error = 60.0 + source.gauss(0.0, 12.0)
    records: List[dict] = []
    healthy_positions = filter_healthy_satellites(satellite_positions, satellite_health)
    for sat_id in sorted(healthy_positions):
        records.append(
            generate_simulated_pseudorange_record(
                epoch_time,
                sat_id,
                healthy_positions[sat_id],
                receiver_true_position,
                0.0 if satellite_health is None else float(satellite_health[sat_id]),
                source,
                receiver_clock_error=receiver_clock_error,
            )
        )
    return records


def pseudorange_records_to_dict(records: List[dict]) -> Dict[str, float]:
    """将伪距明细列表转换为 sat_id -> simulated_pseudorange 字典。"""

    pseudoranges: Dict[str, float] = {}
    for row in records:
        value = row.get("simulated_pseudorange")
        if value in ("", None):
            value = row["raw_simulated_pseudorange"]
        pseudoranges[row["sat_id"]] = float(value)
    return pseudoranges


def generate_simulated_pseudoranges(
    satellite_positions: Dict[str, ECEF],
    receiver_true_position: ECEF,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, float]:
    """为多颗卫星生成模拟伪距字典。"""

    records = generate_simulated_pseudorange_records(
        satellite_positions,
        receiver_true_position,
        datetime(1970, 1, 1),
        seed=seed,
        rng=rng,
        satellite_health=None,
    )
    return pseudorange_records_to_dict(records)


def _compute_dops(design_matrix: np.ndarray) -> Tuple[float, float]:
    """根据最终设计矩阵计算 PDOP 和 GDOP。"""

    q_matrix = np.linalg.inv(design_matrix.T @ design_matrix)
    pdop = math.sqrt(max(q_matrix[0, 0] + q_matrix[1, 1] + q_matrix[2, 2], 0.0))
    gdop = math.sqrt(max(pdop * pdop + q_matrix[3, 3], 0.0))
    return pdop, gdop


def _has_usable_receiver_position(receiver_position: ECEF) -> bool:
    """判断概略接收机坐标是否可用于高度角和残差预筛选。"""

    rx, ry, rz = receiver_position
    return (
        math.isfinite(rx)
        and math.isfinite(ry)
        and math.isfinite(rz)
        and math.sqrt(rx * rx + ry * ry + rz * rz) >= 1_000_000.0
    )


def _filter_by_elevation(
    satellite_positions: Dict[str, ECEF],
    pseudoranges: Dict[str, float],
    receiver_position: ECEF,
    elevation_mask_deg: float,
) -> Tuple[Dict[str, ECEF], Dict[str, float], int]:
    """按高度角筛选可见卫星，返回过滤后的坐标、伪距和剔除数量。"""

    if elevation_mask_deg <= 0.0:
        return satellite_positions, pseudoranges, 0

    # 若概略位置过于接近原点（未提供有效初值），跳过高度角过滤
    if not _has_usable_receiver_position(receiver_position):
        return satellite_positions, pseudoranges, 0

    filtered_positions: Dict[str, ECEF] = {}
    filtered_pseudoranges: Dict[str, float] = {}
    removed_count = 0

    for sat_id in satellite_positions:
        if sat_id not in pseudoranges:
            continue
        elevation = satellite_elevation_deg(receiver_position, satellite_positions[sat_id])
        if elevation >= elevation_mask_deg:
            filtered_positions[sat_id] = satellite_positions[sat_id]
            filtered_pseudoranges[sat_id] = pseudoranges[sat_id]
        else:
            removed_count += 1

    return filtered_positions, filtered_pseudoranges, removed_count


def _reject_pseudorange_outliers(
    satellite_positions: Dict[str, ECEF],
    pseudoranges: Dict[str, float],
    receiver_approx_position: ECEF,
    approx_clock_bias: float = 0.0,
    threshold_sigma: float = 3.0,
) -> Tuple[Dict[str, float], int]:
    """伪距粗差初步剔除，优先使用 MAD 稳健残差尺度。

    若剔除后可用卫星数 < 4，则放弃剔除、保留全部，避免过度剔除导致无法定位。
    """

    if len(satellite_positions) < 5:
        return pseudoranges, 0
    if not _has_usable_receiver_position(receiver_approx_position):
        return pseudoranges, 0

    residuals: Dict[str, float] = {}
    for sat_id, sat_pos in satellite_positions.items():
        if sat_id not in pseudoranges:
            continue
        rho = geometric_distance(sat_pos, receiver_approx_position)
        residual = pseudoranges[sat_id] - (rho + approx_clock_bias)
        residuals[sat_id] = residual

    if len(residuals) < 5:
        return pseudoranges, 0

    values = np.asarray(list(residuals.values()), dtype=float)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    sigma = 1.4826 * mad
    if sigma < 1e-6:
        sigma = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        center = float(np.mean(values))
    if sigma < 1e-6:
        return pseudoranges, 0

    filtered_pseudoranges = dict(pseudoranges)
    removed_count = 0
    for sat_id, residual in residuals.items():
        if abs(residual - center) > threshold_sigma * sigma:
            if sat_id in filtered_pseudoranges:
                del filtered_pseudoranges[sat_id]
                removed_count += 1

    # 若剔除后卫星数 < 4，放弃剔除
    remaining_common = [
        sat_id for sat_id in satellite_positions if sat_id in filtered_pseudoranges
    ]
    if len(remaining_common) < 4:
        return pseudoranges, 0

    return filtered_pseudoranges, removed_count


def solve_spp(
    satellite_positions: Dict[str, ECEF],
    pseudoranges: Dict[str, float],
    initial_position: Optional[ECEF] = None,
    initial_clock_bias: float = 0.0,
    max_iter: int = 10,
    convergence_threshold: float = 1e-2,
    satellite_health: Optional[Dict[str, float]] = None,
    elevation_mask_deg: float = 0.0,
    enable_pseudorange_outlier_filter: bool = False,
    apply_corrections: bool = True,
    satellite_clock_biases: Optional[Dict[str, float]] = None,
    satellite_elevations: Optional[Dict[str, float]] = None,
) -> SppSolution:
    """使用迭代最小二乘求解接收机坐标和接收机钟差。

    参数:
        satellite_positions: 卫星 ECEF 坐标字典
        pseudoranges: 原始模拟伪距字典（sat_id -> 伪距，单位米）
        apply_corrections: 是否对伪距进行确定性修正（卫星钟差/对流层/电离层），默认 True
        satellite_clock_biases: 各卫星钟差（秒），apply_corrections=True 时使用
        satellite_elevations: 各卫星高度角（度），apply_corrections=True 时使用
    """

    healthy_positions = filter_healthy_satellites(satellite_positions, satellite_health)

    # 伪距修正：在最小二乘迭代前，对每颗卫星的伪距进行确定性修正
    if apply_corrections and (satellite_clock_biases is not None or satellite_elevations is not None):
        corrected_pseudoranges: Dict[str, float] = {}
        for sat_id, raw_pr in pseudoranges.items():
            clock_bias_s = (satellite_clock_biases or {}).get(sat_id, 0.0)
            elev_deg = (satellite_elevations or {}).get(sat_id, 90.0)
            corrected_pr, _ = apply_pseudorange_corrections(
                raw_pr, clock_bias_s, elev_deg
            )
            corrected_pseudoranges[sat_id] = corrected_pr
        pseudoranges = corrected_pseudoranges
    common_sats = [sat_id for sat_id in healthy_positions if sat_id in pseudoranges]
    if len(common_sats) < 4:
        return SppSolution(
            status="失败",
            converged=False,
            satellite_count=len(common_sats),
            message="可用卫星数量少于 4 颗",
        )

    x, y, z = initial_position if initial_position is not None else (0.0, 0.0, 0.0)
    clock_bias = initial_clock_bias
    last_design_matrix: Optional[np.ndarray] = None
    total_rejected = 0

    # 按高度角过滤（使用当前概略位置）
    if elevation_mask_deg > 0.0:
        healthy_positions, pseudoranges, _ = _filter_by_elevation(
            healthy_positions, pseudoranges, (x, y, z), elevation_mask_deg
        )
        common_sats = [sat_id for sat_id in healthy_positions if sat_id in pseudoranges]
        if len(common_sats) < 4:
            return SppSolution(
                status="失败",
                converged=False,
                satellite_count=len(common_sats),
                message=f"高度角截止 {elevation_mask_deg}° 后可用卫星不足 4 颗",
            )

    for iteration in range(1, max_iter + 1):
        # 伪距粗差剔除（每次迭代前用当前概略位置，仅在启用时执行）
        if enable_pseudorange_outlier_filter and len(common_sats) >= 5:
            pseudoranges, rejected = _reject_pseudorange_outliers(
                healthy_positions,
                pseudoranges,
                (x, y, z),
                approx_clock_bias=clock_bias,
                threshold_sigma=3.0,
            )
            if rejected > 0:
                total_rejected += rejected
                common_sats = [sat_id for sat_id in healthy_positions if sat_id in pseudoranges]
                if len(common_sats) < 4:
                    return SppSolution(
                        status="失败",
                        converged=False,
                        iterations=iteration,
                        satellite_count=len(common_sats),
                        message="粗差剔除后可用卫星不足 4 颗",
                    )

        design_rows = []
        residual_rows = []

        for sat_id in common_sats:
            xs, ys, zs = healthy_positions[sat_id]
            dx = xs - x
            dy = ys - y
            dz = zs - z
            rho = math.sqrt(dx * dx + dy * dy + dz * dz)
            if rho <= 0.0:
                return SppSolution(
                    status="失败",
                    converged=False,
                    iterations=iteration,
                    satellite_count=len(common_sats),
                    message="几何距离为 0，初值或卫星坐标异常",
                )

            # 伪距观测方程线性化：
            # observed - computed = H * delta_x
            computed_pseudorange = rho + clock_bias
            residual = pseudoranges[sat_id] - computed_pseudorange
            design_rows.append([-dx / rho, -dy / rho, -dz / rho, 1.0])
            residual_rows.append(residual)

        h_matrix = np.asarray(design_rows, dtype=float)
        v_vector = np.asarray(residual_rows, dtype=float)
        last_design_matrix = h_matrix

        try:
            delta = np.linalg.solve(h_matrix.T @ h_matrix, h_matrix.T @ v_vector)
        except np.linalg.LinAlgError:
            return SppSolution(
                status="失败",
                converged=False,
                iterations=iteration,
                satellite_count=len(common_sats),
                message="法方程矩阵不可逆，卫星几何结构不足",
            )

        x += float(delta[0])
        y += float(delta[1])
        z += float(delta[2])
        clock_bias += float(delta[3])

        if np.linalg.norm(delta[:3]) < convergence_threshold:
            lat, lon, height = ecef_to_blh(x, y, z)
            try:
                pdop, gdop = _compute_dops(last_design_matrix)
            except np.linalg.LinAlgError:
                pdop, gdop = math.nan, math.nan
            return SppSolution(
                status="成功",
                converged=True,
                x=x,
                y=y,
                z=z,
                lat=lat,
                lon=lon,
                height=height,
                clock_bias=clock_bias,
                pdop=pdop,
                gdop=gdop,
                iterations=iteration,
                satellite_count=len(common_sats),
                rejected_outliers=total_rejected,
                elevation_mask_deg=elevation_mask_deg,
                message="定位解算收敛",
            )

    return SppSolution(
        status="失败",
        converged=False,
        x=x,
        y=y,
        z=z,
        clock_bias=clock_bias,
        iterations=max_iter,
        satellite_count=len(common_sats),
        rejected_outliers=total_rejected,
        message="达到最大迭代次数仍未收敛",
    )


def save_single_epoch_spp_outputs(
    pseudorange_records: List[dict],
    spp_result: SppSolution,
    output_dir: str | Path,
    epoch_time: datetime,
    receiver_true_position: ECEF,
    elevation_mask_deg: float = 0.0,
) -> Dict[str, Path]:
    """保存模块三输出：伪距明细 CSV 和单历元 SPP 解算报告 TXT。"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "module3_pseudorange_single_epoch.csv"
    txt_path = output_path / "module3_spp_result_single_epoch.txt"

    fieldnames = [
        "epoch_time",
        "sat_id",
        "sat_X",
        "sat_Y",
        "sat_Z",
        "health",
        "satellite_clock_bias",
        "satellite_clock_correction_m",
        "elevation_deg",
        "true_receiver_X",
        "true_receiver_Y",
        "true_receiver_Z",
        "rho",
        "sisre_error",
        "iono_error",
        "tropo_error",
        "receiver_clock_error",
        "noise_error",
        "raw_simulated_pseudorange",
        "simulated_pseudorange",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pseudorange_records)

    if spp_result.converged:
        error_3d = geometric_distance(
            (spp_result.x, spp_result.y, spp_result.z),
            receiver_true_position,
        )
    else:
        error_3d = math.nan
    unhealthy_records = [
        row
        for row in pseudorange_records
        if row.get("health") not in ("", None) and int(round(float(row["health"]))) != 0
    ]

    with txt_path.open("w", encoding="utf-8-sig") as file:
        file.write("模块三：单历元模拟伪距与 SPP 解算结果\n")
        file.write("=" * 52 + "\n")
        file.write(f"测试历元时间：{epoch_time.isoformat(sep=' ')}\n")
        file.write(f"参与定位的卫星数量：{spp_result.satellite_count}\n")
        file.write(f"健康状态筛选：仅使用 health=0 的卫星，异常记录数量 {len(unhealthy_records)}\n")
        file.write(
            "接收机真实 ECEF 坐标："
            f"X={receiver_true_position[0]:.4f} m，"
            f"Y={receiver_true_position[1]:.4f} m，"
            f"Z={receiver_true_position[2]:.4f} m\n"
        )
        file.write(
            "解算得到的接收机 ECEF 坐标："
            f"X={spp_result.x:.4f} m，"
            f"Y={spp_result.y:.4f} m，"
            f"Z={spp_result.z:.4f} m\n"
        )
        file.write(
            "解算得到的经纬高坐标："
            f"纬度={spp_result.lat:.10f} deg，"
            f"经度={spp_result.lon:.10f} deg，"
            f"高程={spp_result.height:.4f} m\n"
        )
        file.write(f"接收机钟差估计值：{spp_result.clock_bias:.6f} m\n")
        file.write(f"三维定位误差：{error_3d:.6f} m\n")
        file.write(f"PDOP：{spp_result.pdop:.6f}\n")
        file.write(f"GDOP：{spp_result.gdop:.6f}\n")
        file.write(f"迭代次数：{spp_result.iterations}\n")
        file.write(f"高度角截止阈值：{elevation_mask_deg:.1f}°\n")
        file.write(f"剔除粗差卫星数：{spp_result.rejected_outliers}\n")
        file.write(f"是否收敛：{'是' if spp_result.converged else '否'}\n")
        file.write(f"解算状态：{spp_result.status}\n")
        file.write(f"失败原因：{'无' if spp_result.converged else spp_result.message}\n")
        file.write("模块运行状态：模拟伪距生成与单点定位解算完成。\n")

    return {"pseudorange_csv": csv_path, "spp_report": txt_path}


if __name__ == "__main__":
    # 模块三独立运行：生成单历元模拟伪距并进行 SPP 解算，输出 CSV 与 TXT
    NAV_FILE_PATH = "nav/tarc0910.26b_cnav"
    RECEIVER_TRUE_POSITION: ECEF = (-2267800.0, 5009340.0, 3221000.0)
    TEST_EPOCH_TIME = datetime(2026, 4, 1, 0, 0, 0)
    RANDOM_SEED = 2026
    MAX_ITERATIONS = 12
    CONVERGENCE_THRESHOLD = 1e-2
    ELEVATION_MASK_DEG = 0.0
    OUTPUT_DIR = "outputs/basic"

    try:
        nav_data, parse_info = parse_nav_file(NAV_FILE_PATH)
        sat_count = len(nav_data)
        eph_count = sum(len(records) for records in nav_data.values())
        print(f"NAV 解析完成：北斗卫星 {sat_count} 颗，星历记录 {eph_count} 条")

        # 计算卫星位置
        satellite_positions: Dict[str, ECEF] = {}
        satellite_health: Dict[str, float] = {}
        for sat_id in sorted(nav_data):
            eph = select_ephemeris(nav_data, sat_id, TEST_EPOCH_TIME, healthy_only=True)
            if eph is None:
                continue
            try:
                x, y, z = compute_satellite_position(eph, TEST_EPOCH_TIME)
                satellite_positions[sat_id] = (x, y, z)
                satellite_health[sat_id] = float(eph.health)
            except Exception:
                continue

        print(f"可用卫星位置：{len(satellite_positions)} 颗")
        if len(satellite_positions) < 4:
            raise ValueError("可用卫星少于 4 颗，无法解算")

        # 生成模拟伪距
        pseudo_records = generate_simulated_pseudorange_records(
            satellite_positions,
            RECEIVER_TRUE_POSITION,
            TEST_EPOCH_TIME,
            seed=RANDOM_SEED,
            satellite_health=satellite_health,
        )
        pseudoranges = pseudorange_records_to_dict(pseudo_records)
        print(f"模拟伪距已生成：{len(pseudo_records)} 条记录")

        # SPP 解算
        solution = solve_spp(
            satellite_positions,
            pseudoranges,
            initial_position=RECEIVER_TRUE_POSITION,
            max_iter=MAX_ITERATIONS,
            convergence_threshold=CONVERGENCE_THRESHOLD,
            satellite_health=satellite_health,
            elevation_mask_deg=ELEVATION_MASK_DEG,
            enable_pseudorange_outlier_filter=False,
        )

        if solution.converged:
            error_3d = geometric_distance(
                (solution.x, solution.y, solution.z),
                RECEIVER_TRUE_POSITION,
            )
            print(
                f"SPP 解算成功：X={solution.x:.4f}, Y={solution.y:.4f}, Z={solution.z:.4f}, "
                f"误差={error_3d:.4f} m, 迭代={solution.iterations}"
            )
        else:
            print(f"SPP 解算未收敛：{solution.message}")

        # 保存输出
        output_paths = save_single_epoch_spp_outputs(
            pseudo_records,
            solution,
            OUTPUT_DIR,
            TEST_EPOCH_TIME,
            RECEIVER_TRUE_POSITION,
            elevation_mask_deg=ELEVATION_MASK_DEG,
        )
        print(f"模块三输出已保存：")
        for key, path in output_paths.items():
            print(f"  {key}: {path}")
    except Exception as exc:
        print(f"模块三独立运行失败：{exc}")
        raise
