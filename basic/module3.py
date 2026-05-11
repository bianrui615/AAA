"""
module3_spp_solver.py

模块三：伪距生成与单点定位解算核心算法模块。

本模块根据卫星 ECEF 坐标和预设接收机真实坐标生成模拟伪距，然后使用
伪距观测方程和迭代最小二乘算法解算接收机 ECEF 坐标和接收机钟差。
最小二乘流程手写实现，仅使用 numpy 进行矩阵运算。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import math
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

import numpy as np


ECEF = Tuple[float, float, float]
C = 299_792_458.0

# WGS84 椭球常数，用于 ECEF 转经纬高。
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
MIN_ELEVATION_FOR_DELAY_DEG = 5.0


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


def satellite_elevation_deg(sat_position: ECEF, receiver_position: ECEF) -> float:
    """计算卫星相对接收机的高度角，单位为 deg。"""

    lat_deg, lon_deg, _ = ecef_to_blh(*receiver_position)
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    dx = sat_position[0] - receiver_position[0]
    dy = sat_position[1] - receiver_position[1]
    dz = sat_position[2] - receiver_position[2]

    east = -math.sin(lon) * dx + math.cos(lon) * dy
    north = (
        -math.sin(lat) * math.cos(lon) * dx
        - math.sin(lat) * math.sin(lon) * dy
        + math.cos(lat) * dz
    )
    up = (
        math.cos(lat) * math.cos(lon) * dx
        + math.cos(lat) * math.sin(lon) * dy
        + math.sin(lat) * dz
    )
    horizontal = math.sqrt(east * east + north * north)
    return math.degrees(math.atan2(up, horizontal))


def _delay_mapping_sin(elevation_deg: float) -> float:
    safe_elevation = max(elevation_deg, MIN_ELEVATION_FOR_DELAY_DEG)
    return max(math.sin(math.radians(safe_elevation)), 1e-3)


def simple_ionosphere_delay(sat_position: ECEF, receiver_position: ECEF) -> float:
    """简化电离层延迟模型，按高度角对 5 m 天顶延迟做投影。"""

    elevation_deg = satellite_elevation_deg(sat_position, receiver_position)
    return 5.0 / _delay_mapping_sin(elevation_deg)


def saastamoinen_troposphere_delay(sat_position: ECEF, receiver_position: ECEF) -> float:
    """简化 Saastamoinen 对流层延迟模型，返回路径延迟，单位 m。"""

    lat_deg, _, height = ecef_to_blh(*receiver_position)
    height = min(max(height, 0.0), 10000.0)
    elevation_deg = satellite_elevation_deg(sat_position, receiver_position)

    pressure_hpa = 1013.25 * (1.0 - 2.2557e-5 * height) ** 5.2568
    temperature_k = 291.15 - 0.0065 * height
    water_vapor_pressure_hpa = 11.7
    lat = math.radians(lat_deg)
    zenith_hydrostatic = 0.0022768 * pressure_hpa / (
        1.0 - 0.00266 * math.cos(2.0 * lat) - 0.00028 * height / 1000.0
    )
    zenith_wet = 0.002277 * (1255.0 / temperature_k + 0.05) * water_vapor_pressure_hpa
    return (zenith_hydrostatic + zenith_wet) / _delay_mapping_sin(elevation_deg)


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


def geometric_distance(sat_position: ECEF, receiver_position: ECEF) -> float:
    """计算卫星到接收机的几何距离 rho，单位 m。"""

    dx = sat_position[0] - receiver_position[0]
    dy = sat_position[1] - receiver_position[1]
    dz = sat_position[2] - receiver_position[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def generate_simulated_pseudorange_record(
    epoch_time: datetime,
    sat_id: str,
    sat_position: ECEF,
    receiver_true_position: ECEF,
    health: float = 0.0,
    rng: Optional[random.Random] = None,
    receiver_clock_error: Optional[float] = None,
    satellite_clock_bias: float = 0.0,
) -> dict:
    """生成单颗卫星模拟伪距，并保存每一项误差。

    误差模型：
    P = rho
        + random.gauss(0, 0.5)       # SISRE
        + random.gauss(10, 3)        # 电离层误差
        + random.gauss(4, 1.5)       # 对流层误差
        + (60 + random.gauss(0, 12)) # 接收机钟差
        + random.gauss(0, 0.3)       # 观测噪声
    """

    source = rng or random
    rho = geometric_distance(sat_position, receiver_true_position)
    sisre_error = source.gauss(0.0, 0.5)
    ionosphere_error = simple_ionosphere_delay(sat_position, receiver_true_position)
    troposphere_error = saastamoinen_troposphere_delay(sat_position, receiver_true_position)
    if receiver_clock_error is None:
        receiver_clock_error = 60.0 + source.gauss(0.0, 12.0)
    noise_error = source.gauss(0.0, 0.3)
    satellite_clock_correction_m = C * satellite_clock_bias
    raw_simulated_pseudorange = (
        rho
        + sisre_error
        + ionosphere_error
        + troposphere_error
        + receiver_clock_error
        - satellite_clock_correction_m
        + noise_error
    )
    simulated_pseudorange = (
        raw_simulated_pseudorange
        + satellite_clock_correction_m
        - ionosphere_error
        - troposphere_error
    )
    elevation_deg = satellite_elevation_deg(sat_position, receiver_true_position)

    return {
        "epoch_time": epoch_time.isoformat(sep=" "),
        "sat_id": sat_id,
        "sat_X": sat_position[0],
        "sat_Y": sat_position[1],
        "sat_Z": sat_position[2],
        "health": health,
        "satellite_clock_bias": satellite_clock_bias,
        "satellite_clock_correction_m": satellite_clock_correction_m,
        "elevation_deg": elevation_deg,
        "true_receiver_X": receiver_true_position[0],
        "true_receiver_Y": receiver_true_position[1],
        "true_receiver_Z": receiver_true_position[2],
        "rho": rho,
        "sisre_error": sisre_error,
        "ionosphere_error": ionosphere_error,
        "troposphere_error": troposphere_error,
        "receiver_clock_error": receiver_clock_error,
        "noise_error": noise_error,
        "raw_simulated_pseudorange": raw_simulated_pseudorange,
        "simulated_pseudorange": simulated_pseudorange,
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
    satellite_clock_biases: Optional[Dict[str, float]] = None,
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
                satellite_clock_bias=(
                    0.0
                    if satellite_clock_biases is None
                    else float(satellite_clock_biases.get(sat_id, 0.0))
                ),
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
        satellite_clock_biases=None,
    )
    return pseudorange_records_to_dict(records)


def ecef_to_blh(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """将 WGS84 ECEF 坐标转换为经纬高。

    返回值为 (lat, lon, height)，纬度和经度单位为度，高程单位为米。
    """

    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))

    # 迭代求解大地纬度和高程。
    for _ in range(20):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        height = p / max(math.cos(lat), 1e-15) - n
        next_lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + height)))
        if abs(next_lat - lat) < 1e-12:
            lat = next_lat
            break
        lat = next_lat

    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    height = p / max(math.cos(lat), 1e-15) - n
    return math.degrees(lat), math.degrees(lon), height


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
        elevation = satellite_elevation_deg(satellite_positions[sat_id], receiver_position)
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
    convergence_threshold: float = 1e-4,
    satellite_health: Optional[Dict[str, float]] = None,
    elevation_mask_deg: float = 0.0,
) -> SppSolution:
    """使用迭代最小二乘求解接收机坐标和接收机钟差。"""

    healthy_positions = filter_healthy_satellites(satellite_positions, satellite_health)
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
        # 伪距粗差剔除（每次迭代前用当前概略位置）
        if len(common_sats) >= 5:
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
        "ionosphere_error",
        "troposphere_error",
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
    print("请运行 basic/module5.py，以使用 NAV 星历执行模块三测试。")
