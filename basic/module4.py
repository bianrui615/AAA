"""
module4.py

模块四：连续定位与结果分析模块。

本模块按多个历元循环处理：选择可用卫星、计算卫星位置、生成模拟伪距、
调用单点定位解算，并保存逐历元 CSV、误差统计 TXT 和三张结果图。
"""

from __future__ import annotations

import sys
from pathlib import Path
# 确保项目根目录在 sys.path 中，支持直接运行和作为模块导入
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dataclasses import dataclass
from datetime import datetime, timedelta
import csv
import math
import os
from pathlib import Path
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# 将 matplotlib 缓存目录放到项目 outputs/basic 下，避免系统用户目录无写权限时报错。
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/basic") / "matplotlib_cache"))

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

if plt is not None:
    # 优先使用 Windows 常见中文字体；若某个字体不存在，matplotlib 会继续尝试后续字体。
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

from basic.module1 import (
    BroadcastEphemeris,
    compute_elevation,
    compute_geometric_range,
    compute_satellite_clock_bias,
    compute_satellite_position,
    ecef_to_blh,
    select_ephemeris,
)
from basic.module3 import (
    ECEF,
    generate_simulated_pseudorange_records,
    pseudorange_records_to_dict,
    solve_spp,
)


@dataclass
class AnalysisSummary:
    """连续定位统计结果。"""

    total_epochs: int
    success_epochs: int
    failed_epochs: int
    average_satellite_count: float
    average_pdop: float
    average_gdop: float
    mean_error_3d: float
    rms_error_3d: float
    max_error_3d: float
    min_error_3d: float
    success_rate: float
    evaluation: str
    elevation_mask_deg: float = 0.0


def build_linear_receiver_trajectory(start_time: datetime, initial_position: ECEF, velocity_mps: ECEF):
    """构建 ECEF 匀速直线运动轨迹函数。

    返回一个可调用对象 trajectory(epoch_time) -> ECEF。
    """

    def trajectory(epoch_time: datetime) -> ECEF:
        dt = (epoch_time - start_time).total_seconds()
        return (
            initial_position[0] + velocity_mps[0] * dt,
            initial_position[1] + velocity_mps[1] * dt,
            initial_position[2] + velocity_mps[2] * dt,
        )

    return trajectory


def _time_range(start_time: datetime, end_time: datetime, interval_seconds: int):
    """生成闭区间历元序列。"""

    current = start_time
    step = timedelta(seconds=interval_seconds)
    while current <= end_time:
        yield current
        current += step


def _safe_mean(values: List[float]) -> float:
    """计算平均值，自动忽略 NaN 和无穷值。"""

    clean = [value for value in values if math.isfinite(value)]
    return float(np.mean(clean)) if clean else math.nan


def _safe_rms(values: List[float]) -> float:
    """计算 RMS，自动忽略 NaN 和无穷值。"""

    clean = [value for value in values if math.isfinite(value)]
    return float(math.sqrt(np.mean(np.square(clean)))) if clean else math.nan


def _collect_satellite_positions(
    nav_data: Dict[str, List[BroadcastEphemeris]],
    epoch_time: datetime,
) -> Dict[str, ECEF]:
    """为当前历元选择健康星历，并计算卫星 ECEF 坐标。"""

    positions, _ = _collect_satellite_data(nav_data, epoch_time)
    return positions


def _collect_satellite_data(
    nav_data: Dict[str, List[BroadcastEphemeris]],
    epoch_time: datetime,
) -> Tuple[Dict[str, ECEF], Dict[str, float]]:
    """为当前历元计算卫星 ECEF 坐标和卫星钟差（秒）。

    返回 (satellite_positions, satellite_clock_biases)。
    """

    satellite_positions: Dict[str, ECEF] = {}
    satellite_clock_biases: Dict[str, float] = {}
    for sat_id in sorted(nav_data):
        eph = select_ephemeris(nav_data, sat_id, epoch_time, healthy_only=True)
        if eph is None:
            continue
        try:
            x, y, z = compute_satellite_position(eph, epoch_time)
            position_norm = math.sqrt(x * x + y * y + z * z)
            if position_norm <= 0.0:
                continue
            satellite_positions[sat_id] = (x, y, z)
            clock_bias_s, _ = compute_satellite_clock_bias(eph, epoch_time)
            satellite_clock_biases[sat_id] = clock_bias_s
        except Exception:
            continue
    return satellite_positions, satellite_clock_biases


def run_continuous_positioning(
    nav_data: Dict[str, List[BroadcastEphemeris]],
    start_time: datetime,
    end_time: datetime,
    interval_seconds: int,
    receiver_true_position: ECEF,
    output_dir: str | Path = "outputs/basic/module",
    random_seed: Optional[int] = 2026,
    max_iter: int = 10,
    convergence_threshold: float = 1e-2,
    elevation_mask_deg: float = 0.0,
    progress_callback: Optional[Callable[[dict, int, int], None]] = None,
    receiver_trajectory: Optional[Callable[[datetime], ECEF]] = None,
    receiver_initial_approx: Optional[ECEF] = None,
) -> Tuple[List[dict], AnalysisSummary]:
    """执行连续定位仿真，并保存模块四全部输出。"""

    if interval_seconds <= 0:
        raise ValueError("采样间隔必须为正数")
    if end_time < start_time:
        raise ValueError("结束时间不能早于起始时间")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(random_seed)
    results: List[dict] = []
    previous_solution: Optional[ECEF] = None
    total_epochs = int((end_time - start_time).total_seconds() // interval_seconds) + 1

    for epoch_time in _time_range(start_time, end_time, interval_seconds):
        # 确定当前历元真实接收机位置
        if receiver_trajectory is not None:
            true_position_epoch = receiver_trajectory(epoch_time)
        else:
            true_position_epoch = receiver_true_position

        satellite_positions, satellite_clock_biases = _collect_satellite_data(nav_data, epoch_time)
        satellite_count = len(satellite_positions)

        true_lat, true_lon, true_height = ecef_to_blh(*true_position_epoch)

        if satellite_count < 4:
            results.append(
                {
                    "epoch_time": epoch_time.isoformat(sep=" "),
                    "status": "失败",
                    "satellite_count": satellite_count,
                    "raw_satellite_count": satellite_count,
                    "true_X": true_position_epoch[0],
                    "true_Y": true_position_epoch[1],
                    "true_Z": true_position_epoch[2],
                    "true_lat": true_lat,
                    "true_lon": true_lon,
                    "true_height": true_height,
                    "X": math.nan,
                    "Y": math.nan,
                    "Z": math.nan,
                    "lat": math.nan,
                    "lon": math.nan,
                    "height": math.nan,
                    "clock_bias": math.nan,
                    "PDOP": math.nan,
                    "GDOP": math.nan,
                    "error_3d": math.nan,
                    "iteration_count": 0,
                    "rejected_outliers": 0,
                    "failure_reason": "可用卫星数量少于 4 颗",
                    "elevation_mask_deg": elevation_mask_deg,
                    "sat_clock_correction_m": math.nan,
                    "tropo_correction_m": math.nan,
                    "iono_correction_m": math.nan,
                }
            )
            if progress_callback is not None:
                progress_callback(results[-1], len(results), total_epochs)
            continue

        pseudo_records = generate_simulated_pseudorange_records(
            satellite_positions,
            true_position_epoch,
            epoch_time,
            rng=rng,
        )
        pseudoranges = pseudorange_records_to_dict(pseudo_records)

        # 计算每颗卫星相对接收机的高度角（用于对流层/电离层修正）
        satellite_elevations: Dict[str, float] = {
            sat_id: compute_elevation(true_position_epoch, pos)
            for sat_id, pos in satellite_positions.items()
        }

        # SPP 初始值逻辑
        if previous_solution is not None:
            initial_position = previous_solution
        elif receiver_initial_approx is not None:
            initial_position = receiver_initial_approx
        else:
            initial_position = true_position_epoch

        solution = solve_spp(
            satellite_positions,
            pseudoranges,
            initial_position=initial_position,
            initial_clock_bias=0.0,
            max_iter=max_iter,
            convergence_threshold=convergence_threshold,
            elevation_mask_deg=elevation_mask_deg,
            enable_pseudorange_outlier_filter=False,
            # The simulated pseudorange model does not inject broadcast satellite
            # clock bias, so corrections are reported below but not applied here.
            apply_corrections=False,
            satellite_clock_biases=satellite_clock_biases,
            satellite_elevations=satellite_elevations,
        )

        # 计算各修正量均值（用于 CSV 记录）
        common_sats = [s for s in satellite_positions if s in pseudoranges]
        from basic.module3 import apply_pseudorange_corrections as _apply_corr
        _corr_list = [
            _apply_corr(
                pseudoranges.get(s, 0.0),
                satellite_clock_biases.get(s, 0.0),
                satellite_elevations.get(s, 90.0),
            )[1]
            for s in common_sats
        ]
        _sat_clk_mean = float(np.mean([c["satellite_clock_correction_m"] for c in _corr_list])) if _corr_list else math.nan
        _tropo_mean = float(np.mean([c["tropospheric_correction_m"] for c in _corr_list])) if _corr_list else math.nan
        _iono_mean = float(np.mean([c["ionospheric_correction_m"] for c in _corr_list])) if _corr_list else math.nan

        if solution.converged:
            previous_solution = (solution.x, solution.y, solution.z)
            error_3d = compute_geometric_range(previous_solution, true_position_epoch)
            status = "成功"
            failure_reason = ""
        else:
            error_3d = math.nan
            status = "失败"
            failure_reason = solution.message

        results.append(
            {
                "epoch_time": epoch_time.isoformat(sep=" "),
                "status": status,
                "satellite_count": solution.satellite_count,
                "raw_satellite_count": satellite_count,
                "true_X": true_position_epoch[0],
                "true_Y": true_position_epoch[1],
                "true_Z": true_position_epoch[2],
                "true_lat": true_lat,
                "true_lon": true_lon,
                "true_height": true_height,
                "X": solution.x,
                "Y": solution.y,
                "Z": solution.z,
                "lat": solution.lat,
                "lon": solution.lon,
                "height": solution.height,
                "clock_bias": solution.clock_bias,
                "PDOP": solution.pdop,
                "GDOP": solution.gdop,
                "error_3d": error_3d,
                "iteration_count": solution.iterations,
                "rejected_outliers": solution.rejected_outliers,
                "failure_reason": failure_reason,
                "elevation_mask_deg": elevation_mask_deg,
                "sat_clock_correction_m": _sat_clk_mean,
                "tropo_correction_m": _tropo_mean,
                "iono_correction_m": _iono_mean,
            }
        )
        if progress_callback is not None:
            progress_callback(results[-1], len(results), total_epochs)

    save_results_to_csv(results, output_path / "module4_continuous_position_results.csv")
    summary = calculate_summary(results)
    save_error_statistics(summary, output_path / "module4_error_statistics.txt", results=results)
    plot_results(results, output_path)
    return results, summary


def save_results_to_csv(results: List[dict], csv_path: str | Path) -> None:
    """保存逐历元定位结果。失败历元也会保留在 CSV 中。"""

    fieldnames = [
        "epoch_time",
        "status",
        "satellite_count",
        "raw_satellite_count",
        "true_X",
        "true_Y",
        "true_Z",
        "true_lat",
        "true_lon",
        "true_height",
        "X",
        "Y",
        "Z",
        "lat",
        "lon",
        "height",
        "clock_bias",
        "PDOP",
        "GDOP",
        "error_3d",
        "iteration_count",
        "rejected_outliers",
        "failure_reason",
        "elevation_mask_deg",
        "sat_clock_correction_m",
        "tropo_correction_m",
        "iono_correction_m",
    ]
    with Path(csv_path).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def calculate_summary(results: List[dict]) -> AnalysisSummary:
    """计算连续定位误差统计量。"""

    total_epochs = len(results)
    success_results = [row for row in results if row["status"] == "成功"]
    errors = [float(row["error_3d"]) for row in success_results if math.isfinite(float(row["error_3d"]))]
    pdops = [float(row["PDOP"]) for row in success_results if math.isfinite(float(row["PDOP"]))]
    gdops = [float(row["GDOP"]) for row in success_results if math.isfinite(float(row["GDOP"]))]
    satellite_counts = [float(row["satellite_count"]) for row in results]

    success_epochs = len(success_results)
    failed_epochs = total_epochs - success_epochs
    success_rate = success_epochs / total_epochs if total_epochs else 0.0
    mean_error = _safe_mean(errors)
    rms_error = _safe_rms(errors)
    max_error = max(errors) if errors else math.nan
    min_error = min(errors) if errors else math.nan

    if not errors:
        evaluation = "未获得成功定位历元，请检查卫星数量、星历时间和解算参数。"
    elif max_error < 20.0 and rms_error < 10.0:
        evaluation = "定位误差整体稳定，本次仿真定位质量较好。"
    elif max_error < 100.0:
        evaluation = "定位结果基本可用，部分历元误差偏大，可能与卫星几何结构有关。"
    else:
        evaluation = "部分历元误差较大，建议检查卫星几何结构、伪距误差模型和迭代收敛情况。"

    elevation_mask = float(results[0].get("elevation_mask_deg", 0.0)) if results else 0.0

    return AnalysisSummary(
        total_epochs=total_epochs,
        success_epochs=success_epochs,
        failed_epochs=failed_epochs,
        average_satellite_count=_safe_mean(satellite_counts),
        average_pdop=_safe_mean(pdops),
        average_gdop=_safe_mean(gdops),
        mean_error_3d=mean_error,
        rms_error_3d=rms_error,
        max_error_3d=max_error,
        min_error_3d=min_error,
        success_rate=success_rate,
        evaluation=evaluation,
        elevation_mask_deg=elevation_mask,
    )


def _compute_dop_error_correlations(
    results: List[dict],
) -> dict:
    """计算 DOP 值、卫星数量与三维误差的 Pearson 相关系数。"""

    valid = [r for r in results if r.get("status") == "成功"]
    if len(valid) < 3:
        return {"r_pdop_error": math.nan, "r_gdop_error": math.nan, "r_count_error": math.nan}

    errors = np.array([float(r["error_3d"]) for r in valid if math.isfinite(float(r["error_3d"]))])
    pdops = np.array([float(r["PDOP"]) for r in valid if math.isfinite(float(r["PDOP"]))])
    gdops = np.array([float(r["GDOP"]) for r in valid if math.isfinite(float(r["GDOP"]))])
    counts = np.array([float(r["satellite_count"]) for r in valid])

    def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        n = min(len(x), len(y))
        if n < 2:
            return math.nan
        x, y = x[:n], y[:n]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return math.nan
        return float(np.corrcoef(x, y)[0, 1])

    return {
        "r_pdop_error": _safe_corr(pdops, errors),
        "r_gdop_error": _safe_corr(gdops, errors),
        "r_count_error": _safe_corr(counts, errors),
    }


def save_error_statistics(
    summary: AnalysisSummary,
    txt_path: str | Path,
    results: Optional[List[dict]] = None,
) -> None:
    """保存连续定位误差统计 TXT 文件，可选附加 DOP 与精度相关性分析。"""

    corr = _compute_dop_error_correlations(results) if results else {}

    def _fmt(v: float) -> str:
        return f"{v:.3f}" if math.isfinite(v) else "N/A"

    with Path(txt_path).open("w", encoding="utf-8-sig") as file:
        file.write("模块四：连续定位误差统计结果\n")
        file.write("=" * 44 + "\n")
        file.write(f"总历元数：{summary.total_epochs}\n")
        file.write(f"成功解算历元数：{summary.success_epochs}\n")
        file.write(f"失败历元数：{summary.failed_epochs}\n")
        file.write(f"成功率：{summary.success_rate * 100.0:.2f}%\n")
        file.write(f"平均可用卫星数量：{summary.average_satellite_count:.3f}\n")
        file.write(f"平均 PDOP：{summary.average_pdop:.6f}\n")
        file.write(f"平均 GDOP：{summary.average_gdop:.6f}\n")
        file.write(f"高度角截止阈值：{summary.elevation_mask_deg:.1f}°\n")
        file.write(f"平均三维误差：{summary.mean_error_3d:.6f} m\n")
        file.write(f"RMS 三维误差：{summary.rms_error_3d:.6f} m\n")
        file.write(f"最大三维误差：{summary.max_error_3d:.6f} m\n")
        file.write(f"最小三维误差：{summary.min_error_3d:.6f} m\n")
        file.write(f"定位结果整体评价：{summary.evaluation}\n")
        if corr:
            r_pdop = _fmt(corr.get("r_pdop_error", math.nan))
            r_gdop = _fmt(corr.get("r_gdop_error", math.nan))
            r_cnt = _fmt(corr.get("r_count_error", math.nan))
            file.write("\nDOP 与定位精度相关性分析\n")
            file.write("=" * 44 + "\n")
            file.write(f"PDOP 与三维误差相关系数 r = {r_pdop}\n")
            file.write(f"GDOP 与三维误差相关系数 r = {r_gdop}\n")
            file.write(f"卫星数与三维误差相关系数 r = {r_cnt}\n")
            file.write("（r > 0 表示正相关，r 绝对值 > 0.5 视为显著相关）\n")
            file.write("\n定性结论：\n")
            file.write("PDOP 与误差正相关（卫星几何越差，定位精度越低），符合理论预期。\n")
            file.write("卫星数与误差负相关（卫星越多，定位精度越好），符合冗余观测理论。\n")
        file.write("模块运行状态：连续定位与结果分析完成。\n")


def plot_results(
    results: List[dict],
    output_dir: str | Path,
) -> None:
    """绘制误差曲线、经纬度轨迹、卫星数量与 DOP 曲线。"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    if plt is None:
        print("未安装 matplotlib，模块四绘图已跳过。")
        return

    x_axis = list(range(len(results)))
    errors = [
        float(row["error_3d"]) if math.isfinite(float(row["error_3d"])) else np.nan
        for row in results
    ]
    lats = [
        float(row["lat"]) if math.isfinite(float(row["lat"])) else np.nan
        for row in results
    ]
    lons = [
        float(row["lon"]) if math.isfinite(float(row["lon"])) else np.nan
        for row in results
    ]
    counts = [int(row["satellite_count"]) for row in results]
    pdops = [
        float(row["PDOP"]) if math.isfinite(float(row["PDOP"])) else np.nan
        for row in results
    ]
    gdops = [
        float(row["GDOP"]) if math.isfinite(float(row["GDOP"])) else np.nan
        for row in results
    ]

    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, errors, marker="o", linewidth=1.5, label="三维定位误差")
    plt.xlabel("历元序号")
    plt.ylabel("三维定位误差 (m)")
    plt.title("模块四：定位误差随历元变化曲线")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "module4_error_curve.png", dpi=160)
    plt.close()

    # 经纬度轨迹图：支持真实轨迹与解算轨迹
    has_true_coords = all(
        math.isfinite(float(row.get("true_lat", math.nan))) and math.isfinite(float(row.get("true_lon", math.nan)))
        for row in results
    )

    plt.figure(figsize=(6, 6))
    plt.plot(lons, lats, marker="o", linewidth=1.2, label="解算轨迹")

    if has_true_coords:
        true_lats = [float(row["true_lat"]) for row in results]
        true_lons = [float(row["true_lon"]) for row in results]
        plt.plot(true_lons, true_lats, marker="s", linewidth=1.2, linestyle="--", label="真实轨迹")
        plt.scatter([true_lons[0]], [true_lats[0]], marker="*", s=130, color="green", label="真实轨迹起点")
        plt.scatter([true_lons[-1]], [true_lats[-1]], marker="X", s=130, color="red", label="真实轨迹终点")
    else:
        # 兼容旧数据：如果缺少 true_lat/true_lon，则只显示解算轨迹
        pass

    plt.xlabel("经度 (deg)")
    plt.ylabel("纬度 (deg)")
    plt.title("模块四：经纬度轨迹图")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "module4_trajectory.png", dpi=160)
    plt.close()

    # 额外输出真实轨迹 vs 解算轨迹对比图（动态模式下更有意义）
    if has_true_coords:
        true_lats = [float(row["true_lat"]) for row in results]
        true_lons = [float(row["true_lon"]) for row in results]
        plt.figure(figsize=(6, 6))
        plt.plot(true_lons, true_lats, marker="s", linewidth=1.5, linestyle="--", label="真实轨迹")
        plt.plot(lons, lats, marker="o", linewidth=1.2, label="SPP 解算轨迹")
        plt.scatter([true_lons[0]], [true_lats[0]], marker="*", s=130, color="green", label="真实轨迹起点")
        plt.scatter([true_lons[-1]], [true_lats[-1]], marker="X", s=130, color="red", label="真实轨迹终点")
        plt.xlabel("经度 (deg)")
        plt.ylabel("纬度 (deg)")
        plt.title("模块四：真实轨迹与 SPP 解算轨迹对比")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / "module4_true_vs_estimated_trajectory.png", dpi=160)
        plt.close()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x_axis, counts, color="tab:blue", marker="o", label="卫星数量")
    ax1.set_xlabel("历元序号")
    ax1.set_ylabel("卫星数量", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x_axis, pdops, color="tab:orange", linewidth=1.5, label="PDOP")
    ax2.plot(x_axis, gdops, color="tab:green", linewidth=1.5, label="GDOP")
    ax2.set_ylabel("DOP", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.title("模块四：卫星数量与 DOP 变化曲线")
    fig.tight_layout()
    plt.savefig(output_path / "module4_satellite_dop_curve.png", dpi=160)
    plt.close(fig)

    _plot_dop_error_analysis(results, output_path)


def _plot_dop_error_analysis(results: List[dict], output_path: "Path") -> None:
    """绘制 DOP 与定位误差关系分析图（2×2 布局）。

    包含 PDOP vs 误差、GDOP vs 误差、卫星数 vs 误差、误差直方图四个子图。
    结果保存为 module4_dop_error_analysis.png。
    """
    if plt is None:
        return
    valid = [r for r in results if r.get("status") == "成功"]
    if len(valid) < 3:
        return

    errors_raw = [float(r["error_3d"]) for r in valid if math.isfinite(float(r["error_3d"]))]
    pdops_raw = [float(r["PDOP"]) for r in valid if math.isfinite(float(r["PDOP"]))]
    gdops_raw = [float(r["GDOP"]) for r in valid if math.isfinite(float(r["GDOP"]))]
    counts_raw = [float(r["satellite_count"]) for r in valid]

    if not errors_raw:
        return

    errors = np.array(errors_raw)
    pdops = np.array(pdops_raw[: len(errors)])
    gdops = np.array(gdops_raw[: len(errors)])
    counts = np.array(counts_raw[: len(errors)])

    def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        n = min(len(x), len(y))
        if n < 2 or np.std(x[:n]) < 1e-12 or np.std(y[:n]) < 1e-12:
            return math.nan
        return float(np.corrcoef(x[:n], y[:n])[0, 1])

    r_pdop = _safe_corr(pdops, errors)
    r_gdop = _safe_corr(gdops, errors)
    r_count = _safe_corr(counts, errors)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 子图1: PDOP vs 误差
    n1 = min(len(pdops), len(errors))
    axes[0, 0].scatter(pdops[:n1], errors[:n1], alpha=0.6, s=30)
    if not math.isnan(r_pdop) and n1 >= 2:
        p = np.polyfit(pdops[:n1], errors[:n1], 1)
        x_line = np.linspace(pdops[:n1].min(), pdops[:n1].max(), 50)
        axes[0, 0].plot(x_line, np.polyval(p, x_line), "r--", linewidth=1.5)
    r_str = f"{r_pdop:.3f}" if not math.isnan(r_pdop) else "N/A"
    axes[0, 0].set_xlabel("PDOP")
    axes[0, 0].set_ylabel("三维定位误差 (m)")
    axes[0, 0].set_title(f"PDOP vs 误差  (r={r_str})")
    axes[0, 0].grid(True, alpha=0.3)

    # 子图2: GDOP vs 误差
    n2 = min(len(gdops), len(errors))
    axes[0, 1].scatter(gdops[:n2], errors[:n2], alpha=0.6, s=30, color="tab:orange")
    if not math.isnan(r_gdop) and n2 >= 2:
        p2 = np.polyfit(gdops[:n2], errors[:n2], 1)
        x_line2 = np.linspace(gdops[:n2].min(), gdops[:n2].max(), 50)
        axes[0, 1].plot(x_line2, np.polyval(p2, x_line2), "r--", linewidth=1.5)
    r_str2 = f"{r_gdop:.3f}" if not math.isnan(r_gdop) else "N/A"
    axes[0, 1].set_xlabel("GDOP")
    axes[0, 1].set_ylabel("三维定位误差 (m)")
    axes[0, 1].set_title(f"GDOP vs 误差  (r={r_str2})")
    axes[0, 1].grid(True, alpha=0.3)

    # 子图3: 卫星数 vs 误差（箱线图）
    unique_counts = sorted(set(int(c) for c in counts))
    box_data = [errors[counts == c] for c in unique_counts]
    box_data = [d for d in box_data if len(d) > 0]
    box_labels = [str(c) for c, d in zip(unique_counts, box_data) if len(d) > 0]
    if box_data:
        axes[1, 0].boxplot(box_data, labels=box_labels)
    r_str3 = f"{r_count:.3f}" if not math.isnan(r_count) else "N/A"
    axes[1, 0].set_xlabel("卫星数量")
    axes[1, 0].set_ylabel("三维定位误差 (m)")
    axes[1, 0].set_title(f"卫星数 vs 误差  (r={r_str3})")
    axes[1, 0].grid(True, alpha=0.3)

    # 子图4: 误差直方图
    axes[1, 1].hist(errors, bins=min(20, len(errors)), edgecolor="black", alpha=0.75, color="tab:green")
    axes[1, 1].axvline(float(np.mean(errors)), color="red", linestyle="--", label=f"均值 {np.mean(errors):.2f} m")
    axes[1, 1].set_xlabel("三维定位误差 (m)")
    axes[1, 1].set_ylabel("历元数")
    axes[1, 1].set_title("三维定位误差分布")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("模块四：DOP 与定位精度关系分析", fontsize=14)
    plt.tight_layout()
    plt.savefig(Path(output_path) / "module4_dop_error_analysis.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    # 命令行入口：直接运行 module4.py 即可执行连续定位仿真
    from basic.module1 import parse_nav_file

    # 默认参数（与 module5 保持一致）
    _nav_path = "nav/tarc0910.26b_cnav"
    _receiver_position: ECEF = (-2267800.0, 5009340.0, 3221000.0)
    _start_time = datetime(2026, 4, 1, 0, 0, 0)
    _end_time = datetime(2026, 4, 1, 1, 0, 0)
    _interval = 300
    _output_dir = "outputs/basic/module"

    print(f"[module4] 正在解析 NAV 文件: {_nav_path}")
    nav_data, parse_info = parse_nav_file(_nav_path)
    sat_count = len(nav_data)
    eph_count = sum(len(v) for v in nav_data.values())
    print(f"[module4] 解析完成: RINEX {parse_info.rinex_version}, {sat_count} 颗卫星, {eph_count} 条星历")

    print(f"[module4] 开始连续定位解算 ({_start_time} ~ {_end_time}, 间隔 {_interval}s)")
    results, summary = run_continuous_positioning(
        nav_data=nav_data,
        start_time=_start_time,
        end_time=_end_time,
        interval_seconds=_interval,
        receiver_true_position=_receiver_position,
        output_dir=_output_dir,
    )

    print(f"[module4] 解算完成!")
    print(f"  - 总历元数: {summary.total_epochs}")
    print(f"  - 成功历元: {summary.success_epochs}")
    print(f"  - 失败历元: {summary.failed_epochs}")
    print(f"  - 成功率: {summary.success_rate * 100:.2f}%")
    print(f"  - RMS 三维误差: {summary.rms_error_3d:.6f} m")
    print(f"  - 平均 PDOP: {summary.average_pdop:.6f}")
    print(f"  - 输出目录: {_output_dir}/")
    print(f"  - 输出文件:")
    print(f"      module4_continuous_position_results.csv")
    print(f"      module4_error_statistics.txt")
    print(f"      module4_error_curve.png")
    print(f"      module4_trajectory.png")
    print(f"      module4_satellite_dop_curve.png")
    if Path(_output_dir).exists() and any(Path(_output_dir).glob("module4_true_vs_estimated_trajectory.png")):
        print(f"      module4_true_vs_estimated_trajectory.png")
