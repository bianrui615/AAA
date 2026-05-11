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
    """为当前历元选择健康星历，并计算卫星 ECEF 坐标。

    底层实现调用 module1 的 compute_satellite_position。
    """

    satellite_positions: Dict[str, ECEF] = {}
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
        except Exception:
            continue
    return satellite_positions


def run_continuous_positioning(
    nav_data: Dict[str, List[BroadcastEphemeris]],
    start_time: datetime,
    end_time: datetime,
    interval_seconds: int,
    receiver_true_position: ECEF,
    output_dir: str | Path = "outputs/basic",
    random_seed: Optional[int] = 2026,
    max_iter: int = 10,
    convergence_threshold: float = 1e-4,
    elevation_mask_deg: float = 0.0,
    progress_callback: Optional[Callable[[dict, int, int], None]] = None,
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
        satellite_positions = _collect_satellite_positions(nav_data, epoch_time)
        satellite_count = len(satellite_positions)

        if satellite_count < 4:
            results.append(
                {
                    "epoch_time": epoch_time.isoformat(sep=" "),
                    "status": "失败",
                    "satellite_count": satellite_count,
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
                    "failure_reason": "可用卫星数量少于 4 颗",
                    "elevation_mask_deg": elevation_mask_deg,
                }
            )
            if progress_callback is not None:
                progress_callback(results[-1], len(results), total_epochs)
            continue

        pseudo_records = generate_simulated_pseudorange_records(
            satellite_positions,
            receiver_true_position,
            epoch_time,
            rng=rng,
        )
        pseudoranges = pseudorange_records_to_dict(pseudo_records)
        solution = solve_spp(
            satellite_positions,
            pseudoranges,
            initial_position=previous_solution if previous_solution is not None else receiver_true_position,
            initial_clock_bias=0.0,
            max_iter=max_iter,
            convergence_threshold=convergence_threshold,
            elevation_mask_deg=elevation_mask_deg,
            enable_pseudorange_outlier_filter=False,
        )

        if solution.converged:
            previous_solution = (solution.x, solution.y, solution.z)
            error_3d = compute_geometric_range(previous_solution, receiver_true_position)
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
            }
        )
        if progress_callback is not None:
            progress_callback(results[-1], len(results), total_epochs)

    save_results_to_csv(results, output_path / "module4_continuous_position_results.csv")
    summary = calculate_summary(results)
    save_error_statistics(summary, output_path / "module4_error_statistics.txt")
    plot_results(results, output_path, receiver_true_position)
    return results, summary


def save_results_to_csv(results: List[dict], csv_path: str | Path) -> None:
    """保存逐历元定位结果。失败历元也会保留在 CSV 中。"""

    fieldnames = [
        "epoch_time",
        "status",
        "satellite_count",
        "raw_satellite_count",
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


def save_error_statistics(summary: AnalysisSummary, txt_path: str | Path) -> None:
    """保存连续定位误差统计 TXT 文件。"""

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
        file.write("模块运行状态：连续定位与结果分析完成。\n")


def plot_results(
    results: List[dict],
    output_dir: str | Path,
    receiver_true_position: ECEF,
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

    true_lat, true_lon, _ = ecef_to_blh(*receiver_true_position)
    plt.figure(figsize=(6, 6))
    plt.plot(lons, lats, marker="o", linewidth=1.2, label="解算轨迹")
    plt.scatter([true_lon], [true_lat], marker="*", s=130, label="真实接收机位置")
    plt.xlabel("经度 (deg)")
    plt.ylabel("纬度 (deg)")
    plt.title("模块四：经纬度轨迹图")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "module4_trajectory.png", dpi=160)
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
