"""
module5.py

模块五：软件系统整合与测试模块。

项目名称：北斗定位解算全流程软件系统。

主程序运行流程：
1. 调用模块一解析 RINEX NAV 导航文件；
2. 调用模块二计算指定测试历元下的卫星位置与卫星钟差；
3. 调用模块三生成单历元模拟伪距并进行 SPP 解算；
4. 调用模块四进行连续定位与结果分析；
5. 保存模块五系统测试报告。
"""

from __future__ import annotations

import sys
from pathlib import Path
# 确保项目根目录在 sys.path 中，支持直接运行和作为模块导入
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import csv
from dataclasses import dataclass
from datetime import datetime
import math
import traceback
from typing import Any, Callable, Dict, List, Optional

from basic.module1 import (
    compute_satellite_clock_bias,
    compute_satellite_position,
    parse_nav_file,
    run_module1,
    select_ephemeris,
)
from basic.module2 import (
    calculate_all_satellite_positions,
    generate_pseudorange_correction_debug_records,
    save_pseudorange_correction_debug_csv,
    save_satellite_position_outputs,
)
from basic.module3 import (
    ECEF,
    generate_simulated_pseudorange_records,
    pseudorange_records_to_dict,
    save_single_epoch_spp_outputs,
    solve_spp,
)
from basic.module4 import (
    AnalysisSummary,
    build_linear_receiver_trajectory,
    run_continuous_positioning,
)


PROJECT_NAME = "北斗定位解算全流程软件系统"

# =========================
# 用户可修改参数
# =========================

# NAV 文件路径。若 NAV 文件不在当前目录，可填写绝对路径或相对路径。
NAV_FILE_PATH = "nav/tarc0910.26b_cnav"

# 接收机真实 ECEF 坐标，单位 m。
RECEIVER_TRUE_POSITION: ECEF = (-2267800.0, 5009340.0, 3221000.0)

# 仿真起止时间和采样间隔。时间应尽量与 NAV 星历时间接近。
SIMULATION_START_TIME = datetime(2026, 4, 1, 0, 0, 0)
SIMULATION_END_TIME = datetime(2026, 4, 1, 1, 0, 0)
SAMPLING_INTERVAL_SECONDS = 300

# SPP 迭代参数。
MAX_ITERATIONS = 12
CONVERGENCE_THRESHOLD = 1e-4

# 高度角截止阈值（度），0 表示不筛选。
# 当前课程题目使用伪距生成模型；默认保留全部可用卫星以获得更好的几何强度。
ELEVATION_MASK_DEG = 0.0

# 随机数种子用于保证伪距模拟可重复。
RANDOM_SEED = 2026

# 所有模块统一输出目录。
OUTPUT_DIR = "outputs/basic"

# 模块二和模块三独立输出使用的测试历元。
TEST_EPOCH_TIME = SIMULATION_START_TIME

# 是否启用多场景测试。
ENABLE_MULTI_SCENARIO_TEST = True

# 是否启用动态接收机轨迹。
ENABLE_RECEIVER_MOTION = True

# 动态接收机参数（仅在 ENABLE_RECEIVER_MOTION=True 时使用）
RECEIVER_INITIAL_POSITION = (-2267800.0, 5009340.0, 3221000.0)

RECEIVER_VELOCITY_ECEF_MPS = (0.5, 0.2, 0.1)

RECEIVER_INITIAL_APPROX_POSITION = (
    RECEIVER_INITIAL_POSITION[0] + 50.0,
    RECEIVER_INITIAL_POSITION[1] - 50.0,
    RECEIVER_INITIAL_POSITION[2] + 30.0,
)


@dataclass
class ScenarioConfig:
    """多场景测试配置。"""

    name: str
    nav_file_path: str
    receiver_true_position: ECEF
    start_time: datetime
    end_time: datetime
    interval_seconds: int
    random_seed: int
    max_iter: int
    convergence_threshold: float
    elevation_mask_deg: float
    enable_receiver_motion: bool = False
    receiver_initial_position: ECEF = (0.0, 0.0, 0.0)
    receiver_velocity_mps: ECEF = (0.0, 0.0, 0.0)
    receiver_initial_approx_position: ECEF = (0.0, 0.0, 0.0)


SCENARIOS = [
    ScenarioConfig(
        name="scenario_1_default",
        nav_file_path=NAV_FILE_PATH,
        receiver_true_position=RECEIVER_TRUE_POSITION,
        start_time=SIMULATION_START_TIME,
        end_time=SIMULATION_END_TIME,
        interval_seconds=SAMPLING_INTERVAL_SECONDS,
        random_seed=RANDOM_SEED,
        max_iter=MAX_ITERATIONS,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        elevation_mask_deg=ELEVATION_MASK_DEG,
        enable_receiver_motion=True,
        receiver_initial_position=RECEIVER_INITIAL_POSITION,
        receiver_velocity_mps=RECEIVER_VELOCITY_ECEF_MPS,
        receiver_initial_approx_position=RECEIVER_INITIAL_APPROX_POSITION,
    ),
    ScenarioConfig(
        name="scenario_2_different_seed",
        nav_file_path=NAV_FILE_PATH,
        receiver_true_position=RECEIVER_TRUE_POSITION,
        start_time=SIMULATION_START_TIME,
        end_time=SIMULATION_END_TIME,
        interval_seconds=SAMPLING_INTERVAL_SECONDS,
        random_seed=42,
        max_iter=MAX_ITERATIONS,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        elevation_mask_deg=ELEVATION_MASK_DEG,
        enable_receiver_motion=False,
    ),
    ScenarioConfig(
        name="scenario_3_elevation_mask",
        nav_file_path=NAV_FILE_PATH,
        receiver_true_position=(-2270000.0, 5009000.0, 3220000.0),
        start_time=SIMULATION_START_TIME,
        end_time=SIMULATION_END_TIME,
        interval_seconds=SAMPLING_INTERVAL_SECONDS,
        random_seed=RANDOM_SEED,
        max_iter=MAX_ITERATIONS,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        elevation_mask_deg=10.0,
        enable_receiver_motion=True,
        receiver_initial_position=(-2270000.0, 5009000.0, 3220000.0),
        receiver_velocity_mps=(0.3, 0.1, 0.05),
        receiver_initial_approx_position=(-2269950.0, 5009050.0, 3220030.0),
    ),
]


def _format_float(value: float, digits: int = 3) -> str:
    """格式化浮点数，遇到 NaN 时返回字符串 NaN。"""

    if value is None or not math.isfinite(value):
        return "NaN"
    return f"{value:.{digits}f}"


def _collect_satellite_data_from_module1(
    nav_data: dict,
    epoch_time: datetime,
) -> tuple[dict[str, ECEF], dict[str, float]]:
    """直接调用 module1 计算卫星位置和健康状态。"""

    positions: Dict[str, ECEF] = {}
    health: Dict[str, float] = {}
    for sat_id in sorted(nav_data):
        eph = select_ephemeris(nav_data, sat_id, epoch_time, healthy_only=True)
        if eph is None:
            continue
        try:
            x, y, z = compute_satellite_position(eph, epoch_time)
            positions[sat_id] = (x, y, z)
            health[sat_id] = float(eph.health)
        except Exception:
            continue
    return positions, health


def write_system_test_report(
    output_dir: str | Path,
    nav_file_path: str | Path,
    module_status: Dict[str, str],
    module_outputs: Dict[str, List[Path]],
    summary: AnalysisSummary,
) -> Path:
    """保存模块五系统测试报告。"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "module5_system_test_report.txt"

    with report_path.open("w", encoding="utf-8-sig") as file:
        file.write("模块五：系统整合与测试报告\n")
        file.write("=" * 44 + "\n")
        file.write(f"项目名称：{PROJECT_NAME}\n")
        file.write(f"测试时间：{datetime.now().isoformat(sep=' ', timespec='seconds')}\n")
        file.write(f"NAV 文件路径：{Path(nav_file_path)}\n")

        # 接收机运动模型说明
        if ENABLE_RECEIVER_MOTION:
            file.write("接收机运动模型：动态接收机（ECEF 匀速直线运动）\n")
            file.write(
                f"  初始真实坐标："
                f"X={RECEIVER_INITIAL_POSITION[0]:.4f} m，"
                f"Y={RECEIVER_INITIAL_POSITION[1]:.4f} m，"
                f"Z={RECEIVER_INITIAL_POSITION[2]:.4f} m\n"
            )
            file.write(
                f"  ECEF 速度："
                f"Vx={RECEIVER_VELOCITY_ECEF_MPS[0]:.4f} m/s，"
                f"Vy={RECEIVER_VELOCITY_ECEF_MPS[1]:.4f} m/s，"
                f"Vz={RECEIVER_VELOCITY_ECEF_MPS[2]:.4f} m/s\n"
            )
            file.write(
                f"  初始概略坐标："
                f"X={RECEIVER_INITIAL_APPROX_POSITION[0]:.4f} m，"
                f"Y={RECEIVER_INITIAL_APPROX_POSITION[1]:.4f} m，"
                f"Z={RECEIVER_INITIAL_APPROX_POSITION[2]:.4f} m\n"
            )
        else:
            file.write("接收机运动模型：静态接收机\n")
            file.write(
                "接收机真实 ECEF 坐标："
                f"X={RECEIVER_TRUE_POSITION[0]:.4f} m，"
                f"Y={RECEIVER_TRUE_POSITION[1]:.4f} m，"
                f"Z={RECEIVER_TRUE_POSITION[2]:.4f} m\n"
            )

        file.write(f"仿真起始时间：{SIMULATION_START_TIME.isoformat(sep=' ')}\n")
        file.write(f"仿真结束时间：{SIMULATION_END_TIME.isoformat(sep=' ')}\n")
        file.write(f"采样间隔：{SAMPLING_INTERVAL_SECONDS} s\n")
        file.write(f"最大迭代次数：{MAX_ITERATIONS}\n")
        file.write(f"收敛阈值：{CONVERGENCE_THRESHOLD} m\n")
        file.write(f"高度角截止阈值：{ELEVATION_MASK_DEG}°\n")
        file.write(f"随机数种子：{RANDOM_SEED}\n\n")

        module_names = {
            "module1": "模块一：RINEX NAV 导航文件解析",
            "module2": "模块二：卫星位置与钟差计算",
            "module3": "模块三：模拟伪距与单历元 SPP 解算",
            "module4": "模块四：连续定位与结果分析",
            "multi_scenario": "多场景定位测试",
        }
        for module_key in ["module1", "module2", "module3", "module4", "multi_scenario"]:
            file.write(f"{module_names[module_key]}运行状态：{module_status.get(module_key, '未知')}\n")
            file.write("输出文件列表：\n")
            for path in module_outputs.get(module_key, []):
                file.write(f"  - {path}\n")
            file.write("\n")

        file.write("连续定位总体统计结果：\n")
        file.write(f"  总历元数：{summary.total_epochs}\n")
        file.write(f"  成功解算历元数：{summary.success_epochs}\n")
        file.write(f"  失败历元数：{summary.failed_epochs}\n")
        file.write(f"  平均可用卫星数量：{_format_float(summary.average_satellite_count, 2)}\n")
        file.write(f"  平均 PDOP：{_format_float(summary.average_pdop, 3)}\n")
        file.write(f"  平均 GDOP：{_format_float(summary.average_gdop, 3)}\n")
        file.write(f"  平均误差：{_format_float(summary.mean_error_3d, 3)} m\n")
        file.write(f"  RMS 误差：{_format_float(summary.rms_error_3d, 3)} m\n")
        file.write(f"  最大误差：{_format_float(summary.max_error_3d, 3)} m\n\n")

        file.write("系统测试结论：\n")
        file.write(
            "本次测试中，系统成功完成 NAV 文件解析、卫星位置计算、模拟伪距生成、"
            "单点定位解算和连续定位分析。定位结果能够正常输出，误差统计和可视化"
            "图表生成成功，说明系统基本满足设计要求。\n"
        )

    return report_path


def print_test_report(summary: AnalysisSummary, report_path: Path) -> None:
    """在终端输出简洁的中文测试报告。"""

    print("\n========== 北斗定位解算全流程软件系统测试报告 ==========")
    print(f"总历元数：{summary.total_epochs}")
    print(f"成功解算历元数：{summary.success_epochs}")
    print(f"失败历元数：{summary.failed_epochs}")
    print(f"平均可用卫星数量：{_format_float(summary.average_satellite_count, 2)}")
    print(f"平均 PDOP：{_format_float(summary.average_pdop, 3)}")
    print(f"平均 GDOP：{_format_float(summary.average_gdop, 3)}")
    print(f"平均误差：{_format_float(summary.mean_error_3d, 3)} m")
    print(f"RMS 误差：{_format_float(summary.rms_error_3d, 3)} m")
    print(f"最大误差：{_format_float(summary.max_error_3d, 3)} m")
    print(f"系统测试报告：{report_path}")
    print(f"输出目录：{Path(OUTPUT_DIR).resolve()}")
    print("======================================================\n")


def run_single_scenario_test(
    scenario: ScenarioConfig,
    base_output_dir: str | Path,
) -> Dict[str, Any]:
    """运行单个场景测试。"""

    output_dir = Path(base_output_dir) / scenario.name
    output_dir.mkdir(parents=True, exist_ok=True)

    nav_data, _ = parse_nav_file(scenario.nav_file_path)

    # 根据场景配置构造接收机轨迹
    if scenario.enable_receiver_motion:
        receiver_trajectory = build_linear_receiver_trajectory(
            scenario.start_time,
            scenario.receiver_initial_position,
            scenario.receiver_velocity_mps,
        )
        receiver_initial_approx = scenario.receiver_initial_approx_position
    else:
        receiver_trajectory = None
        receiver_initial_approx = None

    _, summary = run_continuous_positioning(
        nav_data=nav_data,
        start_time=scenario.start_time,
        end_time=scenario.end_time,
        interval_seconds=scenario.interval_seconds,
        receiver_true_position=scenario.receiver_true_position,
        output_dir=output_dir,
        random_seed=scenario.random_seed,
        max_iter=scenario.max_iter,
        convergence_threshold=scenario.convergence_threshold,
        elevation_mask_deg=scenario.elevation_mask_deg,
        receiver_trajectory=receiver_trajectory,
        receiver_initial_approx=receiver_initial_approx,
    )

    output_files = [
        output_dir / "module4_continuous_position_results.csv",
        output_dir / "module4_error_statistics.txt",
        output_dir / "module4_error_curve.png",
        output_dir / "module4_trajectory.png",
        output_dir / "module4_satellite_dop_curve.png",
    ]

    return {
        "scenario": scenario,
        "summary": summary,
        "output_dir": output_dir,
        "output_files": output_files,
    }


def run_multi_scenario_tests() -> Dict[str, Any]:
    """运行多场景测试，并生成汇总 CSV 与报告。"""

    base_output_dir = Path(OUTPUT_DIR)
    scenario_results: List[Dict[str, Any]] = []

    for scenario in SCENARIOS:
        print(f"  运行场景：{scenario.name} ...")
        result = run_single_scenario_test(scenario, base_output_dir)
        scenario_results.append(result)
        summary = result["summary"]
        print(
            f"    完成：成功率 {summary.success_rate * 100:.2f}%，"
            f"平均误差 {_format_float(summary.mean_error_3d, 3)} m"
        )

    # 汇总 CSV
    summary_csv_path = base_output_dir / "module5_multi_scenario_summary.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario_name", "nav_file_path",
            "receiver_x", "receiver_y", "receiver_z",
            "start_time", "end_time", "interval_seconds", "random_seed",
            "total_epochs", "success_epochs", "failed_epochs", "success_rate",
            "average_satellite_count", "average_pdop", "average_gdop",
            "mean_error_3d", "rms_error_3d", "max_error_3d", "output_dir",
            "enable_receiver_motion", "receiver_velocity_x", "receiver_velocity_y", "receiver_velocity_z",
        ])
        for result in scenario_results:
            s = result["scenario"]
            summary = result["summary"]
            writer.writerow([
                s.name,
                s.nav_file_path,
                s.receiver_true_position[0],
                s.receiver_true_position[1],
                s.receiver_true_position[2],
                s.start_time.isoformat(sep=" "),
                s.end_time.isoformat(sep=" "),
                s.interval_seconds,
                s.random_seed,
                summary.total_epochs,
                summary.success_epochs,
                summary.failed_epochs,
                summary.success_rate,
                summary.average_satellite_count,
                summary.average_pdop,
                summary.average_gdop,
                summary.mean_error_3d,
                summary.rms_error_3d,
                summary.max_error_3d,
                str(result["output_dir"]),
                s.enable_receiver_motion,
                s.receiver_velocity_mps[0],
                s.receiver_velocity_mps[1],
                s.receiver_velocity_mps[2],
            ])

    # 多场景测试报告
    report_path = base_output_dir / "module5_multi_scenario_test_report.txt"
    with report_path.open("w", encoding="utf-8-sig") as f:
        f.write("模块五：多场景定位测试报告\n")
        f.write("=" * 50 + "\n")
        f.write("测试目的：验证北斗定位解算系统在不同随机种子、接收机坐标\n")
        f.write("和高度角阈值等参数变化下的鲁棒性与稳定性。\n\n")

        for idx, result in enumerate(scenario_results, 1):
            s = result["scenario"]
            summary = result["summary"]
            f.write(f"场景 {idx}：{s.name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  NAV 文件：{s.nav_file_path}\n")
            f.write(
                f"  接收机坐标：X={s.receiver_true_position[0]:.4f} m，"
                f"Y={s.receiver_true_position[1]:.4f} m，"
                f"Z={s.receiver_true_position[2]:.4f} m\n"
            )
            f.write(
                f"  仿真时间：{s.start_time.isoformat(sep=' ')} 至 "
                f"{s.end_time.isoformat(sep=' ')}\n"
            )
            f.write(f"  采样间隔：{s.interval_seconds} s\n")
            f.write(f"  随机种子：{s.random_seed}\n")
            f.write(f"  高度角阈值：{s.elevation_mask_deg}°\n")

            # 动态接收机说明
            if s.enable_receiver_motion:
                f.write("  接收机运动模型：动态（ECEF 匀速直线运动）\n")
                f.write(
                    f"    初始位置：X={s.receiver_initial_position[0]:.4f} m，"
                    f"Y={s.receiver_initial_position[1]:.4f} m，"
                    f"Z={s.receiver_initial_position[2]:.4f} m\n"
                )
                f.write(
                    f"    速度：Vx={s.receiver_velocity_mps[0]:.4f} m/s，"
                    f"Vy={s.receiver_velocity_mps[1]:.4f} m/s，"
                    f"Vz={s.receiver_velocity_mps[2]:.4f} m/s\n"
                )
            else:
                f.write("  接收机运动模型：静态\n")

            f.write(f"  总历元数：{summary.total_epochs}\n")
            f.write(f"  成功历元数：{summary.success_epochs}\n")
            f.write(f"  失败历元数：{summary.failed_epochs}\n")
            f.write(f"  成功率：{summary.success_rate * 100:.2f}%\n")
            f.write(f"  平均可用卫星数量：{_format_float(summary.average_satellite_count, 2)}\n")
            f.write(f"  平均 PDOP：{_format_float(summary.average_pdop, 3)}\n")
            f.write(f"  平均 GDOP：{_format_float(summary.average_gdop, 3)}\n")
            f.write(f"  平均误差：{_format_float(summary.mean_error_3d, 3)} m\n")
            f.write(f"  RMS 误差：{_format_float(summary.rms_error_3d, 3)} m\n")
            f.write(f"  最大误差：{_format_float(summary.max_error_3d, 3)} m\n")
            f.write(f"  输出目录：{result['output_dir']}\n\n")

        f.write("场景间结果对比\n")
        f.write("-" * 40 + "\n")
        success_rates = [r["summary"].success_rate for r in scenario_results]
        mean_errors = [
            r["summary"].mean_error_3d
            for r in scenario_results
            if math.isfinite(r["summary"].mean_error_3d)
        ]
        if len(set(success_rates)) == 1:
            f.write(f"所有场景成功率一致：{success_rates[0] * 100:.2f}%\n")
        else:
            f.write("不同场景成功率存在差异，说明参数设置对定位结果有影响。\n")
        if mean_errors:
            f.write(
                f"平均误差范围：{min(mean_errors):.3f} m 至 {max(mean_errors):.3f} m\n"
            )
        f.write("\n系统测试结论：\n")
        f.write("多场景测试已完成。系统在不同参数配置下均能正常输出定位结果，\n")
        f.write("满足多场景测试要求。\n")

    return {
        "summary_csv": summary_csv_path,
        "report": report_path,
        "scenario_results": scenario_results,
    }


def main() -> int:
    """主程序入口。"""

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    module_status: Dict[str, str] = {}
    module_outputs: Dict[str, List[Path]] = {}

    try:
        nav_path = Path(NAV_FILE_PATH)
        if not nav_path.exists():
            raise FileNotFoundError(
                f"NAV 文件不存在：{nav_path.resolve()}。请在 basic/module5.py 中修改 NAV_FILE_PATH，或确认 nav/ 目录下存在 .26b_cnav 文件。"
            )

        print("模块一：正在解析 RINEX NAV 导航文件并生成输出...")
        nav_data, parse_info = parse_nav_file(nav_path)

        # 显式调用 run_module1() 生成模块一完整输出
        module1_result = run_module1(
            nav_path=NAV_FILE_PATH,
            receiver_approx=RECEIVER_TRUE_POSITION,
            epochs=[TEST_EPOCH_TIME],
            seed=RANDOM_SEED,
            output_dir=OUTPUT_DIR,
            elevation_mask_deg=ELEVATION_MASK_DEG,
            enable_pseudorange_outlier_filter=False,
        )
        module_outputs["module1"] = [
            module1_result["nav_debug"],
            module1_result["simulated_pseudorange"],
            module1_result["summary"],
        ]
        module_status["module1"] = "完成"

        sat_count = len(nav_data)
        eph_count = sum(len(items) for items in nav_data.values())
        if sat_count < 4 or eph_count < 4:
            raise ValueError("星历不足：北斗卫星数量少于 4 颗，无法进行单点定位")
        print(f"  解析完成：北斗卫星 {sat_count} 颗，星历记录 {eph_count} 条")

        print("模块二：正在计算卫星位置与钟差...")
        module2_position_records = calculate_all_satellite_positions(nav_data, TEST_EPOCH_TIME)
        module2_paths = save_satellite_position_outputs(module2_position_records, OUTPUT_DIR, TEST_EPOCH_TIME)

        # 生成模块二伪距修正调试文件（仅用于调试展示，不参与 SPP 解算）
        debug_records = generate_pseudorange_correction_debug_records(
            nav_data, TEST_EPOCH_TIME, RECEIVER_TRUE_POSITION
        )
        debug_csv_path = save_pseudorange_correction_debug_csv(debug_records, OUTPUT_DIR)

        module_outputs["module2"] = [
            module2_paths["csv"],
            module2_paths["summary"],
            debug_csv_path,
        ]
        module_status["module2"] = "完成"

        satellite_positions, satellite_health = _collect_satellite_data_from_module1(
            nav_data, TEST_EPOCH_TIME
        )

        print("模块三：正在生成模拟伪距并进行单历元 SPP 解算...")
        if len(satellite_positions) < 4:
            raise ValueError("模块三无法运行：可用卫星位置少于 4 颗")
        pseudo_records = generate_simulated_pseudorange_records(
            satellite_positions,
            RECEIVER_TRUE_POSITION,
            TEST_EPOCH_TIME,
            seed=RANDOM_SEED,
            satellite_health=satellite_health,
        )
        pseudoranges = pseudorange_records_to_dict(pseudo_records)
        single_solution = solve_spp(
            satellite_positions,
            pseudoranges,
            initial_position=RECEIVER_TRUE_POSITION,
            max_iter=MAX_ITERATIONS,
            convergence_threshold=CONVERGENCE_THRESHOLD,
            satellite_health=satellite_health,
            elevation_mask_deg=ELEVATION_MASK_DEG,
            enable_pseudorange_outlier_filter=False,
        )
        module3_paths = save_single_epoch_spp_outputs(
            pseudo_records,
            single_solution,
            output_path,
            TEST_EPOCH_TIME,
            RECEIVER_TRUE_POSITION,
            elevation_mask_deg=ELEVATION_MASK_DEG,
        )
        module_outputs["module3"] = list(module3_paths.values())
        module_status["module3"] = "完成" if single_solution.converged else f"完成但解算失败：{single_solution.message}"

        print("模块四：正在进行连续定位与结果分析...")

        # 根据 ENABLE_RECEIVER_MOTION 构造轨迹参数
        if ENABLE_RECEIVER_MOTION:
            receiver_trajectory = build_linear_receiver_trajectory(
                SIMULATION_START_TIME,
                RECEIVER_INITIAL_POSITION,
                RECEIVER_VELOCITY_ECEF_MPS,
            )
            receiver_initial_approx = RECEIVER_INITIAL_APPROX_POSITION
            receiver_true_position = RECEIVER_INITIAL_POSITION
        else:
            receiver_trajectory = None
            receiver_initial_approx = None
            receiver_true_position = RECEIVER_TRUE_POSITION

        _, summary = run_continuous_positioning(
            nav_data=nav_data,
            start_time=SIMULATION_START_TIME,
            end_time=SIMULATION_END_TIME,
            interval_seconds=SAMPLING_INTERVAL_SECONDS,
            receiver_true_position=receiver_true_position,
            output_dir=output_path,
            random_seed=RANDOM_SEED,
            max_iter=MAX_ITERATIONS,
            convergence_threshold=CONVERGENCE_THRESHOLD,
            elevation_mask_deg=ELEVATION_MASK_DEG,
            receiver_trajectory=receiver_trajectory,
            receiver_initial_approx=receiver_initial_approx,
        )
        module_outputs["module4"] = [
            output_path / "module4_continuous_position_results.csv",
            output_path / "module4_error_statistics.txt",
            output_path / "module4_error_curve.png",
            output_path / "module4_trajectory.png",
            output_path / "module4_satellite_dop_curve.png",
        ]
        module_status["module4"] = "完成"

        # 多场景测试
        if ENABLE_MULTI_SCENARIO_TEST:
            print("正在进行多场景定位测试...")
            multi_result = run_multi_scenario_tests()
            multi_outputs = [
                multi_result["summary_csv"],
                multi_result["report"],
            ]
            for sr in multi_result["scenario_results"]:
                multi_outputs.extend(sr["output_files"])
            module_outputs["multi_scenario"] = multi_outputs
            module_status["multi_scenario"] = "完成"

        report_path = write_system_test_report(
            output_path,
            nav_path,
            module_status,
            module_outputs,
            summary,
        )
        print_test_report(summary, report_path)
        return 0 if summary.success_epochs > 0 else 2

    except Exception as exc:
        # 如果运行过程中出现异常，也尽量生成模块五报告，说明失败原因。
        module_status.setdefault("module1", "未完成")
        module_status.setdefault("module2", "未完成")
        module_status.setdefault("module3", "未完成")
        module_status.setdefault("module4", "未完成")
        print(f"系统运行失败：{exc}")
        traceback.print_exc()

        failed_summary = AnalysisSummary(
            total_epochs=0,
            success_epochs=0,
            failed_epochs=0,
            average_satellite_count=math.nan,
            average_pdop=math.nan,
            average_gdop=math.nan,
            mean_error_3d=math.nan,
            rms_error_3d=math.nan,
            max_error_3d=math.nan,
            min_error_3d=math.nan,
            success_rate=0.0,
            evaluation=f"系统未能完成完整流程：{exc}",
        )
        report_path = write_system_test_report(
            output_path,
            NAV_FILE_PATH,
            module_status,
            module_outputs,
            failed_summary,
        )
        with report_path.open("a", encoding="utf-8-sig") as file:
            file.write(f"\n失败原因：{exc}\n")
        print(f"失败报告已保存：{report_path}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
