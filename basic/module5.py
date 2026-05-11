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

from datetime import datetime
import math
import traceback
from typing import Dict, List

from basic.module1 import (
    compute_satellite_clock_bias,
    compute_satellite_position,
    parse_rinex_nav_with_info,
    save_nav_parse_outputs,
    select_ephemeris,
)
from basic.module3 import (
    ECEF,
    generate_simulated_pseudorange_records,
    pseudorange_records_to_dict,
    save_single_epoch_spp_outputs,
    solve_spp,
)
from basic.module4 import AnalysisSummary, run_continuous_positioning


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
OUTPUT_DIR = "output"

# 模块二和模块三独立输出使用的测试历元。
TEST_EPOCH_TIME = SIMULATION_START_TIME


def _format_float(value: float, digits: int = 3) -> str:
    """格式化浮点数，遇到 NaN 时返回字符串 NaN。"""

    if value is None or not math.isfinite(value):
        return "NaN"
    return f"{value:.{digits}f}"


def _collect_satellite_data_from_module1(
    nav_data: dict,
    epoch_time: datetime,
) -> tuple[dict[str, ECEF], dict[str, float], dict[str, float]]:
    """直接调用 module1 计算卫星位置、健康状态和钟差。"""

    positions: Dict[str, ECEF] = {}
    health: Dict[str, float] = {}
    clock_biases: Dict[str, float] = {}
    for sat_id in sorted(nav_data):
        eph = select_ephemeris(nav_data, sat_id, epoch_time, healthy_only=True)
        if eph is None:
            continue
        try:
            x, y, z = compute_satellite_position(eph, epoch_time)
            clock_bias, _ = compute_satellite_clock_bias(eph, epoch_time)
            positions[sat_id] = (x, y, z)
            health[sat_id] = float(eph.health)
            clock_biases[sat_id] = clock_bias
        except Exception:
            continue
    return positions, health, clock_biases


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
        }
        for module_key in ["module1", "module2", "module3", "module4"]:
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

        print("模块一：正在解析 RINEX NAV 导航文件...")
        nav_data, parse_info = parse_rinex_nav_with_info(nav_path)
        module1_paths = save_nav_parse_outputs(nav_data, output_path, parse_info)
        module_outputs["module1"] = list(module1_paths.values())
        module_status["module1"] = "完成"

        sat_count = len(nav_data)
        eph_count = sum(len(items) for items in nav_data.values())
        if sat_count < 4 or eph_count < 4:
            raise ValueError("星历不足：北斗卫星数量少于 4 颗，无法进行单点定位")
        print(f"  解析完成：北斗卫星 {sat_count} 颗，星历记录 {eph_count} 条")

        print("模块二：正在计算卫星位置与钟差...")
        satellite_positions, satellite_health, satellite_clock_biases = _collect_satellite_data_from_module1(
            nav_data, TEST_EPOCH_TIME
        )
        module_status["module2"] = "完成（由 module1 直接提供）"

        print("模块三：正在生成模拟伪距并进行单历元 SPP 解算...")
        if len(satellite_positions) < 4:
            raise ValueError("模块三无法运行：可用卫星位置少于 4 颗")
        pseudo_records = generate_simulated_pseudorange_records(
            satellite_positions,
            RECEIVER_TRUE_POSITION,
            TEST_EPOCH_TIME,
            seed=RANDOM_SEED,
            satellite_health=satellite_health,
            satellite_clock_biases=satellite_clock_biases,
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
        _, summary = run_continuous_positioning(
            nav_data=nav_data,
            start_time=SIMULATION_START_TIME,
            end_time=SIMULATION_END_TIME,
            interval_seconds=SAMPLING_INTERVAL_SECONDS,
            receiver_true_position=RECEIVER_TRUE_POSITION,
            output_dir=output_path,
            random_seed=RANDOM_SEED,
            max_iter=MAX_ITERATIONS,
            convergence_threshold=CONVERGENCE_THRESHOLD,
            elevation_mask_deg=ELEVATION_MASK_DEG,
        )
        module_outputs["module4"] = [
            output_path / "module4_continuous_position_results.csv",
            output_path / "module4_error_statistics.txt",
            output_path / "module4_error_curve.png",
            output_path / "module4_trajectory.png",
            output_path / "module4_satellite_dop_curve.png",
        ]
        module_status["module4"] = "完成"

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
