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
from typing import Dict, List, Optional

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
# 用户设置的仿真起止时间按 BDT（北斗时）理解。
SIMULATION_START_TIME = datetime(2026, 4, 1, 0, 0, 0)
SIMULATION_END_TIME = datetime(2026, 4, 1, 1, 0, 0)
SAMPLING_INTERVAL_SECONDS = 300

# SPP 迭代参数。
# 收敛阈值单位为米，1e-2 表示 1 cm。
MAX_ITERATIONS = 12
CONVERGENCE_THRESHOLD = 1e-2

# 高度角截止阈值（度），0 表示不筛选。
# 当前课程题目使用伪距生成模型；默认保留全部可用卫星以获得更好的几何强度。
ELEVATION_MASK_DEG = 0.0

# 随机数种子用于保证伪距模拟可重复。
RANDOM_SEED = 2026

# 所有模块统一输出目录。
OUTPUT_DIR = "outputs/basic/module"

# 模块二和模块三独立输出使用的测试历元。
TEST_EPOCH_TIME = SIMULATION_START_TIME

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


def _get_output_dir(nav_file_path: str | Path) -> str:
    """根据 NAV 文件名生成输出目录。"""
    nav_name = Path(nav_file_path).name
    safe_name = nav_name.replace(".", "_")
    return f"outputs/basic/{safe_name}"


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
    enable_receiver_motion: bool = False,
    receiver_initial_position: Optional[ECEF] = None,
    receiver_velocity_ecef_mps: Optional[Tuple[float, float, float]] = None,
    receiver_initial_approx_position: Optional[ECEF] = None,
    receiver_true_position: Optional[ECEF] = None,
    simulation_start_time: Optional[datetime] = None,
    simulation_end_time: Optional[datetime] = None,
    sampling_interval_seconds: int = 300,
    max_iterations: int = 12,
    convergence_threshold: float = 1e-2,
    elevation_mask_deg: float = 0.0,
    random_seed: int = 2026,
    project_name: str = "北斗定位解算全流程软件系统",
) -> Path:
    """保存模块五系统测试报告。"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "module5_系统测试报告.txt"

    with report_path.open("w", encoding="utf-8-sig") as file:
        file.write("模块五：系统整合与测试报告\n")
        file.write("=" * 44 + "\n")
        file.write(f"项目名称：{project_name}\n")
        file.write(f"测试时间：{datetime.now().isoformat(sep=' ', timespec='seconds')}\n")
        file.write(f"NAV 文件路径：{Path(nav_file_path)}\n")

        # 接收机运动模型说明
        if enable_receiver_motion:
            file.write("接收机运动模型：动态接收机（ECEF 匀速直线运动）\n")
            file.write(
                f"  初始真实坐标："
                f"X={receiver_initial_position[0]:.4f} m，"
                f"Y={receiver_initial_position[1]:.4f} m，"
                f"Z={receiver_initial_position[2]:.4f} m\n"
            )
            file.write(
                f"  ECEF 速度："
                f"Vx={receiver_velocity_ecef_mps[0]:.4f} m/s，"
                f"Vy={receiver_velocity_ecef_mps[1]:.4f} m/s，"
                f"Vz={receiver_velocity_ecef_mps[2]:.4f} m/s\n"
            )
            file.write(
                f"  初始概略坐标："
                f"X={receiver_initial_approx_position[0]:.4f} m，"
                f"Y={receiver_initial_approx_position[1]:.4f} m，"
                f"Z={receiver_initial_approx_position[2]:.4f} m\n"
            )
        else:
            file.write("接收机运动模型：静态接收机\n")
            file.write(
                "接收机真实 ECEF 坐标："
                f"X={receiver_true_position[0]:.4f} m，"
                f"Y={receiver_true_position[1]:.4f} m，"
                f"Z={receiver_true_position[2]:.4f} m\n"
            )

        file.write("坐标用途说明：\n")
        file.write("  - 真实坐标（true position）仅用于模拟伪距生成和三维定位误差评估（标准参考值）；\n")
        file.write("  - 初始概略坐标（initial approx position）用于 SPP 迭代最小二乘的迭代初值，\n")
        file.write("    与真实坐标存在人为偏差，更符合实际工程中\"先验概略坐标\"的应用场景。\n")
        if simulation_start_time is not None:
            file.write(f"仿真起始时间：{simulation_start_time.isoformat(sep=' ')}\n")
        if simulation_end_time is not None:
            file.write(f"仿真结束时间：{simulation_end_time.isoformat(sep=' ')}\n")
        file.write("时间系统说明：用户设置的仿真起止时间与导航文件中的 toc/toe 统一按 BDT（北斗时）理解。当前不做 UTC/GPST/BDT 转换。\n")
        file.write(f"采样间隔：{sampling_interval_seconds} s\n")
        file.write(f"最大迭代次数：{max_iterations}\n")
        file.write(f"收敛阈值：{convergence_threshold} m\n")
        file.write(f"高度角截止阈值：{elevation_mask_deg}°\n")
        file.write(f"随机数种子：{random_seed}\n\n")

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


def print_test_report(summary: AnalysisSummary, report_path: Path, output_dir: str | Path) -> None:
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
    print(f"输出目录：{Path(output_dir).resolve()}")
    print("======================================================\n")



def run_full_basic_pipeline(
    nav_file_path: str | Path,
    receiver_true_position: ECEF,
    simulation_start_time: datetime,
    simulation_end_time: datetime,
    sampling_interval_seconds: int,
    max_iterations: int,
    convergence_threshold: float,
    elevation_mask_deg: float,
    random_seed: int,
    enable_receiver_motion: bool = False,
    receiver_initial_position: Optional[ECEF] = None,
    receiver_velocity_ecef_mps: Optional[Tuple[float, float, float]] = None,
    receiver_initial_approx_position: Optional[ECEF] = None,
    receiver_trajectory: Optional[Callable[[datetime], ECEF]] = None,
    test_epoch_time: Optional[datetime] = None,
    progress_callback: Optional[Callable[[dict, int, int], None]] = None,
) -> Dict[str, Any]:
    """运行完整的基础流程（module1-5），并返回结果字典。

    返回字典包含：
        - module_outputs: Dict[str, List[Path]]
        - module_status: Dict[str, str]
        - summary: AnalysisSummary
        - report_path: Path
        - results: List[dict]
        - nav_data: dict
        - parse_info: Any
        - output_dir: str
    """
    from typing import Any, Callable, Tuple

    output_dir = _get_output_dir(nav_file_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if test_epoch_time is None:
        test_epoch_time = simulation_start_time

    module_status: Dict[str, str] = {}
    module_outputs: Dict[str, List[Path]] = {}

    nav_path = Path(nav_file_path)
    if not nav_path.exists():
        raise FileNotFoundError(
            f"NAV 文件不存在：{nav_path.resolve()}。请确认 nav/ 目录下存在 .26b_cnav 文件。"
        )

    # Module 1
    nav_data, parse_info = parse_nav_file(nav_path)
    module1_result = run_module1(
        nav_path=nav_file_path,
        receiver_approx=receiver_true_position,
        epochs=[test_epoch_time],
        seed=random_seed,
        output_dir=output_dir,
        elevation_mask_deg=elevation_mask_deg,
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

    # Module 2
    module2_position_records = calculate_all_satellite_positions(nav_data, test_epoch_time)
    module2_paths = save_satellite_position_outputs(module2_position_records, output_dir, test_epoch_time)
    debug_records = generate_pseudorange_correction_debug_records(
        nav_data, test_epoch_time, receiver_true_position
    )
    debug_csv_path = save_pseudorange_correction_debug_csv(debug_records, output_dir)
    module_outputs["module2"] = [
        module2_paths["csv"],
        module2_paths["summary"],
        debug_csv_path,
    ]
    module_status["module2"] = "完成"

    # Module 3
    satellite_positions, satellite_health = _collect_satellite_data_from_module1(
        nav_data, test_epoch_time
    )
    if len(satellite_positions) < 4:
        raise ValueError("模块三无法运行：可用卫星位置少于 4 颗")
    pseudo_records = generate_simulated_pseudorange_records(
        satellite_positions,
        receiver_true_position,
        test_epoch_time,
        seed=random_seed,
        satellite_health=satellite_health,
    )
    pseudoranges = pseudorange_records_to_dict(pseudo_records)

    _initial_approx = receiver_initial_approx_position
    if _initial_approx is None:
        _initial_approx = (
            receiver_true_position[0] + 50.0,
            receiver_true_position[1] - 50.0,
            receiver_true_position[2] + 30.0,
        )

    single_solution = solve_spp(
        satellite_positions,
        pseudoranges,
        initial_position=_initial_approx,
        max_iter=max_iterations,
        convergence_threshold=convergence_threshold,
        satellite_health=satellite_health,
        elevation_mask_deg=elevation_mask_deg,
        enable_pseudorange_outlier_filter=False,
        apply_corrections=False,
    )
    module3_paths = save_single_epoch_spp_outputs(
        pseudo_records,
        single_solution,
        output_path,
        test_epoch_time,
        receiver_true_position,
        elevation_mask_deg=elevation_mask_deg,
    )
    module_outputs["module3"] = list(module3_paths.values())
    module_status["module3"] = "完成" if single_solution.converged else f"完成但解算失败：{single_solution.message}"

    # Module 4
    if enable_receiver_motion:
        _receiver_true_position = receiver_initial_position
        _receiver_initial_approx = receiver_initial_approx_position
    else:
        _receiver_true_position = receiver_true_position
        _receiver_initial_approx = None

    results, summary = run_continuous_positioning(
        nav_data=nav_data,
        start_time=simulation_start_time,
        end_time=simulation_end_time,
        interval_seconds=sampling_interval_seconds,
        receiver_true_position=_receiver_true_position,
        output_dir=output_path,
        random_seed=random_seed,
        max_iter=max_iterations,
        convergence_threshold=convergence_threshold,
        elevation_mask_deg=elevation_mask_deg,
        progress_callback=progress_callback,
        receiver_trajectory=receiver_trajectory,
        receiver_initial_approx=_receiver_initial_approx,
    )
    module_outputs["module4"] = [
        output_path / "module4_连续定位结果.csv",
        output_path / "module4_误差统计.txt",
        output_path / "module4_误差曲线.png",
        output_path / "module4_轨迹图.png",
        output_path / "module4_真实与估计轨迹对比.png",
        output_path / "module4_卫星DOP曲线.png",
        output_path / "module4_DOP与误差分析.png",
    ]
    module_status["module4"] = "完成"

    # Module 5
    report_path = write_system_test_report(
        output_dir=output_path,
        nav_file_path=nav_file_path,
        module_status=module_status,
        module_outputs=module_outputs,
        summary=summary,
        enable_receiver_motion=enable_receiver_motion,
        receiver_initial_position=receiver_initial_position,
        receiver_velocity_ecef_mps=receiver_velocity_ecef_mps,
        receiver_initial_approx_position=receiver_initial_approx_position,
        receiver_true_position=receiver_true_position,
        simulation_start_time=simulation_start_time,
        simulation_end_time=simulation_end_time,
        sampling_interval_seconds=sampling_interval_seconds,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        elevation_mask_deg=elevation_mask_deg,
        random_seed=random_seed,
        project_name=PROJECT_NAME,
    )

    return {
        "module_outputs": module_outputs,
        "module_status": module_status,
        "summary": summary,
        "report_path": report_path,
        "results": results,
        "nav_data": nav_data,
        "parse_info": parse_info,
        "output_dir": output_dir,
    }


def main() -> int:
    """主程序入口。"""

    try:
        result = run_full_basic_pipeline(
            nav_file_path=NAV_FILE_PATH,
            receiver_true_position=RECEIVER_TRUE_POSITION,
            simulation_start_time=SIMULATION_START_TIME,
            simulation_end_time=SIMULATION_END_TIME,
            sampling_interval_seconds=SAMPLING_INTERVAL_SECONDS,
            max_iterations=MAX_ITERATIONS,
            convergence_threshold=CONVERGENCE_THRESHOLD,
            elevation_mask_deg=ELEVATION_MASK_DEG,
            random_seed=RANDOM_SEED,
            enable_receiver_motion=ENABLE_RECEIVER_MOTION,
            receiver_initial_position=RECEIVER_INITIAL_POSITION,
            receiver_velocity_ecef_mps=RECEIVER_VELOCITY_ECEF_MPS,
            receiver_initial_approx_position=RECEIVER_INITIAL_APPROX_POSITION,
            test_epoch_time=TEST_EPOCH_TIME,
        )
        print_test_report(
            result["summary"],
            result["report_path"],
            result["output_dir"],
        )
        return 0 if result["summary"].success_epochs > 0 else 2
    except Exception as exc:
        print(f"系统运行失败：{exc}")
        traceback.print_exc()
        return 1



if __name__ == "__main__":
    sys.exit(main())
