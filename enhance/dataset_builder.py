"""
dataset_builder.py

构建机器学习数据集。

对 enhance_config.py 中定义的每个场景：
1. 调用 basic/ 中的星历解析、卫星位置计算、伪距模拟、SPP 解算；
2. 逐历元收集定位结果与伪距统计特征；
3. 输出 机器学习数据集.csv（仅包含成功解算历元）。
"""

from __future__ import annotations

import csv
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from basic.module1 import (
    compute_satellite_position,
    parse_nav_file,
    select_ephemeris,
)
from basic.module3 import (
    generate_simulated_pseudorange_records,
    pseudorange_records_to_dict,
    solve_spp,
)
from enhance.enhance_config import (
    BASE_OUTPUT_DIR,
    EXTRA_ANALYSIS_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    SCENARIO_OUTPUT_DIR,
    SCENARIOS,
    ScenarioConfig,
)


def _time_range(start_time, end_time, interval_seconds: int):
    """生成闭区间历元序列。"""
    current = start_time
    step = timedelta(seconds=interval_seconds)
    while current <= end_time:
        yield current
        current += step


def _get_basic_output_dir(scenario_name: str) -> Path:
    """根据场景名找到 gui_scenario_runner 对应的输出目录。"""
    basic_scenario_name = scenario_name
    if scenario_name.startswith("scenario") and not scenario_name.startswith("scenario_"):
        basic_scenario_name = scenario_name.replace("scenario", "scenario_", 1)
    return Path(f"outputs/basic/gui_scenario_runner/{basic_scenario_name}")


def _load_from_basic_csv(scenario: ScenarioConfig) -> Optional[List[dict]]:
    """尝试从基础部分 outputs/basic/<nav_name>/module4_连续定位结果.csv 加载已有数据，
    并补充缺失的伪距统计特征。

    如果 CSV 不存在或解析失败，返回 None。
    """
    basic_dir = _get_basic_output_dir(scenario.name)
    csv_path = basic_dir / "module4_连续定位结果.csv"
    if not csv_path.exists():
        return None

    print(f"  [dataset_builder] 从基础部分加载已有数据：{csv_path}")
    nav_data, _ = parse_nav_file(scenario.nav_file_path)
    rng = random.Random(scenario.random_seed)

    rows: List[dict] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch_time = datetime.fromisoformat(row["epoch_time"].replace(" ", "T"))
            # 收集卫星位置
            satellite_positions: Dict[str, Tuple[float, float, float]] = {}
            for sat_id in sorted(nav_data):
                eph = select_ephemeris(nav_data, sat_id, epoch_time, healthy_only=True)
                if eph is None:
                    continue
                try:
                    x, y, z = compute_satellite_position(eph, epoch_time)
                    if math.sqrt(x * x + y * y + z * z) <= 0.0:
                        continue
                    satellite_positions[sat_id] = (x, y, z)
                except Exception:
                    continue

            raw_satellite_count = len(satellite_positions)
            if raw_satellite_count < 4:
                rows.append(
                    {
                        "scenario_name": scenario.name,
                        "epoch_time": row["epoch_time"],
                        "status": "失败",
                        "failure_reason": "可用卫星数量少于 4 颗",
                        "raw_satellite_count": raw_satellite_count,
                    }
                )
                continue

            true_x = float(row.get("true_X", 0))
            true_y = float(row.get("true_Y", 0))
            true_z = float(row.get("true_Z", 0))
            spp_x = float(row.get("X", 0))
            spp_y = float(row.get("Y", 0))
            spp_z = float(row.get("Z", 0))
            error_3d = float(row.get("error_3d", "nan"))
            status = row.get("status", "失败")

            # 使用 CSV 中的真实坐标作为接收机位置，保证伪距统计与定位结果一致
            receiver_true_position = (true_x, true_y, true_z)
            pseudo_records = generate_simulated_pseudorange_records(
                satellite_positions,
                receiver_true_position,
                epoch_time,
                rng=rng,
            )
            pseudorange_values = [r["simulated_pseudorange"] for r in pseudo_records]
            rho_values = [r["rho"] for r in pseudo_records]
            sisre_values = [r["sisre_error"] for r in pseudo_records]
            iono_values = [r["iono_error"] for r in pseudo_records]
            tropo_values = [r["tropo_error"] for r in pseudo_records]
            clock_values = [r["receiver_clock_error"] for r in pseudo_records]
            noise_values = [r["noise_error"] for r in pseudo_records]
            elevation_values = [r["elevation_deg"] for r in pseudo_records]

            record: dict = {
                "scenario_name": scenario.name,
                "epoch_time": row["epoch_time"],
                "satellite_count": int(row.get("satellite_count", 0)),
                "raw_satellite_count": int(row.get("raw_satellite_count", raw_satellite_count)),
                "PDOP": float(row.get("PDOP", "nan")),
                "GDOP": float(row.get("GDOP", "nan")),
                "clock_bias": float(row.get("clock_bias", "nan")),
                "iteration_count": int(row.get("iteration_count", 0)),
                "elevation_mask_deg": float(row.get("elevation_mask_deg", 0.0)),
                "mean_pseudorange": float(np.mean(pseudorange_values)),
                "std_pseudorange": float(np.std(pseudorange_values)),
                "mean_rho": float(np.mean(rho_values)),
                "std_rho": float(np.std(rho_values)),
                "mean_sisre_error": float(np.mean(sisre_values)),
                "mean_iono_error": float(np.mean(iono_values)),
                "mean_tropo_error": float(np.mean(tropo_values)),
                "mean_receiver_clock_error": float(np.mean(clock_values)),
                "mean_noise_error": float(np.mean(noise_values)),
                "mean_elevation_deg": float(np.mean(elevation_values)),
                "min_elevation_deg": float(np.min(elevation_values)),
                "max_elevation_deg": float(np.max(elevation_values)),
                "spp_x": spp_x,
                "spp_y": spp_y,
                "spp_z": spp_z,
                "true_x": true_x,
                "true_y": true_y,
                "true_z": true_z,
                "error_x": true_x - spp_x,
                "error_y": true_y - spp_y,
                "error_z": true_z - spp_z,
                "error_3d_before": error_3d if math.isfinite(error_3d) else math.nan,
                "status": status,
                "failure_reason": "" if status == "成功" else row.get("failure_reason", ""),
            }
            rows.append(record)

    success_count = sum(1 for r in rows if r["status"] == "成功")
    print(
        f"  [dataset_builder] 从基础部分加载完成："
        f"成功 {success_count} 历元，失败 {len(rows) - success_count} 历元"
    )
    return rows


def run_scenario_and_collect(
    scenario: ScenarioConfig,
    save_scenario_csv: bool = True,
) -> List[dict]:
    """运行单个场景，逐历元解算并收集特征与标签。

    优先尝试从基础部分 outputs/basic/ 加载已有结果；
    如果不存在，则重新调用 basic/ 模块生成。

    返回列表中每个字典对应一个历元（成功或失败）。
    失败历元会被记录但后续不参与训练。
    """
    print(f"  [dataset_builder] 正在运行场景：{scenario.name}")

    # 优先从基础部分加载已有数据
    loaded = _load_from_basic_csv(scenario)
    if loaded is not None:
        return loaded

    nav_data, _ = parse_nav_file(scenario.nav_file_path)
    rng = random.Random(scenario.random_seed)

    output_dir = SCENARIO_OUTPUT_DIR / scenario.name
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []
    previous_solution: Optional[Tuple[float, float, float]] = None

    for epoch_time in _time_range(
        scenario.start_time, scenario.end_time, scenario.interval_seconds
    ):
        # 1. 收集卫星位置（仅健康卫星）
        satellite_positions: Dict[str, Tuple[float, float, float]] = {}
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

        raw_satellite_count = len(satellite_positions)

        if raw_satellite_count < 4:
            records.append(
                {
                    "scenario_name": scenario.name,
                    "epoch_time": epoch_time.isoformat(sep=" "),
                    "status": "失败",
                    "failure_reason": "可用卫星数量少于 4 颗",
                    "raw_satellite_count": raw_satellite_count,
                }
            )
            continue

        # 2. 生成模拟伪距并收集统计量
        pseudo_records = generate_simulated_pseudorange_records(
            satellite_positions,
            scenario.receiver_true_position,
            epoch_time,
            rng=rng,
        )

        pseudorange_values = [r["simulated_pseudorange"] for r in pseudo_records]
        rho_values = [r["rho"] for r in pseudo_records]
        sisre_values = [r["sisre_error"] for r in pseudo_records]
        iono_values = [r["iono_error"] for r in pseudo_records]
        tropo_values = [r["tropo_error"] for r in pseudo_records]
        clock_values = [r["receiver_clock_error"] for r in pseudo_records]
        noise_values = [r["noise_error"] for r in pseudo_records]
        elevation_values = [r["elevation_deg"] for r in pseudo_records]

        # 3. SPP 解算
        pseudorange_dict = pseudorange_records_to_dict(pseudo_records)
        solution = solve_spp(
            satellite_positions,
            pseudorange_dict,
            initial_position=previous_solution
            if previous_solution is not None
            else scenario.receiver_true_position,
            max_iter=scenario.max_iter,
            convergence_threshold=scenario.convergence_threshold,
            elevation_mask_deg=scenario.elevation_mask_deg,
            enable_pseudorange_outlier_filter=False,
        )

        true_x, true_y, true_z = scenario.receiver_true_position
        spp_x, spp_y, spp_z = solution.x, solution.y, solution.z

        record: dict = {
            "scenario_name": scenario.name,
            "epoch_time": epoch_time.isoformat(sep=" "),
            "satellite_count": solution.satellite_count,
            "raw_satellite_count": raw_satellite_count,
            "PDOP": solution.pdop,
            "GDOP": solution.gdop,
            "clock_bias": solution.clock_bias,
            "iteration_count": solution.iterations,
            "elevation_mask_deg": scenario.elevation_mask_deg,
            "mean_pseudorange": float(np.mean(pseudorange_values)),
            "std_pseudorange": float(np.std(pseudorange_values)),
            "mean_rho": float(np.mean(rho_values)),
            "std_rho": float(np.std(rho_values)),
            "mean_sisre_error": float(np.mean(sisre_values)),
            "mean_iono_error": float(np.mean(iono_values)),
            "mean_tropo_error": float(np.mean(tropo_values)),
            "mean_receiver_clock_error": float(np.mean(clock_values)),
            "mean_noise_error": float(np.mean(noise_values)),
            "mean_elevation_deg": float(np.mean(elevation_values)),
            "min_elevation_deg": float(np.min(elevation_values)),
            "max_elevation_deg": float(np.max(elevation_values)),
            "spp_x": spp_x,
            "spp_y": spp_y,
            "spp_z": spp_z,
            "true_x": true_x,
            "true_y": true_y,
            "true_z": true_z,
            "error_x": true_x - spp_x,
            "error_y": true_y - spp_y,
            "error_z": true_z - spp_z,
            "error_3d_before": math.sqrt(
                (true_x - spp_x) ** 2 + (true_y - spp_y) ** 2 + (true_z - spp_z) ** 2
            )
            if solution.converged
            else math.nan,
            "status": "成功" if solution.converged else "失败",
            "failure_reason": ""
            if solution.converged
            else solution.message,
        }

        if solution.converged:
            previous_solution = (spp_x, spp_y, spp_z)

        records.append(record)

    # 保存场景原始定位结果 CSV（与基础部分 module4 格式保持一致，便于对照）
    if save_scenario_csv:
        _save_scenario_results(records, output_dir)

    success_count = sum(1 for r in records if r["status"] == "成功")
    fail_count = len(records) - success_count
    print(
        f"  [dataset_builder] 场景 {scenario.name} 完成："
        f"成功 {success_count} 历元，失败 {fail_count} 历元"
    )
    return records


def _save_scenario_results(records: List[dict], output_dir: Path) -> None:
    """保存场景连续定位结果到 scenarios/scenarioX/module4_连续定位结果.csv。"""
    csv_path = output_dir / "module4_连续定位结果.csv"
    if not records:
        return

    fieldnames = [
        "epoch_time",
        "status",
        "satellite_count",
        "raw_satellite_count",
        "spp_x",
        "spp_y",
        "spp_z",
        "error_3d_before",
        "PDOP",
        "GDOP",
        "clock_bias",
        "iteration_count",
        "failure_reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})


def build_dataset() -> Path:
    """运行所有场景并构建 机器学习数据集.csv。

    仅保留成功解算历元写入数据集。
    返回数据集 CSV 路径。
    """
    print("[dataset_builder] 开始构建机器学习数据集...")
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_records: List[dict] = []
    for scenario in SCENARIOS:
        try:
            scenario_records = run_scenario_and_collect(scenario)
            all_records.extend(scenario_records)
        except Exception as exc:
            print(f"  [dataset_builder] 场景 {scenario.name} 运行异常：{exc}")
            continue

    # 仅保留成功历元作为训练/测试样本
    success_records = [r for r in all_records if r.get("status") == "成功"]
    if not success_records:
        raise RuntimeError("所有场景均无成功解算历元，无法构建数据集。")

    dataset_path = BASE_OUTPUT_DIR / "机器学习数据集.csv"
    all_columns = [
        "scenario_name",
        "epoch_time",
    ] + FEATURE_COLUMNS + EXTRA_ANALYSIS_COLUMNS + [
        "spp_x", "spp_y", "spp_z",
        "true_x", "true_y", "true_z",
    ] + LABEL_COLUMNS + ["error_3d_before"]

    with dataset_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        for rec in success_records:
            writer.writerow({k: rec.get(k, "") for k in all_columns})

    total_epochs = len(all_records)
    success_epochs = len(success_records)
    print(
        f"[dataset_builder] 数据集构建完成："
        f"总历元 {total_epochs}，成功历元 {success_epochs}，"
        f"保存至 {dataset_path}"
    )
    return dataset_path


if __name__ == "__main__":
    build_dataset()
