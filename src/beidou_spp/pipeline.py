"""北斗 SPP 全流程调度。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import math
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from module3_spp_solver import geometric_distance

from .analysis.accuracy import accuracy_stats
from .analysis.report import write_accuracy_report, write_test_report
from .analysis.visualization import plot_standard_results
from .config import PipelineConfig
from .gnss.corrections import correct_pseudorange, save_corrected_pseudorange
from .gnss.satellite_position import compute_satellite_states, satellite_maps, save_satellite_debug
from .models import PipelineResult
from .positioning.spp_solver import save_spp_epoch_result, solve_epoch_spp
from .rinex.nav_parser import parse_nav_file_with_info, save_parsed_nav_debug
from .rinex.pseudorange_simulator import pseudorange_dict, save_simulated_pseudorange, simulate_pseudorange
from .table import make_dataframe


POSITIONING_COLUMNS = [
    "epoch",
    "x_m",
    "y_m",
    "z_m",
    "lat_deg",
    "lon_deg",
    "height_m",
    "receiver_clock_bias_m",
    "receiver_clock_bias_s",
    "num_sats",
    "GDOP",
    "PDOP",
    "HDOP",
    "VDOP",
    "TDOP",
    "converged",
    "iterations",
    "message",
    "error_3d_m",
]


def _epochs(config: PipelineConfig) -> Iterator:
    current = config.start
    step = timedelta(seconds=config.interval)
    while current <= config.end:
        yield current
        current += step


def _filtered_corrected_dict(corrected_table, pseudo_table) -> Dict[str, float]:
    corrected_rows = corrected_table.to_dict("records") if hasattr(corrected_table, "to_dict") else list(corrected_table)
    pseudo_rows = pseudo_table.to_dict("records") if hasattr(pseudo_table, "to_dict") else list(pseudo_table)
    keep = {
        row["sat_id"]
        for row in pseudo_rows
        if row.get("passed_health_filter", True)
        and row.get("passed_elevation_filter", True)
        and row.get("passed_outlier_filter", True)
    }
    return {
        row["sat_id"]: float(row["corrected_pseudorange_m"])
        for row in corrected_rows
        if row["sat_id"] in keep
    }


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """运行模块一到模块五，并保存所有标准输出。"""

    if config.interval <= 0:
        raise ValueError("历元间隔必须为正数")
    if config.end < config.start:
        raise ValueError("end time cannot be earlier than start time")

    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    files: Dict[str, Path] = {}

    nav_data, parse_info = parse_nav_file_with_info(config.nav)
    files["parsed_nav_debug"] = save_parsed_nav_debug(nav_data, output, parse_info)

    all_pseudo_rows: List[Dict] = []
    all_corrected_rows: List[Dict] = []
    all_sat_rows: List[Dict] = []
    epoch_rows: List[Dict] = []
    previous_position = config.receiver_ecef
    single_saved = False

    for epoch in _epochs(config):
        sat_table = compute_satellite_states(nav_data, epoch)
        sat_rows = sat_table.to_dict("records") if hasattr(sat_table, "to_dict") else list(sat_table)
        all_sat_rows.extend(sat_rows)
        positions, clocks, health = satellite_maps(sat_table)
        if len(positions) < 4:
            solution_row = solve_epoch_spp(
                {},
                {},
                epoch=epoch,
                initial_position=previous_position,
                max_iter=config.max_iter,
                threshold=config.threshold,
                elevation_mask=config.elevation_mask,
            ).as_dict()
            solution_row["message"] = "可用健康卫星少于 4 颗"
            solution_row["error_3d_m"] = math.nan
            epoch_rows.append(solution_row)
            continue

        pseudo_table = simulate_pseudorange(
            positions,
            config.receiver_ecef,
            epoch,
            satellite_health=health,
            satellite_clock_biases=clocks,
            seed=config.seed + len(epoch_rows),
            elevation_mask=config.elevation_mask,
        )
        corrected_table = correct_pseudorange(pseudo_table, config.receiver_ecef)
        all_pseudo_rows.extend(pseudo_table.to_dict("records"))
        all_corrected_rows.extend(corrected_table.to_dict("records"))
        pseudoranges = _filtered_corrected_dict(corrected_table, pseudo_table)

        solution = solve_epoch_spp(
            positions,
            pseudoranges,
            epoch=epoch,
            initial_position=previous_position,
            max_iter=config.max_iter,
            threshold=config.threshold,
            elevation_mask=config.elevation_mask,
        )
        row = solution.as_dict()
        if solution.converged:
            previous_position = (solution.x_m, solution.y_m, solution.z_m)
            row["error_3d_m"] = geometric_distance(previous_position, config.receiver_ecef)
        else:
            row["error_3d_m"] = math.nan
        epoch_rows.append(row)

        if not single_saved:
            files["spp_epoch_result"] = save_spp_epoch_result(solution, output)
            single_saved = True

    sat_all = make_dataframe(all_sat_rows)
    files["satellite_debug"] = save_satellite_debug(sat_all, output)
    pseudo_all = make_dataframe(all_pseudo_rows)
    files["simulated_pseudorange"] = save_simulated_pseudorange(pseudo_all, output)
    corrected_all = make_dataframe(all_corrected_rows)
    files["corrected_pseudorange"] = save_corrected_pseudorange(corrected_all, output)

    positioning = make_dataframe(epoch_rows, POSITIONING_COLUMNS)
    files["positioning_results"] = output / "positioning_results.csv"
    positioning.to_csv(files["positioning_results"], index=False, encoding="utf-8-sig")
    # 兼容旧版 GUI 和模块四原有文件命名。
    positioning.to_csv(output / "module4_continuous_position_results.csv", index=False, encoding="utf-8-sig")

    stats = accuracy_stats(epoch_rows)
    files["accuracy_report"] = write_accuracy_report(epoch_rows, stats, output)
    for fig_path in plot_standard_results(epoch_rows, config.receiver_ecef, output):
        files[fig_path.stem] = fig_path

    files["test_report"] = write_test_report(
        [
            {
                "nav_file": str(config.nav),
                "success_epochs": sum(1 for row in epoch_rows if row.get("converged")),
                "total_epochs": len(epoch_rows),
                "rms_error_m": stats["rms_error_m"],
                "max_error_m": stats["max_error_m"],
                "mean_gdop": _mean(row.get("GDOP", math.nan) for row in epoch_rows),
            }
        ],
        output / "test_report.md",
    )
    docs_report = write_test_report(
        [
            {
                "nav_file": str(config.nav),
                "success_epochs": sum(1 for row in epoch_rows if row.get("converged")),
                "total_epochs": len(epoch_rows),
                "rms_error_m": stats["rms_error_m"],
                "max_error_m": stats["max_error_m"],
                "mean_gdop": _mean(row.get("GDOP", math.nan) for row in epoch_rows),
            }
        ],
        Path("docs") / "test_report.md",
    )
    files["docs_test_report"] = docs_report

    return PipelineResult(
        output_dir=output,
        files=files,
        epoch_results=epoch_rows,
        success_epochs=sum(1 for row in epoch_rows if row.get("converged")),
        total_epochs=len(epoch_rows),
        rms_error_m=stats["rms_error_m"],
        mean_error_m=stats["mean_error_m"],
        max_error_m=stats["max_error_m"],
    )


def _mean(values) -> float:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else math.nan
