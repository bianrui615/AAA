"""模块一的伪距模拟与预处理。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from module3_spp_solver import (
    ECEF,
    generate_simulated_pseudorange_records,
    pseudorange_records_to_dict,
    satellite_elevation_deg,
)

from ..table import make_dataframe


PSEUDORANGE_COLUMNS = [
    "epoch",
    "sat_id",
    "sat_x_m",
    "sat_y_m",
    "sat_z_m",
    "health",
    "is_healthy",
    "elevation_deg",
    "rho_m",
    "sisre_error_m",
    "ionosphere_error_m",
    "troposphere_error_m",
    "receiver_clock_error_m",
    "noise_error_m",
    "satellite_clock_correction_m",
    "raw_pseudorange_m",
    "simulated_pseudorange_m",
    "passed_health_filter",
    "passed_elevation_filter",
    "passed_outlier_filter",
    "aligned_epoch",
]


def _mad_filter(rows: List[dict], sigma_factor: float = 3.0) -> None:
    values = [float(row["simulated_pseudorange_m"] - row["rho_m"]) for row in rows]
    if len(values) < 5:
        for row in rows:
            row["passed_outlier_filter"] = True
        return
    import statistics

    center = statistics.median(values)
    deviations = [abs(value - center) for value in values]
    mad = statistics.median(deviations)
    sigma = 1.4826 * mad
    if sigma <= 1e-9:
        for row in rows:
            row["passed_outlier_filter"] = True
        return
    for row, value in zip(rows, values):
        row["passed_outlier_filter"] = abs(value - center) <= sigma_factor * sigma


def simulate_pseudorange(
    satellite_positions: Dict[str, ECEF],
    receiver_ecef: ECEF,
    epoch: datetime,
    *,
    satellite_health: Optional[Dict[str, float]] = None,
    satellite_clock_biases: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    elevation_mask: float = 0.0,
) -> object:
    """模拟伪距观测值，并标记各项预处理筛选结果。"""

    raw = generate_simulated_pseudorange_records(
        satellite_positions,
        receiver_ecef,
        epoch,
        seed=seed,
        satellite_health=satellite_health,
        satellite_clock_biases=satellite_clock_biases,
    )
    rows: List[dict] = []
    for row in raw:
        health = float(row.get("health", 0.0))
        elevation = float(row.get("elevation_deg", satellite_elevation_deg(
            (float(row["sat_X"]), float(row["sat_Y"]), float(row["sat_Z"])),
            receiver_ecef,
        )))
        rows.append(
            {
                "epoch": row["epoch_time"],
                "sat_id": row["sat_id"],
                "sat_x_m": row["sat_X"],
                "sat_y_m": row["sat_Y"],
                "sat_z_m": row["sat_Z"],
                "health": health,
                "is_healthy": int(round(health)) == 0,
                "elevation_deg": elevation,
                "rho_m": row["rho"],
                "sisre_error_m": row["sisre_error"],
                "ionosphere_error_m": row["ionosphere_error"],
                "troposphere_error_m": row["troposphere_error"],
                "receiver_clock_error_m": row["receiver_clock_error"],
                "noise_error_m": row["noise_error"],
                "satellite_clock_correction_m": row["satellite_clock_correction_m"],
                "raw_pseudorange_m": row["raw_simulated_pseudorange"],
                "simulated_pseudorange_m": row["simulated_pseudorange"],
                "passed_health_filter": int(round(health)) == 0,
                "passed_elevation_filter": elevation >= elevation_mask,
                "passed_outlier_filter": True,
                "aligned_epoch": True,
            }
        )
    _mad_filter(rows)
    return make_dataframe(rows, PSEUDORANGE_COLUMNS)


def pseudorange_dict(table) -> Dict[str, float]:
    """将伪距表转换为 sat_id -> 模拟伪距 字典。"""

    rows = table.to_dict("records") if hasattr(table, "to_dict") else list(table)
    return {
        row["sat_id"]: float(row["simulated_pseudorange_m"])
        for row in rows
        if row.get("passed_health_filter", True)
        and row.get("passed_elevation_filter", True)
        and row.get("passed_outlier_filter", True)
    }


def save_simulated_pseudorange(table, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "simulated_pseudorange.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return path
