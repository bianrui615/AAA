"""标准 SPP 解算入口。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from module3_spp_solver import ECEF, solve_spp

from ..gnss.dop import compute_dops
from ..models import C, EpochSolution
from ..table import make_dataframe


SPP_COLUMNS = [
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
]


def solve_epoch_spp(
    satellite_positions: Dict[str, ECEF],
    pseudoranges: Dict[str, float],
    *,
    epoch: datetime,
    initial_position: Optional[ECEF] = None,
    max_iter: int = 10,
    threshold: float = 1e-4,
    elevation_mask: float = 0.0,
) -> EpochSolution:
    """解算单个历元，并返回课程要求的公开字段。"""

    raw = solve_spp(
        satellite_positions,
        pseudoranges,
        initial_position=initial_position,
        max_iter=max_iter,
        convergence_threshold=threshold,
        elevation_mask_deg=elevation_mask,
    )
    if raw.converged:
        receiver = (raw.x, raw.y, raw.z)
        try:
            gdop, pdop, hdop, vdop, tdop = compute_dops(satellite_positions, receiver)
        except Exception:
            gdop, pdop, hdop, vdop, tdop = raw.gdop, raw.pdop, float("nan"), float("nan"), float("nan")
    else:
        gdop, pdop, hdop, vdop, tdop = raw.gdop, raw.pdop, float("nan"), float("nan"), float("nan")
    clock_s = raw.clock_bias / C if raw.converged else float("nan")
    return EpochSolution(
        epoch=epoch.isoformat(sep=" "),
        x_m=raw.x,
        y_m=raw.y,
        z_m=raw.z,
        lat_deg=raw.lat,
        lon_deg=raw.lon,
        height_m=raw.height,
        receiver_clock_bias_m=raw.clock_bias,
        receiver_clock_bias_s=clock_s,
        num_sats=raw.satellite_count,
        GDOP=gdop,
        PDOP=pdop,
        HDOP=hdop,
        VDOP=vdop,
        TDOP=tdop,
        converged=raw.converged,
        iterations=raw.iterations,
        message=raw.message,
    )


def save_spp_epoch_result(solution: EpochSolution, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "spp_epoch_result.csv"
    make_dataframe([solution.as_dict()], SPP_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")
    return path
