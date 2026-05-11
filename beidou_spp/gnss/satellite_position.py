"""广播星历卫星位置计算。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from module1_nav_parser import BroadcastEphemeris, select_ephemeris
from module2_satellite_position_clock import calculate_satellite_position_clock

from ..models import ECEF
from ..table import make_dataframe


SATELLITE_DEBUG_COLUMNS = [
    "epoch",
    "sat_id",
    "x_m",
    "y_m",
    "z_m",
    "satellite_clock_bias_s",
    "satellite_clock_correction_m",
    "relativistic_correction_s",
    "position_norm_m",
    "health",
    "status",
]


def compute_satellite_states(
    nav_data: Dict[str, List[BroadcastEphemeris]],
    epoch: datetime,
) -> object:
    """计算单个历元的卫星 ECEF 坐标和钟差。"""

    rows: List[dict] = []
    for sat_id in sorted(nav_data):
        eph = select_ephemeris(nav_data, sat_id, epoch, healthy_only=False)
        if eph is None:
            continue
        try:
            state = calculate_satellite_position_clock(
                eph,
                epoch,
                raise_on_abnormal=False,
                require_healthy=False,
            )
            x, y, z = state.position
            rows.append(
                {
                    "epoch": epoch.isoformat(sep=" "),
                    "sat_id": sat_id,
                    "x_m": x,
                    "y_m": y,
                    "z_m": z,
                    "satellite_clock_bias_s": state.clock_bias,
                    "satellite_clock_correction_m": state.clock_bias * 299_792_458.0,
                    "relativistic_correction_s": state.relativistic_correction,
                    "position_norm_m": state.position_norm,
                    "health": eph.health,
                    "status": state.status if int(round(float(eph.health))) == 0 else "健康状态异常",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "epoch": epoch.isoformat(sep=" "),
                    "sat_id": sat_id,
                    "health": getattr(eph, "health", ""),
                    "status": f"计算失败：{exc}",
                }
            )
    return make_dataframe(rows, SATELLITE_DEBUG_COLUMNS)


def satellite_maps(table) -> Tuple[Dict[str, ECEF], Dict[str, float], Dict[str, float]]:
    rows = table.to_dict("records") if hasattr(table, "to_dict") else list(table)
    positions: Dict[str, ECEF] = {}
    clocks: Dict[str, float] = {}
    health: Dict[str, float] = {}
    for row in rows:
        try:
            if row.get("status") not in {"计算成功", "卫星坐标数量级异常"}:
                continue
            sat_id = row["sat_id"]
            health[sat_id] = float(row.get("health", 0.0))
            if int(round(health[sat_id])) != 0:
                continue
            positions[sat_id] = (float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))
            clocks[sat_id] = float(row["satellite_clock_bias_s"])
        except (TypeError, ValueError, KeyError):
            continue
    return positions, clocks, health


def save_satellite_debug(table, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "satellite_debug.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return path
