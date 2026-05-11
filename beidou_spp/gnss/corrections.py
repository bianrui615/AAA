"""Pseudorange corrections: satellite clock, ionosphere and troposphere."""

from __future__ import annotations

from pathlib import Path
from typing import List

from module3_spp_solver import ECEF, saastamoinen_troposphere_delay, simple_ionosphere_delay

from ..table import make_dataframe


CORRECTED_COLUMNS = [
    "epoch",
    "sat_id",
    "raw_pseudorange_m",
    "satellite_clock_correction_m",
    "ionosphere_delay_m",
    "troposphere_delay_m",
    "corrected_pseudorange_m",
]


def correct_pseudorange(pseudorange_table, receiver_ecef: ECEF) -> object:
    """Apply basic measurement corrections.

    Formula used by the SPP solver:
        P_corr = P_raw + c * dts - I - T

    where P_raw is raw pseudorange in meters, dts is satellite clock bias in
    seconds, c is speed of light in m/s, I is ionosphere delay in meters, and T
    is Saastamoinen troposphere delay in meters.
    """

    rows = pseudorange_table.to_dict("records") if hasattr(pseudorange_table, "to_dict") else list(pseudorange_table)
    corrected: List[dict] = []
    for row in rows:
        sat_pos = (float(row["sat_x_m"]), float(row["sat_y_m"]), float(row["sat_z_m"]))
        iono = simple_ionosphere_delay(sat_pos, receiver_ecef)
        trop = saastamoinen_troposphere_delay(sat_pos, receiver_ecef)
        clock_corr = float(row.get("satellite_clock_correction_m", 0.0) or 0.0)
        raw = float(row.get("raw_pseudorange_m", row.get("simulated_pseudorange_m")))
        corrected.append(
            {
                "epoch": row["epoch"],
                "sat_id": row["sat_id"],
                "raw_pseudorange_m": raw,
                "satellite_clock_correction_m": clock_corr,
                "ionosphere_delay_m": iono,
                "troposphere_delay_m": trop,
                "corrected_pseudorange_m": raw + clock_corr - iono - trop,
            }
        )
    return make_dataframe(corrected, CORRECTED_COLUMNS)


def save_corrected_pseudorange(table, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "corrected_pseudorange.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return path

