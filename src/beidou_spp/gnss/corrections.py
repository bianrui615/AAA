"""伪距修正：卫星钟差、电离层和对流层。"""

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
    """应用基础伪距修正。

    SPP 解算使用的修正公式：
        P_corr = P_raw + c * dts - I - T

    其中 P_raw 为原始伪距，单位 m；dts 为卫星钟差，单位 s；c 为光速，
    单位 m/s；I 为电离层延迟，单位 m；T 为 Saastamoinen 对流层延迟，
    单位 m。
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
