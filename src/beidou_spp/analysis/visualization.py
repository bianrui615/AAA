"""Standard figures for continuous positioning."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..positioning.coordinates import ecef_to_blh


def plot_standard_results(rows: List[Dict], receiver_ecef, output_dir: str | Path) -> List[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib_cache"))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []

    paths: List[Path] = []
    x_axis = list(range(len(rows)))
    errors = [float(row.get("error_3d_m", math.nan)) for row in rows]
    lats = [float(row.get("lat_deg", math.nan)) for row in rows]
    lons = [float(row.get("lon_deg", math.nan)) for row in rows]
    sats = [int(row.get("num_sats", 0) or 0) for row in rows]
    pdops = [float(row.get("PDOP", math.nan)) for row in rows]
    gdops = [float(row.get("GDOP", math.nan)) for row in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, errors, marker="o", linewidth=1.5)
    plt.xlabel("Epoch index")
    plt.ylabel("3D position error (m)")
    plt.title("Position Error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output / "position_error.png"
    plt.savefig(path, dpi=160)
    plt.savefig(output / "module4_error_curve.png", dpi=160)
    plt.close()
    paths.append(path)

    true_lat, true_lon, _ = ecef_to_blh(*receiver_ecef)
    plt.figure(figsize=(6, 6))
    plt.plot(lons, lats, marker="o", linewidth=1.2, label="Solved")
    plt.scatter([true_lon], [true_lat], marker="*", s=130, label="Reference")
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title("Trajectory")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output / "trajectory.png"
    plt.savefig(path, dpi=160)
    plt.savefig(output / "module4_trajectory.png", dpi=160)
    plt.close()
    paths.append(path)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x_axis, sats, marker="o", color="tab:blue", label="Satellites")
    ax1.set_xlabel("Epoch index")
    ax1.set_ylabel("Satellite count")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(x_axis, pdops, color="tab:orange", label="PDOP")
    ax2.plot(x_axis, gdops, color="tab:green", label="GDOP")
    ax2.set_ylabel("DOP")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    path = output / "dop_and_sat_count.png"
    plt.savefig(path, dpi=160)
    plt.savefig(output / "module4_satellite_dop_curve.png", dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths
