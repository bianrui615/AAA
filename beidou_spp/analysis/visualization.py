"""连续定位的标准结果图。"""

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
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

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
    plt.xlabel("历元序号")
    plt.ylabel("三维定位误差 (m)")
    plt.title("定位误差曲线")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output / "position_error.png"
    plt.savefig(path, dpi=160)
    plt.savefig(output / "module4_error_curve.png", dpi=160)
    plt.close()
    paths.append(path)

    true_lat, true_lon, _ = ecef_to_blh(*receiver_ecef)
    plt.figure(figsize=(6, 6))
    plt.plot(lons, lats, marker="o", linewidth=1.2, label="解算轨迹")
    plt.scatter([true_lon], [true_lat], marker="*", s=130, label="参考位置")
    plt.xlabel("经度 (deg)")
    plt.ylabel("纬度 (deg)")
    plt.title("定位轨迹")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output / "trajectory.png"
    plt.savefig(path, dpi=160)
    plt.savefig(output / "module4_trajectory.png", dpi=160)
    plt.close()
    paths.append(path)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(x_axis, sats, marker="o", color="tab:blue", label="卫星数量")
    ax1.set_xlabel("历元序号")
    ax1.set_ylabel("卫星数量")
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
