"""DOP 计算工具。"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from module3_spp_solver import ECEF, satellite_elevation_deg
from ..positioning.coordinates import ecef_to_blh


def design_matrix(satellite_positions: Dict[str, ECEF], receiver_ecef: ECEF) -> np.ndarray:
    rows = []
    x, y, z = receiver_ecef
    for sat in satellite_positions.values():
        dx = sat[0] - x
        dy = sat[1] - y
        dz = sat[2] - z
        rho = math.sqrt(dx * dx + dy * dy + dz * dz)
        if rho <= 0:
            continue
        rows.append([-dx / rho, -dy / rho, -dz / rho, 1.0])
    return np.asarray(rows, dtype=float)


def compute_dops_from_design(h_matrix: np.ndarray) -> Tuple[float, float, float, float, float]:
    """根据几何设计矩阵返回 GDOP、PDOP、HDOP、VDOP 和 TDOP。"""

    if h_matrix.shape[0] < 4:
        return (math.nan, math.nan, math.nan, math.nan, math.nan)
    q = np.linalg.inv(h_matrix.T @ h_matrix)
    hdop = math.sqrt(max(q[0, 0] + q[1, 1], 0.0))
    vdop = math.sqrt(max(q[2, 2], 0.0))
    pdop = math.sqrt(max(q[0, 0] + q[1, 1] + q[2, 2], 0.0))
    tdop = math.sqrt(max(q[3, 3], 0.0))
    gdop = math.sqrt(max(pdop * pdop + tdop * tdop, 0.0))
    return gdop, pdop, hdop, vdop, tdop


def compute_dops(satellite_positions: Dict[str, ECEF], receiver_ecef: ECEF) -> Tuple[float, float, float, float, float]:
    return compute_dops_from_design(design_matrix(satellite_positions, receiver_ecef))
