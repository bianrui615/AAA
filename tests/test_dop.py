import math

import numpy as np

from beidou_spp.gnss.dop import compute_dops_from_design


def test_dop_values_are_finite_positive():
    """固定几何矩阵计算出的各类 DOP 应为有限正数。"""

    h = np.array(
        [
            [-0.5, -0.4, -0.7, 1.0],
            [0.6, -0.3, -0.7, 1.0],
            [-0.2, 0.7, -0.6, 1.0],
            [0.4, 0.5, -0.75, 1.0],
            [-0.7, 0.1, -0.7, 1.0],
        ]
    )
    gdop, pdop, hdop, vdop, tdop = compute_dops_from_design(h)
    for value in (gdop, pdop, hdop, vdop, tdop):
        assert math.isfinite(value)
        assert value > 0
