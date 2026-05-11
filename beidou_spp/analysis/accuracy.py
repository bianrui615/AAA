"""精度统计。"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


def finite(values):
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def accuracy_stats(rows: List[Dict]) -> Dict[str, float]:
    errors = finite(row.get("error_3d_m", math.nan) for row in rows)
    if not errors:
        return {"mean_error_m": math.nan, "rms_error_m": math.nan, "max_error_m": math.nan}
    return {
        "mean_error_m": float(np.mean(errors)),
        "rms_error_m": float(math.sqrt(np.mean(np.square(errors)))),
        "max_error_m": float(max(errors)),
    }
