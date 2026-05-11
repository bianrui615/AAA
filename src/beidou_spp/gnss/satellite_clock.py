"""卫星钟差修正工具。"""

from __future__ import annotations

C = 299_792_458.0


def clock_bias_to_range(clock_bias_s: float) -> float:
    """将卫星钟差由秒转换为米：c * dt。"""

    return C * clock_bias_s
