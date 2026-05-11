"""Satellite clock correction helpers."""

from __future__ import annotations

C = 299_792_458.0


def clock_bias_to_range(clock_bias_s: float) -> float:
    """Convert satellite clock bias from seconds to meters: c * dt."""

    return C * clock_bias_s

