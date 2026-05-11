"""Coordinate conversion helpers."""

from __future__ import annotations

import math
from typing import Tuple

from module3_spp_solver import ecef_to_blh as _ecef_to_blh

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def ecef_to_blh(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert ECEF meters to latitude/longitude degrees and height meters."""

    return _ecef_to_blh(x, y, z)


def blh_to_ecef(lat_deg: float, lon_deg: float, height_m: float) -> Tuple[float, float, float]:
    """Convert geodetic latitude/longitude/height to WGS84 ECEF meters."""

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + height_m) * cos_lat * math.cos(lon)
    y = (n + height_m) * cos_lat * math.sin(lon)
    z = (n * (1.0 - WGS84_E2) + height_m) * sin_lat
    return x, y, z

