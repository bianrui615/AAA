"""RINEX NAV parser wrapper with standard debug output."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from module1_nav_parser import (
    BroadcastEphemeris,
    NavParseInfo,
    parse_rinex_nav_with_info,
    save_nav_parse_outputs,
    select_ephemeris,
)

from ..table import make_dataframe


NAV_DEBUG_COLUMNS = [
    "sat_id",
    "toc_time",
    "toe_s",
    "af0_s",
    "af1_s_per_s",
    "af2_s_per_s2",
    "sqrt_a_sqrt_m",
    "eccentricity",
    "i0_rad",
    "omega0_rad",
    "omega_rad",
    "m0_rad",
    "delta_n_rad_s",
    "health",
    "is_healthy",
]


def parse_nav_file(nav_path: str | Path) -> Dict[str, List[BroadcastEphemeris]]:
    """Parse a RINEX NAV file and return Beidou broadcast ephemerides."""

    nav_data, _ = parse_rinex_nav_with_info(nav_path)
    return nav_data


def parse_nav_file_with_info(
    nav_path: str | Path,
) -> Tuple[Dict[str, List[BroadcastEphemeris]], NavParseInfo]:
    """Parse a RINEX NAV file and keep parser diagnostics."""

    return parse_rinex_nav_with_info(nav_path)


def nav_debug_rows(nav_data: Dict[str, List[BroadcastEphemeris]]) -> List[dict]:
    rows: List[dict] = []
    for sat_id in sorted(nav_data):
        for eph in nav_data[sat_id]:
            rows.append(
                {
                    "sat_id": eph.sat_id,
                    "toc_time": eph.toc.isoformat(sep=" "),
                    "toe_s": eph.toe,
                    "af0_s": eph.af0,
                    "af1_s_per_s": eph.af1,
                    "af2_s_per_s2": eph.af2,
                    "sqrt_a_sqrt_m": eph.sqrt_a,
                    "eccentricity": eph.eccentricity,
                    "i0_rad": eph.i0,
                    "omega0_rad": eph.omega0,
                    "omega_rad": eph.omega,
                    "m0_rad": eph.m0,
                    "delta_n_rad_s": eph.delta_n,
                    "health": eph.health,
                    "is_healthy": int(round(float(eph.health))) == 0,
                }
            )
    return rows


def save_parsed_nav_debug(
    nav_data: Dict[str, List[BroadcastEphemeris]],
    output_dir: str | Path,
    parse_info: NavParseInfo | None = None,
) -> Path:
    """Save standard module-1 parser output: parsed_nav_debug.csv."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_nav_parse_outputs(nav_data, output, parse_info)
    table = make_dataframe(nav_debug_rows(nav_data), NAV_DEBUG_COLUMNS)
    path = output / "parsed_nav_debug.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return path

