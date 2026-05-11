"""Configuration objects for the Beidou SPP pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

ECEF = Tuple[float, float, float]


@dataclass
class PipelineConfig:
    """User-facing configuration for one full positioning run."""

    nav: Path
    output: Path = Path("outputs")
    receiver_ecef: ECEF = (-2267800.0, 5009340.0, 3221000.0)
    start: datetime = datetime(2026, 4, 1, 0, 0, 0)
    end: datetime = datetime(2026, 4, 1, 1, 0, 0)
    interval: int = 300
    seed: int = 2026
    max_iter: int = 10
    threshold: float = 1e-4
    elevation_mask: float = 15.0

