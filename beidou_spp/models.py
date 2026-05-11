"""北斗 SPP 软件包共用数据模型。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Tuple

ECEF = Tuple[float, float, float]
C = 299_792_458.0


@dataclass
class EpochSolution:
    """使用课程要求字段名的标准单历元 SPP 结果。"""

    epoch: str
    x_m: float = math.nan
    y_m: float = math.nan
    z_m: float = math.nan
    lat_deg: float = math.nan
    lon_deg: float = math.nan
    height_m: float = math.nan
    receiver_clock_bias_m: float = math.nan
    receiver_clock_bias_s: float = math.nan
    num_sats: int = 0
    GDOP: float = math.nan
    PDOP: float = math.nan
    HDOP: float = math.nan
    VDOP: float = math.nan
    TDOP: float = math.nan
    converged: bool = False
    iterations: int = 0
    message: str = ""

    def as_dict(self) -> Dict:
        return {
            "epoch": self.epoch,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "z_m": self.z_m,
            "lat_deg": self.lat_deg,
            "lon_deg": self.lon_deg,
            "height_m": self.height_m,
            "receiver_clock_bias_m": self.receiver_clock_bias_m,
            "receiver_clock_bias_s": self.receiver_clock_bias_s,
            "num_sats": self.num_sats,
            "GDOP": self.GDOP,
            "PDOP": self.PDOP,
            "HDOP": self.HDOP,
            "VDOP": self.VDOP,
            "TDOP": self.TDOP,
            "converged": self.converged,
            "iterations": self.iterations,
            "message": self.message,
        }


@dataclass
class PipelineResult:
    """一次完整运行生成的文件路径和总体统计结果。"""

    output_dir: Path
    files: Dict[str, Path]
    epoch_results: List[Dict]
    success_epochs: int
    total_epochs: int
    rms_error_m: float
    mean_error_m: float
    max_error_m: float
