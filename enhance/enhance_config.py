"""
enhance_config.py

提高部分全局配置：场景定义、输出路径、特征列与标签列。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

ECEF = Tuple[float, float, float]


@dataclass
class ScenarioConfig:
    """多场景测试配置，用于生成机器学习数据集。"""

    name: str
    nav_file_path: str
    receiver_true_position: ECEF
    start_time: datetime
    end_time: datetime
    interval_seconds: int
    random_seed: int
    max_iter: int = 12
    convergence_threshold: float = 1e-4
    elevation_mask_deg: float = 0.0


# 输出目录统一放在 outputs/enhance/，避免与基础部分 outputs/basic/ 混淆
BASE_OUTPUT_DIR = Path("outputs/enhance")
SCENARIO_OUTPUT_DIR = BASE_OUTPUT_DIR / "scenarios"
MODEL_OUTPUT_DIR = BASE_OUTPUT_DIR / "models"
PREDICTION_OUTPUT_DIR = BASE_OUTPUT_DIR / "predictions"
FIGURE_OUTPUT_DIR = BASE_OUTPUT_DIR / "figures"

# 3 个不同场景，每场景 2 小时 × 30 秒采样 = 241 历元，总计约 720 样本
SCENARIOS: List[ScenarioConfig] = [
    ScenarioConfig(
        name="scenario_1_default",
        nav_file_path="nav/tarc0910.26b_cnav",
        receiver_true_position=(-2267800.0, 5009340.0, 3221000.0),
        start_time=datetime(2026, 4, 1, 0, 0, 0),
        end_time=datetime(2026, 4, 1, 2, 0, 0),
        interval_seconds=30,
        random_seed=2026,
        max_iter=12,
        convergence_threshold=1e-4,
        elevation_mask_deg=0.0,
    ),
    ScenarioConfig(
        name="scenario_2_different_seed",
        nav_file_path="nav/tarc1210.26b_cnav",
        receiver_true_position=(-2350000.0, 5100000.0, 3150000.0),
        start_time=datetime(2026, 5, 1, 0, 0, 0),
        end_time=datetime(2026, 5, 1, 2, 0, 0),
        interval_seconds=30,
        random_seed=42,
        max_iter=12,
        convergence_threshold=1e-4,
        elevation_mask_deg=0.0,
    ),
    ScenarioConfig(
        name="scenario_3_elevation_mask",
        nav_file_path="nav/tarc1230.26b_cnav",
        receiver_true_position=(-2200000.0, 4950000.0, 3300000.0),
        start_time=datetime(2026, 5, 3, 0, 0, 0),
        end_time=datetime(2026, 5, 3, 2, 0, 0),
        interval_seconds=30,
        random_seed=2027,
        max_iter=12,
        convergence_threshold=1e-4,
        elevation_mask_deg=10.0,
    ),
]

# 数值特征列（用于模型训练）
FEATURE_COLUMNS: List[str] = [
    "satellite_count",
    "raw_satellite_count",
    "PDOP",
    "GDOP",
    "clock_bias",
    "iteration_count",
    "elevation_mask_deg",
    "mean_pseudorange",
    "std_pseudorange",
    "mean_rho",
    "std_rho",
    "mean_sisre_error",
    "mean_iono_error",
    "mean_tropo_error",
    "mean_receiver_clock_error",
    "mean_noise_error",
    "mean_elevation_deg",
    "min_elevation_deg",
    "max_elevation_deg",
]

# 标签列（三维坐标误差）
LABEL_COLUMNS: List[str] = ["error_x", "error_y", "error_z"]
