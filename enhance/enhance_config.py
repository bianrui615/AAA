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
    convergence_threshold: float = 1e-2
    elevation_mask_deg: float = 0.0


# 输出目录统一放在 outputs/enhance/，txt/csv 类机器学习结果集中放在 ml/ 下。
ENHANCE_OUTPUT_ROOT = Path("outputs/enhance")
ML_OUTPUT_DIR = ENHANCE_OUTPUT_ROOT / "ml"
BASE_OUTPUT_DIR = ML_OUTPUT_DIR
SCENARIO_OUTPUT_DIR = ML_OUTPUT_DIR / "scenarios"
MODEL_OUTPUT_DIR = ENHANCE_OUTPUT_ROOT / "models"
PREDICTION_OUTPUT_DIR = ML_OUTPUT_DIR / "predictions"
FIGURE_OUTPUT_DIR = ENHANCE_OUTPUT_ROOT / "figures"

# 默认场景集合：9 个场景，覆盖不同 NAV 文件 / 接收机位置 / 时段 / 随机种子 /
# 高度角掩膜，用于 LOSO 交叉验证下让模型见到更多分布。
#
# 前 3 个场景与 basic/gui_scenario_runner.py 保持一致，若已有基础部分结果会
# 优先读取；scenario4-9 是 enhance 专属，CSV 不存在时 dataset_builder 会
# 自动用 NAV + 模拟伪距重新生成（约 0.5-1s/场景）。
#
# 用户可在 enhance GUI「场景配置」标签页中勾选/编辑/新增/删除场景，
# 「构建数据集」按钮只会跑勾选的场景。
SCENARIOS: List[ScenarioConfig] = [
    ScenarioConfig(
        name="scenario1",
        nav_file_path="nav/tarc0910.26b_cnav",
        receiver_true_position=(-2267800.0, 5009340.0, 3221000.0),
        start_time=datetime(2026, 4, 1, 0, 0, 0),
        end_time=datetime(2026, 4, 1, 6, 0, 0),
        interval_seconds=300,
        random_seed=2026,
        max_iter=12,
        convergence_threshold=1e-2,
        elevation_mask_deg=0.0,
    ),
    ScenarioConfig(
        name="scenario2",
        nav_file_path="nav/tarc1220.26b_cnav",
        receiver_true_position=(-2267800.0, 5009340.0, 3221000.0),
        start_time=datetime(2026, 5, 2, 0, 0, 0),
        end_time=datetime(2026, 5, 2, 6, 0, 0),
        interval_seconds=300,
        random_seed=2026,
        max_iter=12,
        convergence_threshold=1e-2,
        elevation_mask_deg=0.0,
    ),
    ScenarioConfig(
        name="scenario3",
        nav_file_path="nav/tarc1230.26b_cnav",
        receiver_true_position=(-2267800.0, 5009340.0, 3221000.0),
        start_time=datetime(2026, 5, 3, 0, 0, 0),
        end_time=datetime(2026, 5, 3, 6, 0, 0),
        interval_seconds=300,
        random_seed=2026,
        max_iter=12,
        convergence_threshold=1e-2,
        elevation_mask_deg=0.0,
    ),
    ScenarioConfig(
        name="scenario4",
        nav_file_path="nav/tarc1240.26b_cnav",
        receiver_true_position=(-2167800.0, 5109340.0, 3221000.0),
        start_time=datetime(2026, 5, 4, 0, 0, 0),
        end_time=datetime(2026, 5, 4, 6, 0, 0),
        interval_seconds=300,
        random_seed=2027,
        max_iter=12,
        convergence_threshold=1e-2,
        elevation_mask_deg=5.0,
    ),
    ScenarioConfig(
        name="scenario5",
        nav_file_path="nav/tarc1250.26b_cnav",
        receiver_true_position=(-2367800.0, 4909340.0, 3221000.0),
        start_time=datetime(2026, 5, 5, 6, 0, 0),
        end_time=datetime(2026, 5, 5, 12, 0, 0),
        interval_seconds=300,
        random_seed=2028,
        max_iter=12,
        convergence_threshold=1e-2,
        elevation_mask_deg=5.0,
    ),
    ScenarioConfig(
        name="scenario6",
        nav_file_path="nav/tarc1260.26b_cnav",
        receiver_true_position=(-2267800.0, 5009340.0, 3321000.0),
        start_time=datetime(2026, 5, 6, 12, 0, 0),
        end_time=datetime(2026, 5, 6, 18, 0, 0),
        interval_seconds=300,
        random_seed=2029,
        max_iter=12,
        convergence_threshold=1e-2,
        elevation_mask_deg=10.0,
    ),
    ScenarioConfig(
        name="scenario7",
        nav_file_path="nav/tarc1270.26b_cnav",
        receiver_true_position=(-2167800.0, 4909340.0, 3121000.0),
        start_time=datetime(2026, 5, 7, 18, 0, 0),
        end_time=datetime(2026, 5, 7, 23, 59, 0),
        interval_seconds=300,
        random_seed=2030,
        max_iter=12,
        convergence_threshold=1e-2,
        elevation_mask_deg=10.0,
    ),
    ScenarioConfig(
        name="scenario8",
        nav_file_path="nav/tarc1280.26b_cnav",
        receiver_true_position=(-2367800.0, 5109340.0, 3221000.0),
        start_time=datetime(2026, 5, 8, 0, 0, 0),
        end_time=datetime(2026, 5, 8, 6, 0, 0),
        interval_seconds=300,
        random_seed=2031,
        max_iter=12,
        convergence_threshold=1e-2,
        elevation_mask_deg=15.0,
    ),
    ScenarioConfig(
        name="scenario9",
        nav_file_path="nav/tarc1290.26b_cnav",
        receiver_true_position=(-2267800.0, 5009340.0, 3121000.0),
        start_time=datetime(2026, 5, 9, 12, 0, 0),
        end_time=datetime(2026, 5, 9, 18, 0, 0),
        interval_seconds=300,
        random_seed=2032,
        max_iter=12,
        convergence_threshold=1e-2,
        elevation_mask_deg=15.0,
    ),
]

# 数值特征列（用于模型训练）
# 注意：为避免数据泄漏，只使用接收端可观测或解算过程中可获得的特征，
# 不使用伪距模拟误差的真值（如 mean_sisre_error、mean_iono_error 等）。
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
    "mean_elevation_deg",
    "min_elevation_deg",
    "max_elevation_deg",
]

# 额外分析列（仅用于 机器学习数据集.csv 中的误差分析，不作为模型输入特征）
EXTRA_ANALYSIS_COLUMNS: List[str] = [
    "mean_sisre_error",
    "mean_iono_error",
    "mean_tropo_error",
    "mean_receiver_clock_error",
    "mean_noise_error",
]

# 标签列（三维坐标误差）
LABEL_COLUMNS: List[str] = ["error_x", "error_y", "error_z"]
