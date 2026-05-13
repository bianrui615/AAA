"""
train_models.py

读取 ml_dataset.csv，划分训练集/测试集，
训练 LinearRegression 与 RandomForestRegressor 两个模型，
并保存模型文件与划分摘要。
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

from enhance.enhance_config import (
    BASE_OUTPUT_DIR,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    MODEL_OUTPUT_DIR,
)


def load_dataset(dataset_path: Path) -> Tuple[List[dict], np.ndarray, np.ndarray]:
    """读取数据集并返回原始行、特征矩阵 X 和标签矩阵 y。"""
    rows: List[dict] = []
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    X = np.zeros((len(rows), len(FEATURE_COLUMNS)), dtype=float)
    y = np.zeros((len(rows), len(LABEL_COLUMNS)), dtype=float)

    for i, row in enumerate(rows):
        for j, col in enumerate(FEATURE_COLUMNS):
            val = row.get(col, "")
            X[i, j] = float(val) if val != "" else 0.0
        for j, col in enumerate(LABEL_COLUMNS):
            val = row.get(col, "")
            y[i, j] = float(val) if val != "" else 0.0

    return rows, X, y


def _scenario_based_split(
    rows: List[dict],
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.3,
    random_state: int = 2026,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按 scenario_name 整场景划分训练集与测试集。

    将场景列表随机打乱后按 test_ratio 分配：整个场景要么全部进训练集，
    要么全部进测试集，避免同一场景的相邻历元同时出现在训练和测试中。

    返回: (X_train, X_test, y_train, y_test, idx_train, idx_test)
    """
    scenario_names = sorted(set(row.get("scenario_name", "") for row in rows))
    n_scenarios = len(scenario_names)

    rng = random.Random(random_state)
    shuffled = list(scenario_names)
    rng.shuffle(shuffled)

    n_test_scenarios = max(1, round(n_scenarios * test_ratio))
    test_set = set(shuffled[:n_test_scenarios])

    idx_train = np.array([i for i, r in enumerate(rows) if r.get("scenario_name", "") not in test_set])
    idx_test = np.array([i for i, r in enumerate(rows) if r.get("scenario_name", "") in test_set])

    if len(idx_train) == 0 or len(idx_test) == 0:
        raise RuntimeError(
            "场景划分后训练集或测试集为空，请检查场景数量是否足够（至少 2 个场景）。"
        )

    train_scenarios = [s for s in shuffled if s not in test_set]
    print(
        f"[train_models] 场景划分：训练场景 {train_scenarios}，"
        f"测试场景 {sorted(test_set)}"
    )
    return X[idx_train], X[idx_test], y[idx_train], y[idx_test], idx_train, idx_test


def train_models(
    dataset_path: Path,
    test_size: float = 0.3,
    random_state: int = 2026,
    enable_grid_search: bool = False,
) -> Dict[str, any]:
    """训练线性回归和随机森林模型，保存模型并返回划分信息及测试集元数据。

    参数:
        dataset_path: ml_dataset.csv 路径
        test_size: 测试集比例，默认 0.3（按场景整体划分，非随机样本划分）
        random_state: 随机种子
        enable_grid_search: 是否启用 GridSearchCV 调优随机森林超参数（默认关闭）
    """
    print("[train_models] 开始加载数据集并训练模型...")
    rows, X, y = load_dataset(dataset_path)
    n_samples = X.shape[0]

    if n_samples < 10:
        raise RuntimeError(f"数据集样本数过少（{n_samples}），无法训练模型。")

    # 按 scenario_name 整场景划分，避免同场景相邻历元同时进入训练集和测试集
    X_train, X_test, y_train, y_test, idx_train, idx_test = _scenario_based_split(
        rows, X, y, test_ratio=test_size, random_state=random_state
    )

    # 提取测试集元数据（用于后续补偿）
    test_metadata: List[dict] = []
    for i in idx_test:
        row = rows[i]
        test_metadata.append(
            {
                "scenario_name": row.get("scenario_name", ""),
                "epoch_time": row.get("epoch_time", ""),
                "spp_x": row.get("spp_x", "0"),
                "spp_y": row.get("spp_y", "0"),
                "spp_z": row.get("spp_z", "0"),
                "true_x": row.get("true_x", "0"),
                "true_y": row.get("true_y", "0"),
                "true_z": row.get("true_z", "0"),
            }
        )

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 线性回归
    print("[train_models] 训练 LinearRegression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_path = MODEL_OUTPUT_DIR / "linear_regression_model.joblib"
    joblib.dump(lr_model, lr_path)
    print(f"[train_models] LinearRegression 已保存：{lr_path}")

    # 2. 随机森林（可选 GridSearchCV 超参数调优）
    if enable_grid_search:
        print("[train_models] 启用 GridSearchCV 调优 RandomForestRegressor...")
        param_grid = {
            "n_estimators": [100, 200, 400],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
        }
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=random_state, n_jobs=-1),
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        rf_grid.fit(X_train, y_train)
        rf_model = rf_grid.best_estimator_
        print(f"[train_models] GridSearch 最佳参数：{rf_grid.best_params_}")
    else:
        print("[train_models] 训练 RandomForestRegressor...")
        rf_model = RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            max_depth=10,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)
    rf_path = MODEL_OUTPUT_DIR / "random_forest_model.joblib"
    joblib.dump(rf_model, rf_path)
    print(f"[train_models] RandomForestRegressor 已保存：{rf_path}")

    # 保存划分摘要
    summary_path = BASE_OUTPUT_DIR / "train_test_split_summary.txt"
    with summary_path.open("w", encoding="utf-8-sig") as f:
        f.write("机器学习数据集划分摘要\n")
        f.write("=" * 40 + "\n")
        f.write(f"数据集文件：{dataset_path}\n")
        f.write(f"总样本数：{n_samples}\n")
        f.write(f"训练集样本数：{X_train.shape[0]}\n")
        f.write(f"测试集样本数：{X_test.shape[0]}\n")
        f.write(f"特征数量：{len(FEATURE_COLUMNS)}\n")
        f.write(f"标签数量：{len(LABEL_COLUMNS)} (error_x, error_y, error_z)\n")
        f.write(f"测试集比例（按场景）：约 {test_size * 100:.0f}%\n")
        f.write(f"划分方式：按 scenario_name 整场景划分，避免同场景相邻历元数据泄漏\n")
        f.write(f"划分随机种子：{random_state}\n")
        f.write(f"特征列表：{', '.join(FEATURE_COLUMNS)}\n")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "test_metadata": test_metadata,
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "lr_model": lr_model,
        "rf_model": rf_model,
        "lr_path": lr_path,
        "rf_path": rf_path,
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    dataset_path = BASE_OUTPUT_DIR / "ml_dataset.csv"
    train_models(dataset_path)
