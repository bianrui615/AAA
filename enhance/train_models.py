"""
train_models.py

读取 机器学习数据集.csv，按 scenario_name 做 Leave-One-Scenario-Out (LOSO)
交叉验证训练 LinearRegression 与 RandomForestRegressor 两个模型；
所有预处理放进 sklearn Pipeline；CV 结束后用全量数据训练最终发布模型；
同时输出零模型 (不补偿) baseline 与每折/汇总指标。
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from enhance.enhance_config import (
    BASE_OUTPUT_DIR,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    MODEL_OUTPUT_DIR,
)

MODEL_DISPLAY_NAMES = ("线性回归", "随机森林")


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


def _build_pipelines(random_state: int) -> Dict[str, Pipeline]:
    """构造两个 sklearn Pipeline：StandardScaler + 模型。

    Scaler 放进 Pipeline 后，CV 每折只会用训练 fold 拟合 scaler，
    避免测试 fold 的统计量泄漏。RandomForest 对缩放不敏感但保留 scaler
    可在以后切换到 Ridge/SVR 时无缝复用。
    """
    lr_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    rf_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=6,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return {"线性回归": lr_pipe, "随机森林": rf_pipe}


def _row_meta(row: dict) -> dict:
    return {
        "scenario_name": row.get("scenario_name", ""),
        "epoch_time": row.get("epoch_time", ""),
        "spp_x": row.get("spp_x", "0"),
        "spp_y": row.get("spp_y", "0"),
        "spp_z": row.get("spp_z", "0"),
        "true_x": row.get("true_x", "0"),
        "true_y": row.get("true_y", "0"),
        "true_z": row.get("true_z", "0"),
    }


def _err3d(spp: np.ndarray, true: np.ndarray) -> np.ndarray:
    """逐行 3D 欧氏距离。"""
    return np.linalg.norm(spp - true, axis=1)


def _fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spp_xyz: np.ndarray,
    true_xyz: np.ndarray,
) -> Dict[str, float]:
    """计算单折的核心指标。"""
    err_before = _err3d(spp_xyz, true_xyz)
    compensated = spp_xyz + y_pred
    err_after = _err3d(compensated, true_xyz)

    rmse_before = float(np.sqrt(np.mean(err_before ** 2)))
    rmse_after = float(np.sqrt(np.mean(err_after ** 2)))
    improvement = (
        (rmse_before - rmse_after) / rmse_before * 100.0
        if rmse_before > 1e-9
        else 0.0
    )

    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    try:
        r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    except ValueError:
        r2 = np.array([math.nan] * y_true.shape[1])

    return {
        "rmse_3d_before": rmse_before,
        "rmse_3d_after": rmse_after,
        "improvement_percent": improvement,
        "mae_x": float(mae[0]),
        "mae_y": float(mae[1]),
        "mae_z": float(mae[2]),
        "r2_x": float(r2[0]),
        "r2_y": float(r2[1]),
        "r2_z": float(r2[2]),
    }


def _summarize(folds: List[Dict[str, float]], keys: List[str]) -> Dict[str, Dict[str, float]]:
    """对一组折叠指标做 mean/std/min/max 汇总。"""
    summary: Dict[str, Dict[str, float]] = {}
    for k in keys:
        values = np.array([f[k] for f in folds if math.isfinite(f.get(k, math.nan))], dtype=float)
        if values.size == 0:
            summary[k] = {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
        else:
            summary[k] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return summary


def train_models(
    dataset_path: Path,
    random_state: int = 2026,
    enable_grid_search: bool = False,  # 保留入参兼容，但 LOSO 流程下不启用
    test_size: float = 0.3,  # 仅为向后兼容 GUI 旧签名；LOSO 流程下未使用
) -> Dict[str, Any]:
    """LOSO-CV 训练评估两个模型，并用全量数据训练最终发布模型。"""
    print("[train_models] 开始加载数据集并执行 LOSO 交叉验证...")
    rows, X, y = load_dataset(dataset_path)
    n_samples = X.shape[0]
    if n_samples < 10:
        raise RuntimeError(f"数据集样本数过少（{n_samples}），无法训练模型。")

    groups = np.array([r.get("scenario_name", "") for r in rows])
    unique_scenarios = sorted(set(groups))
    if len(unique_scenarios) < 2:
        raise RuntimeError(
            f"场景数过少（{len(unique_scenarios)}），LOSO-CV 至少需要 2 个场景。"
        )

    spp_xyz_all = np.array(
        [[float(r["spp_x"]), float(r["spp_y"]), float(r["spp_z"])] for r in rows],
        dtype=float,
    )
    true_xyz_all = np.array(
        [[float(r["true_x"]), float(r["true_y"]), float(r["true_z"])] for r in rows],
        dtype=float,
    )

    # Baseline (不补偿)：error_after == error_before
    err_before_all = _err3d(spp_xyz_all, true_xyz_all)
    baseline_metrics = {
        "rmse_3d": float(np.sqrt(np.mean(err_before_all ** 2))),
        "mean_3d": float(np.mean(err_before_all)),
        "max_3d": float(np.max(err_before_all)),
        "min_3d": float(np.min(err_before_all)),
    }
    print(
        f"[train_models] Baseline (不补偿): RMS={baseline_metrics['rmse_3d']:.4f} m, "
        f"Mean={baseline_metrics['mean_3d']:.4f} m"
    )

    pipelines = _build_pipelines(random_state)

    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)
    print(f"[train_models] 场景数 = {len(unique_scenarios)}，LOSO 折数 = {n_folds}")

    cv_fold_metrics: Dict[str, List[Dict[str, Any]]] = {name: [] for name in MODEL_DISPLAY_NAMES}
    oof_predictions: Dict[str, List[dict]] = {name: [] for name in MODEL_DISPLAY_NAMES}

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
        test_scenario = str(groups[test_idx][0])
        train_scenarios = sorted({str(s) for s in groups[train_idx]})
        print(
            f"[train_models] fold {fold_idx}: 训练场景={train_scenarios} "
            f"({len(train_idx)})  测试场景={test_scenario} ({len(test_idx)})"
        )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        spp_test = spp_xyz_all[test_idx]
        true_test = true_xyz_all[test_idx]

        for name, pipe in pipelines.items():
            pipe.fit(X_train, y_train)
            y_pred_test = pipe.predict(X_test)
            y_pred_train = pipe.predict(X_train)

            metrics = _fold_metrics(y_test, y_pred_test, spp_test, true_test)
            train_err_after = _err3d(spp_xyz_all[train_idx] + y_pred_train, true_xyz_all[train_idx])
            metrics["train_rmse_3d"] = float(np.sqrt(np.mean(train_err_after ** 2)))
            metrics["fold"] = fold_idx
            metrics["test_scenario"] = test_scenario
            metrics["n_train"] = int(len(train_idx))
            metrics["n_test"] = int(len(test_idx))
            cv_fold_metrics[name].append(metrics)

            for local_i, global_i in enumerate(test_idx):
                meta = _row_meta(rows[global_i])
                meta.update(
                    {
                        "fold_index": fold_idx,
                        "pred_error_x": float(y_pred_test[local_i, 0]),
                        "pred_error_y": float(y_pred_test[local_i, 1]),
                        "pred_error_z": float(y_pred_test[local_i, 2]),
                    }
                )
                oof_predictions[name].append(meta)

    metric_keys = [
        "rmse_3d_before",
        "rmse_3d_after",
        "improvement_percent",
        "mae_x",
        "mae_y",
        "mae_z",
        "r2_x",
        "r2_y",
        "r2_z",
        "train_rmse_3d",
    ]
    cv_summary_metrics: Dict[str, Dict[str, Dict[str, float]]] = {
        name: _summarize(cv_fold_metrics[name], metric_keys) for name in MODEL_DISPLAY_NAMES
    }

    # 训练最终发布模型：用全量数据 fit 一次
    print("[train_models] 训练最终发布模型（全量数据）...")
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_pipelines = _build_pipelines(random_state)
    lr_final = final_pipelines["线性回归"]
    rf_final = final_pipelines["随机森林"]
    lr_final.fit(X, y)
    rf_final.fit(X, y)
    lr_path = MODEL_OUTPUT_DIR / "线性回归模型.joblib"
    rf_path = MODEL_OUTPUT_DIR / "随机森林模型.joblib"
    joblib.dump(lr_final, lr_path)
    joblib.dump(rf_final, rf_path)
    print(f"[train_models] 线性回归模型已保存：{lr_path}")
    print(f"[train_models] 随机森林模型已保存：{rf_path}")

    # 输出 LOSO 划分摘要
    summary_path = BASE_OUTPUT_DIR / "LOSO_CV_划分汇总.txt"
    with summary_path.open("w", encoding="utf-8-sig") as f:
        f.write("LOSO 交叉验证划分摘要\n")
        f.write("=" * 40 + "\n")
        f.write(f"数据集文件：{dataset_path}\n")
        f.write(f"总样本数：{n_samples}\n")
        f.write(f"场景数（LOSO 折数）：{n_folds}\n")
        f.write(f"特征数量：{len(FEATURE_COLUMNS)}\n")
        f.write(f"标签数量：{len(LABEL_COLUMNS)} (error_x, error_y, error_z)\n")
        f.write(f"划分方式：LeaveOneGroupOut（按 scenario_name 留一场景验证）\n")
        f.write(f"Pipeline：StandardScaler + 模型（仅训练 fold 拟合 scaler）\n")
        f.write(f"随机种子：{random_state}\n\n")
        f.write("每折划分明细：\n")
        for fold_idx, (train_idx, test_idx) in enumerate(
            LeaveOneGroupOut().split(X, y, groups=groups)
        ):
            test_scenario = str(groups[test_idx][0])
            train_scenarios = sorted({str(s) for s in groups[train_idx]})
            f.write(
                f"  fold {fold_idx}: 训练={train_scenarios} ({len(train_idx)} 样本)，"
                f"测试={test_scenario} ({len(test_idx)} 样本)\n"
            )
        f.write(f"\n特征列表：{', '.join(FEATURE_COLUMNS)}\n")

    # 输出 CV 每折指标 CSV
    fold_metrics_csv = BASE_OUTPUT_DIR / "CV每折指标.csv"
    fold_fieldnames = [
        "model",
        "fold",
        "test_scenario",
        "n_train",
        "n_test",
        "rmse_3d_before",
        "rmse_3d_after",
        "improvement_percent",
        "train_rmse_3d",
        "mae_x",
        "mae_y",
        "mae_z",
        "r2_x",
        "r2_y",
        "r2_z",
    ]
    with fold_metrics_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fold_fieldnames)
        writer.writeheader()
        for name in MODEL_DISPLAY_NAMES:
            for m in cv_fold_metrics[name]:
                writer.writerow(
                    {
                        "model": name,
                        "fold": m["fold"],
                        "test_scenario": m["test_scenario"],
                        "n_train": m["n_train"],
                        "n_test": m["n_test"],
                        "rmse_3d_before": f"{m['rmse_3d_before']:.6f}",
                        "rmse_3d_after": f"{m['rmse_3d_after']:.6f}",
                        "improvement_percent": f"{m['improvement_percent']:.4f}",
                        "train_rmse_3d": f"{m['train_rmse_3d']:.6f}",
                        "mae_x": f"{m['mae_x']:.6f}",
                        "mae_y": f"{m['mae_y']:.6f}",
                        "mae_z": f"{m['mae_z']:.6f}",
                        "r2_x": f"{m['r2_x']:.6f}",
                        "r2_y": f"{m['r2_y']:.6f}",
                        "r2_z": f"{m['r2_z']:.6f}",
                    }
                )

    # 输出 CV 汇总指标 CSV / TXT（含 baseline 行）
    summary_csv = BASE_OUTPUT_DIR / "CV汇总指标.csv"
    summary_txt = BASE_OUTPUT_DIR / "CV汇总指标.txt"
    summary_fieldnames = [
        "model",
        "rmse_3d_before_mean",
        "rmse_3d_after_mean",
        "rmse_3d_after_std",
        "improvement_percent_mean",
        "improvement_percent_std",
        "train_rmse_3d_mean",
        "mae_x_mean",
        "mae_y_mean",
        "mae_z_mean",
        "r2_x_mean",
        "r2_y_mean",
        "r2_z_mean",
        "beats_baseline",
    ]
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "model": "Baseline_不补偿",
                "rmse_3d_before_mean": f"{baseline_metrics['rmse_3d']:.6f}",
                "rmse_3d_after_mean": f"{baseline_metrics['rmse_3d']:.6f}",
                "rmse_3d_after_std": "0.000000",
                "improvement_percent_mean": "0.0000",
                "improvement_percent_std": "0.0000",
                "train_rmse_3d_mean": "",
                "mae_x_mean": "",
                "mae_y_mean": "",
                "mae_z_mean": "",
                "r2_x_mean": "",
                "r2_y_mean": "",
                "r2_z_mean": "",
                "beats_baseline": "",
            }
        )
        for name in MODEL_DISPLAY_NAMES:
            s = cv_summary_metrics[name]
            beats = s["rmse_3d_after"]["mean"] < baseline_metrics["rmse_3d"]
            writer.writerow(
                {
                    "model": name,
                    "rmse_3d_before_mean": f"{s['rmse_3d_before']['mean']:.6f}",
                    "rmse_3d_after_mean": f"{s['rmse_3d_after']['mean']:.6f}",
                    "rmse_3d_after_std": f"{s['rmse_3d_after']['std']:.6f}",
                    "improvement_percent_mean": f"{s['improvement_percent']['mean']:.4f}",
                    "improvement_percent_std": f"{s['improvement_percent']['std']:.4f}",
                    "train_rmse_3d_mean": f"{s['train_rmse_3d']['mean']:.6f}",
                    "mae_x_mean": f"{s['mae_x']['mean']:.6f}",
                    "mae_y_mean": f"{s['mae_y']['mean']:.6f}",
                    "mae_z_mean": f"{s['mae_z']['mean']:.6f}",
                    "r2_x_mean": f"{s['r2_x']['mean']:.6f}",
                    "r2_y_mean": f"{s['r2_y']['mean']:.6f}",
                    "r2_z_mean": f"{s['r2_z']['mean']:.6f}",
                    "beats_baseline": "YES" if beats else "NO",
                }
            )

    with summary_txt.open("w", encoding="utf-8-sig") as f:
        f.write("LOSO 交叉验证汇总指标\n")
        f.write("=" * 50 + "\n\n")
        f.write(
            f"数据集：{n_samples} 样本，{n_folds} 个场景（LOSO）\n"
            f"特征数：{len(FEATURE_COLUMNS)}，标签：error_x/y/z\n\n"
        )
        f.write("Baseline (不补偿)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  3D RMS = {baseline_metrics['rmse_3d']:.4f} m\n")
        f.write(f"  3D Mean = {baseline_metrics['mean_3d']:.4f} m\n")
        f.write(f"  3D Max  = {baseline_metrics['max_3d']:.4f} m\n\n")
        for name in MODEL_DISPLAY_NAMES:
            s = cv_summary_metrics[name]
            beats = s["rmse_3d_after"]["mean"] < baseline_metrics["rmse_3d"]
            f.write(f"{name}\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"  RMS_after  = {s['rmse_3d_after']['mean']:.4f} ± "
                f"{s['rmse_3d_after']['std']:.4f} m  (LOSO mean ± std)\n"
            )
            f.write(
                f"  改善百分比 = {s['improvement_percent']['mean']:.2f}% ± "
                f"{s['improvement_percent']['std']:.2f}%\n"
            )
            f.write(
                f"  训练 RMS_after = {s['train_rmse_3d']['mean']:.4f} m   "
                f"(对比验证 RMS={s['rmse_3d_after']['mean']:.4f} m，差距越大越过拟合)\n"
            )
            f.write(
                f"  MAE (x/y/z) = {s['mae_x']['mean']:.3f} / {s['mae_y']['mean']:.3f} / "
                f"{s['mae_z']['mean']:.3f} m\n"
            )
            f.write(
                f"  R² (x/y/z)  = {s['r2_x']['mean']:.3f} / {s['r2_y']['mean']:.3f} / "
                f"{s['r2_z']['mean']:.3f}\n"
            )
            f.write(f"  是否优于 Baseline (不补偿)：{'是' if beats else '否'}\n\n")
        f.write(
            "注：N=3 折 LOSO 下，std 估计极不稳定，模型间差异是否显著需更多场景方能确认。\n"
        )

    return {
        "oof_predictions": oof_predictions,
        "cv_fold_metrics": cv_fold_metrics,
        "cv_summary_metrics": cv_summary_metrics,
        "baseline_metrics": baseline_metrics,
        "lr_path": lr_path,
        "rf_path": rf_path,
        "n_total": n_samples,
        "n_folds": n_folds,
        "summary_path": summary_path,
        "fold_metrics_csv": fold_metrics_csv,
        "summary_metrics_csv": summary_csv,
        "summary_metrics_txt": summary_txt,
    }


if __name__ == "__main__":
    dataset_path = BASE_OUTPUT_DIR / "机器学习数据集.csv"
    train_models(dataset_path)
