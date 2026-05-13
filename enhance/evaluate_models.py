"""
evaluate_models.py

评估线性回归和随机森林的补偿效果，
输出对比统计 CSV、TXT 报告以及可视化图表。
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from enhance.enhance_config import BASE_OUTPUT_DIR, FIGURE_OUTPUT_DIR

# 设置 matplotlib 缓存与字体（避免中文乱码）
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/enhance") / "matplotlib_cache"))
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def _safe_float(value) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def read_prediction_csv(csv_path: Path) -> List[dict]:
    rows: List[dict] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_metrics(rows: List[dict]) -> Dict[str, float]:
    before = [_safe_float(r["error_before"]) for r in rows]
    after = [_safe_float(r["error_after"]) for r in rows]

    before = [v for v in before if math.isfinite(v)]
    after = [v for v in after if math.isfinite(v)]

    if not before or not after:
        return {}

    mean_before = float(np.mean(before))
    mean_after = float(np.mean(after))
    rms_before = float(np.sqrt(np.mean(np.square(before))))
    rms_after = float(np.sqrt(np.mean(np.square(after))))
    max_before = float(np.max(before))
    max_after = float(np.max(after))
    min_before = float(np.min(before))
    min_after = float(np.min(after))

    improvement_mean = (
        (mean_before - mean_after) / mean_before * 100.0 if mean_before > 1e-9 else 0.0
    )
    improvement_rms = (
        (rms_before - rms_after) / rms_before * 100.0 if rms_before > 1e-9 else 0.0
    )
    improvement_max = (
        (max_before - max_after) / max_before * 100.0 if max_before > 1e-9 else 0.0
    )

    return {
        "mean_error_before": mean_before,
        "mean_error_after": mean_after,
        "rms_error_before": rms_before,
        "rms_error_after": rms_after,
        "max_error_before": max_before,
        "max_error_after": max_after,
        "min_error_before": min_before,
        "min_error_after": min_after,
        "improvement_mean_percent": improvement_mean,
        "improvement_rms_percent": improvement_rms,
        "improvement_max_percent": improvement_max,
    }


def evaluate_and_visualize(
    prediction_paths: Dict[str, Path],
    n_train: int,
    n_test: int,
    feature_columns: List[str],
) -> Dict[str, Path]:
    """评估两个模型，输出统计文件与可视化图表。"""
    print("[evaluate_models] 开始评估模型效果...")
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lr_rows = read_prediction_csv(prediction_paths["linear_regression"])
    rf_rows = read_prediction_csv(prediction_paths["random_forest"])

    lr_metrics = compute_metrics(lr_rows)
    rf_metrics = compute_metrics(rf_rows)

    # 1. 模型对比 summary CSV
    summary_csv = BASE_OUTPUT_DIR / "模型对比汇总.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "mean_error_before",
            "mean_error_after",
            "rms_error_before",
            "rms_error_after",
            "max_error_before",
            "max_error_after",
            "min_error_before",
            "min_error_after",
            "improvement_mean_percent",
            "improvement_rms_percent",
            "improvement_max_percent",
        ])
        for name, metrics in [("LinearRegression", lr_metrics), ("RandomForest", rf_metrics)]:
            writer.writerow([
                name,
                metrics.get("mean_error_before", math.nan),
                metrics.get("mean_error_after", math.nan),
                metrics.get("rms_error_before", math.nan),
                metrics.get("rms_error_after", math.nan),
                metrics.get("max_error_before", math.nan),
                metrics.get("max_error_after", math.nan),
                metrics.get("min_error_before", math.nan),
                metrics.get("min_error_after", math.nan),
                metrics.get("improvement_mean_percent", math.nan),
                metrics.get("improvement_rms_percent", math.nan),
                metrics.get("improvement_max_percent", math.nan),
            ])

    # 2. 中文统计报告 TXT
    stats_txt = BASE_OUTPUT_DIR / "补偿效果统计.txt"
    with stats_txt.open("w", encoding="utf-8-sig") as f:
        f.write("机器学习误差补偿统计报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"数据集规模：{n_train + n_test} 个成功解算历元\n")
        f.write(f"训练集数量：{n_train}\n")
        f.write(f"测试集数量：{n_test}\n")
        f.write(f"使用特征：{', '.join(feature_columns)}\n\n")

        for name, metrics, rows in [
            ("线性回归", lr_metrics, lr_rows),
            ("随机森林", rf_metrics, rf_rows),
        ]:
            f.write(f"{name} 补偿效果\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"  补偿前平均误差：{metrics.get('mean_error_before', math.nan):.6f} m\n"
            )
            f.write(
                f"  补偿后平均误差：{metrics.get('mean_error_after', math.nan):.6f} m\n"
            )
            f.write(
                f"  补偿前 RMS 误差：{metrics.get('rms_error_before', math.nan):.6f} m\n"
            )
            f.write(
                f"  补偿后 RMS 误差：{metrics.get('rms_error_after', math.nan):.6f} m\n"
            )
            f.write(
                f"  补偿前最大误差：{metrics.get('max_error_before', math.nan):.6f} m\n"
            )
            f.write(
                f"  补偿后最大误差：{metrics.get('max_error_after', math.nan):.6f} m\n"
            )
            f.write(
                f"  平均误差改善：{metrics.get('improvement_mean_percent', math.nan):.2f}%\n"
            )
            f.write(
                f"  RMS 误差改善：{metrics.get('improvement_rms_percent', math.nan):.2f}%\n"
            )
            f.write(
                f"  最大误差改善：{metrics.get('improvement_max_percent', math.nan):.2f}%\n\n"
            )

        # 对比
        f.write("模型对比与结论\n")
        f.write("-" * 40 + "\n")
        if lr_metrics and rf_metrics:
            if rf_metrics["rms_error_after"] < lr_metrics["rms_error_after"]:
                f.write(
                    "随机森林模型补偿后 RMS 误差更低，效果优于线性回归。\n"
                )
                f.write(
                    "原因：随机森林能够捕捉特征间的非线性关系，"
                    "对复杂误差结构的拟合能力更强。\n"
                )
            else:
                f.write(
                    "线性回归模型补偿后 RMS 误差更低，效果优于随机森林。\n"
                )
                f.write(
                    "原因：当前数据量或特征维度下，线性关系已能较好描述误差规律，\n"
                    "随机森林可能出现过拟合。\n"
                )
        f.write("\n不足与改进方向：\n")
        f.write("1. 当前特征仅包含历元级统计量，可进一步引入卫星几何结构特征；\n")
        f.write("2. 可尝试 XGBoost、神经网络等更复杂的模型；\n")
        f.write("3. 增加更多不同时间、不同地理位置的场景，提升模型泛化能力；\n")
        f.write("4. 可引入动态窗口或时序模型（如 LSTM）利用历元间相关性。\n")

    # 3. 可视化
    _plot_error_curves(lr_rows, "linear_regression")
    _plot_error_curves(rf_rows, "random_forest")
    _plot_model_comparison_bar(lr_metrics, rf_metrics)
    _plot_predicted_vs_true(lr_rows, rf_rows)

    print("[evaluate_models] 评估与可视化完成。")
    return {
        "summary_csv": summary_csv,
        "stats_txt": stats_txt,
    }


def _plot_error_curves(rows: List[dict], model_name: str) -> None:
    """绘制补偿前后误差曲线。"""
    before = [_safe_float(r["error_before"]) for r in rows]
    after = [_safe_float(r["error_after"]) for r in rows]
    indices = list(range(1, len(before) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(indices, before, marker="o", linewidth=1.2, label="补偿前误差", color="#dc2626")
    ax.plot(indices, after, marker="s", linewidth=1.2, label="补偿后误差", color="#16a34a")
    ax.set_xlabel("测试样本序号")
    ax.set_ylabel("三维定位误差 (m)")
    ax.set_title(f"{model_name}：补偿前后误差对比")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_OUTPUT_DIR / f"{model_name}_误差曲线.png", dpi=160)
    plt.close(fig)


def _plot_model_comparison_bar(lr_metrics: dict, rf_metrics: dict) -> None:
    """绘制两个模型 RMS / Mean / Max 对比柱状图。"""
    categories = ["Mean", "RMS", "Max"]
    x = np.arange(len(categories))
    width = 0.2

    lr_before = [
        lr_metrics.get("mean_error_before", 0),
        lr_metrics.get("rms_error_before", 0),
        lr_metrics.get("max_error_before", 0),
    ]
    lr_after = [
        lr_metrics.get("mean_error_after", 0),
        lr_metrics.get("rms_error_after", 0),
        lr_metrics.get("max_error_after", 0),
    ]
    rf_before = [
        rf_metrics.get("mean_error_before", 0),
        rf_metrics.get("rms_error_before", 0),
        rf_metrics.get("max_error_before", 0),
    ]
    rf_after = [
        rf_metrics.get("mean_error_after", 0),
        rf_metrics.get("rms_error_after", 0),
        rf_metrics.get("max_error_after", 0),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5 * width, lr_before, width, label="线性回归-补偿前", color="#fca5a5")
    ax.bar(x - 0.5 * width, lr_after, width, label="线性回归-补偿后", color="#dc2626")
    ax.bar(x + 0.5 * width, rf_before, width, label="随机森林-补偿前", color="#86efac")
    ax.bar(x + 1.5 * width, rf_after, width, label="随机森林-补偿后", color="#16a34a")

    ax.set_ylabel("误差 (m)")
    ax.set_title("模型补偿效果对比（Mean / RMS / Max）")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURE_OUTPUT_DIR / "模型对比柱状图.png", dpi=160)
    plt.close(fig)


def _plot_predicted_vs_true(lr_rows: List[dict], rf_rows: List[dict]) -> None:
    """绘制预测误差与真实误差散点图。"""
    lr_true = []
    lr_pred = []
    for r in lr_rows:
        ex = _safe_float(r["pred_error_x"])
        ey = _safe_float(r["pred_error_y"])
        ez = _safe_float(r["pred_error_z"])
        tx = _safe_float(r["true_x"]) - _safe_float(r["spp_x"])
        ty = _safe_float(r["true_y"]) - _safe_float(r["spp_y"])
        tz = _safe_float(r["true_z"]) - _safe_float(r["spp_z"])
        lr_pred.append(math.sqrt(ex ** 2 + ey ** 2 + ez ** 2))
        lr_true.append(math.sqrt(tx ** 2 + ty ** 2 + tz ** 2))

    rf_true = []
    rf_pred = []
    for r in rf_rows:
        ex = _safe_float(r["pred_error_x"])
        ey = _safe_float(r["pred_error_y"])
        ez = _safe_float(r["pred_error_z"])
        tx = _safe_float(r["true_x"]) - _safe_float(r["spp_x"])
        ty = _safe_float(r["true_y"]) - _safe_float(r["spp_y"])
        tz = _safe_float(r["true_z"]) - _safe_float(r["spp_z"])
        rf_pred.append(math.sqrt(ex ** 2 + ey ** 2 + ez ** 2))
        rf_true.append(math.sqrt(tx ** 2 + ty ** 2 + tz ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    max_val = max(
        max(lr_true + lr_pred + rf_true + rf_pred, default=1.0), 1.0
    )

    for ax, true, pred, title in [
        (axes[0], lr_true, lr_pred, "线性回归"),
        (axes[1], rf_true, rf_pred, "随机森林"),
    ]:
        ax.scatter(true, pred, s=30, alpha=0.6, edgecolors="none")
        ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.2, label="y=x")
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(0, max_val * 1.05)
        ax.set_xlabel("真实误差 3D (m)")
        ax.set_ylabel("预测误差 3D (m)")
        ax.set_title(f"{title}：预测误差 vs 真实误差")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURE_OUTPUT_DIR / "预测与真实误差对比.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    print("请使用 run_enhance.py 运行完整流程。")
