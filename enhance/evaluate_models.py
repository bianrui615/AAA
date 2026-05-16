"""
evaluate_models.py

读取 LOSO 补偿预测 CSV，输出补偿效果统计 TXT、模型对比 CSV、
以及按 fold 着色的可视化图表（含 baseline 不补偿基线）。
"""

from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from enhance.enhance_config import BASE_OUTPUT_DIR, FIGURE_OUTPUT_DIR

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/enhance") / "matplotlib_cache"))
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

FOLD_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#a855f7", "#f59e0b", "#0891b2"]


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


def compute_aggregate_metrics(rows: List[dict]) -> Dict[str, float]:
    """对一组（OOF）预测行计算汇总指标。"""
    before = [_safe_float(r["error_before"]) for r in rows]
    after = [_safe_float(r["error_after"]) for r in rows]
    before = [v for v in before if math.isfinite(v)]
    after = [v for v in after if math.isfinite(v)]
    if not before or not after:
        return {}

    return {
        "mean_error_before": float(np.mean(before)),
        "mean_error_after": float(np.mean(after)),
        "rms_error_before": float(np.sqrt(np.mean(np.square(before)))),
        "rms_error_after": float(np.sqrt(np.mean(np.square(after)))),
        "max_error_before": float(np.max(before)),
        "max_error_after": float(np.max(after)),
        "min_error_before": float(np.min(before)),
        "min_error_after": float(np.min(after)),
    }


def _per_fold_metrics(rows: List[dict]) -> List[Dict[str, Any]]:
    """按 fold_index 分组后逐折计算 RMS/Mean/改善百分比。"""
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get("fold_index", "")].append(r)

    result: List[Dict[str, Any]] = []
    for fold_key in sorted(groups.keys(), key=lambda k: int(k) if str(k).lstrip("-").isdigit() else 9999):
        fold_rows = groups[fold_key]
        before = np.array([_safe_float(r["error_before"]) for r in fold_rows])
        after = np.array([_safe_float(r["error_after"]) for r in fold_rows])
        before = before[np.isfinite(before)]
        after = after[np.isfinite(after)]
        if before.size == 0 or after.size == 0:
            continue
        rms_b = float(np.sqrt(np.mean(before ** 2)))
        rms_a = float(np.sqrt(np.mean(after ** 2)))
        improvement = (rms_b - rms_a) / rms_b * 100.0 if rms_b > 1e-9 else 0.0
        result.append(
            {
                "fold_index": fold_key,
                "test_scenario": fold_rows[0].get("scenario_name", ""),
                "n": len(fold_rows),
                "rms_before": rms_b,
                "rms_after": rms_a,
                "mean_before": float(before.mean()),
                "mean_after": float(after.mean()),
                "improvement_percent": improvement,
            }
        )
    return result


def evaluate_and_visualize(
    prediction_paths: Dict[str, Path],
    cv_summary_metrics: Dict[str, Dict[str, Dict[str, float]]],
    baseline_metrics: Dict[str, float],
    feature_columns: List[str],
    n_total: int,
    n_folds: int,
) -> Dict[str, Path]:
    """按 LOSO 形式评估两个模型，输出统计文件与可视化图表。"""
    print("[evaluate_models] 开始评估 LOSO 补偿效果...")
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lr_rows = read_prediction_csv(prediction_paths["线性回归"])
    rf_rows = read_prediction_csv(prediction_paths["随机森林"])

    lr_agg = compute_aggregate_metrics(lr_rows)
    rf_agg = compute_aggregate_metrics(rf_rows)
    lr_per_fold = _per_fold_metrics(lr_rows)
    rf_per_fold = _per_fold_metrics(rf_rows)

    # 1. 模型对比 summary CSV
    summary_csv = BASE_OUTPUT_DIR / "模型对比汇总.csv"
    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "rms_before_oof",
                "rms_after_oof",
                "mean_before_oof",
                "mean_after_oof",
                "max_before_oof",
                "max_after_oof",
                "rms_after_loso_mean",
                "rms_after_loso_std",
                "improvement_loso_mean_percent",
                "improvement_loso_std_percent",
                "beats_baseline",
            ]
        )
        baseline_rms = baseline_metrics["rmse_3d"]
        for name, agg, cv in [
            ("LinearRegression", lr_agg, cv_summary_metrics.get("线性回归", {})),
            ("RandomForest", rf_agg, cv_summary_metrics.get("随机森林", {})),
        ]:
            rms_after_mean = cv.get("rmse_3d_after", {}).get("mean", math.nan)
            rms_after_std = cv.get("rmse_3d_after", {}).get("std", math.nan)
            imp_mean = cv.get("improvement_percent", {}).get("mean", math.nan)
            imp_std = cv.get("improvement_percent", {}).get("std", math.nan)
            beats = (
                rms_after_mean < baseline_rms if math.isfinite(rms_after_mean) else False
            )
            writer.writerow(
                [
                    name,
                    f"{agg.get('rms_error_before', math.nan):.6f}",
                    f"{agg.get('rms_error_after', math.nan):.6f}",
                    f"{agg.get('mean_error_before', math.nan):.6f}",
                    f"{agg.get('mean_error_after', math.nan):.6f}",
                    f"{agg.get('max_error_before', math.nan):.6f}",
                    f"{agg.get('max_error_after', math.nan):.6f}",
                    f"{rms_after_mean:.6f}",
                    f"{rms_after_std:.6f}",
                    f"{imp_mean:.4f}",
                    f"{imp_std:.4f}",
                    "YES" if beats else "NO",
                ]
            )

    # 2. 中文统计报告 TXT
    stats_txt = BASE_OUTPUT_DIR / "补偿效果统计.txt"
    with stats_txt.open("w", encoding="utf-8-sig") as f:
        f.write("机器学习误差补偿统计报告（LOSO 交叉验证）\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"数据集规模：{n_total} 个成功解算历元\n")
        f.write(f"评估方式：LeaveOneGroupOut，折数 = {n_folds}\n")
        f.write(f"使用特征 ({len(feature_columns)} 维)：{', '.join(feature_columns)}\n\n")

        f.write("Baseline (不补偿，零模型)\n")
        f.write("-" * 50 + "\n")
        f.write(f"  3D RMS  = {baseline_metrics['rmse_3d']:.6f} m\n")
        f.write(f"  3D Mean = {baseline_metrics['mean_3d']:.6f} m\n")
        f.write(f"  3D Max  = {baseline_metrics['max_3d']:.6f} m\n\n")

        for name, agg, per_fold, cv in [
            ("线性回归", lr_agg, lr_per_fold, cv_summary_metrics.get("线性回归", {})),
            ("随机森林", rf_agg, rf_per_fold, cv_summary_metrics.get("随机森林", {})),
        ]:
            f.write(f"{name} 补偿效果\n")
            f.write("-" * 50 + "\n")
            f.write("  [OOF 整体（所有样本合并计算）]\n")
            f.write(
                f"    补偿前 RMS = {agg.get('rms_error_before', math.nan):.6f} m\n"
            )
            f.write(
                f"    补偿后 RMS = {agg.get('rms_error_after', math.nan):.6f} m\n"
            )
            f.write(
                f"    补偿前 Mean = {agg.get('mean_error_before', math.nan):.6f} m\n"
            )
            f.write(
                f"    补偿后 Mean = {agg.get('mean_error_after', math.nan):.6f} m\n"
            )
            f.write(
                f"    补偿前 Max  = {agg.get('max_error_before', math.nan):.6f} m\n"
            )
            f.write(
                f"    补偿后 Max  = {agg.get('max_error_after', math.nan):.6f} m\n"
            )

            rms_after_mean = cv.get("rmse_3d_after", {}).get("mean", math.nan)
            rms_after_std = cv.get("rmse_3d_after", {}).get("std", math.nan)
            imp_mean = cv.get("improvement_percent", {}).get("mean", math.nan)
            imp_std = cv.get("improvement_percent", {}).get("std", math.nan)
            train_rms = cv.get("train_rmse_3d", {}).get("mean", math.nan)
            r2_x = cv.get("r2_x", {}).get("mean", math.nan)
            r2_y = cv.get("r2_y", {}).get("mean", math.nan)
            r2_z = cv.get("r2_z", {}).get("mean", math.nan)
            beats = (
                rms_after_mean < baseline_metrics["rmse_3d"]
                if math.isfinite(rms_after_mean)
                else False
            )

            f.write("  [LOSO mean ± std]\n")
            f.write(
                f"    RMS_after  = {rms_after_mean:.6f} ± {rms_after_std:.6f} m\n"
            )
            f.write(
                f"    改善百分比 = {imp_mean:.2f}% ± {imp_std:.2f}%\n"
            )
            f.write(
                f"    训练 RMS_after = {train_rms:.6f} m  "
                f"(差距 ↔ 过拟合监控)\n"
            )
            f.write(
                f"    R² (x/y/z) = {r2_x:.3f} / {r2_y:.3f} / {r2_z:.3f}\n"
            )
            f.write(f"    是否优于 Baseline：{'是 ✓' if beats else '否 ✗'}\n")

            f.write("  [每折详情]\n")
            for pf in per_fold:
                f.write(
                    f"    fold {pf['fold_index']}  测试={pf['test_scenario']}  "
                    f"n={pf['n']}  RMS {pf['rms_before']:.4f} → "
                    f"{pf['rms_after']:.4f} m  Δ={pf['improvement_percent']:+.2f}%\n"
                )
            f.write("\n")

        f.write("模型对比与结论\n")
        f.write("-" * 50 + "\n")
        baseline_rms = baseline_metrics["rmse_3d"]
        lr_mean = cv_summary_metrics.get("线性回归", {}).get("rmse_3d_after", {}).get("mean", math.nan)
        rf_mean = cv_summary_metrics.get("随机森林", {}).get("rmse_3d_after", {}).get("mean", math.nan)
        if math.isfinite(lr_mean) and math.isfinite(rf_mean):
            if rf_mean < lr_mean:
                f.write(
                    "随机森林 LOSO 平均 RMS 较低，但 N=3 折的 std 估计极不稳定，\n"
                    "差异是否显著需更多场景方能确认。\n"
                )
            else:
                f.write(
                    "线性回归 LOSO 平均 RMS 较低，但 N=3 折的 std 估计极不稳定，\n"
                    "差异是否显著需更多场景方能确认。\n"
                )
        lr_beats = math.isfinite(lr_mean) and lr_mean < baseline_rms
        rf_beats = math.isfinite(rf_mean) and rf_mean < baseline_rms
        if not lr_beats and not rf_beats:
            f.write(
                "\n⚠ 两个模型补偿后 RMS 均高于 Baseline（不补偿），\n"
                "说明当前特征/数据规模不足以从 SPP 残差中学到有效误差信号。\n"
            )
        f.write("\n改进方向：\n")
        f.write("1. 增加场景数（不同 NAV 文件 / 接收机位置 / 时间段）至 ≥ 8 个；\n")
        f.write("2. 引入卫星级特征（每颗卫星的高度角、信噪比、残差）而非历元级聚合；\n")
        f.write("3. 尝试 Ridge / XGBoost / 卫星几何感知模型；\n")
        f.write("4. 评估稳定后再做 GridSearchCV 超参调优（在嵌套 CV 内）。\n")

    # 3. 可视化
    _plot_error_curves_by_fold(lr_rows, "线性回归", baseline_metrics["rmse_3d"])
    _plot_error_curves_by_fold(rf_rows, "随机森林", baseline_metrics["rmse_3d"])
    _plot_model_comparison_bar_with_baseline(
        cv_summary_metrics, baseline_metrics
    )
    _plot_predicted_vs_true_by_fold(lr_rows, rf_rows)

    print("[evaluate_models] 评估与可视化完成。")
    return {
        "summary_csv": summary_csv,
        "stats_txt": stats_txt,
        "线性回归": prediction_paths["线性回归"],
        "随机森林": prediction_paths["随机森林"],
    }


def _plot_error_curves_by_fold(
    rows: List[dict], model_name: str, baseline_rms: float
) -> None:
    """按 fold 着色绘制补偿前后误差曲线。"""
    folds: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        folds[r.get("fold_index", "")].append(r)

    fig, ax = plt.subplots(figsize=(11, 5))
    cursor = 0
    for idx, fold_key in enumerate(
        sorted(folds.keys(), key=lambda k: int(k) if str(k).lstrip("-").isdigit() else 9999)
    ):
        fold_rows = folds[fold_key]
        before = [_safe_float(r["error_before"]) for r in fold_rows]
        after = [_safe_float(r["error_after"]) for r in fold_rows]
        xs = list(range(cursor + 1, cursor + 1 + len(fold_rows)))
        cursor += len(fold_rows)
        color = FOLD_COLORS[idx % len(FOLD_COLORS)]
        scenario_label = fold_rows[0].get("scenario_name", "?")
        ax.plot(
            xs,
            before,
            linestyle="--",
            linewidth=1.0,
            color=color,
            alpha=0.45,
            label=f"fold {fold_key} ({scenario_label}) - 补偿前",
        )
        ax.plot(
            xs,
            after,
            linestyle="-",
            linewidth=1.4,
            color=color,
            label=f"fold {fold_key} ({scenario_label}) - 补偿后",
        )

    ax.axhline(
        baseline_rms,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label=f"Baseline RMS = {baseline_rms:.2f} m",
    )
    ax.set_xlabel("OOF 样本序号（按 fold 顺序）")
    ax.set_ylabel("三维定位误差 (m)")
    ax.set_title(f"{model_name}：LOSO 补偿前后误差曲线（按 fold 着色）")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_OUTPUT_DIR / f"{model_name}_误差曲线.png", dpi=160)
    plt.close(fig)


def _plot_model_comparison_bar_with_baseline(
    cv_summary_metrics: Dict[str, Dict[str, Dict[str, float]]],
    baseline_metrics: Dict[str, float],
) -> None:
    """柱状图：Baseline / LR / RF 的 RMS_after，含 errorbar。"""
    names = ["Baseline\n(不补偿)", "线性回归\nLOSO", "随机森林\nLOSO"]
    means = [
        baseline_metrics["rmse_3d"],
        cv_summary_metrics.get("线性回归", {}).get("rmse_3d_after", {}).get("mean", math.nan),
        cv_summary_metrics.get("随机森林", {}).get("rmse_3d_after", {}).get("mean", math.nan),
    ]
    stds = [
        0.0,
        cv_summary_metrics.get("线性回归", {}).get("rmse_3d_after", {}).get("std", 0.0),
        cv_summary_metrics.get("随机森林", {}).get("rmse_3d_after", {}).get("std", 0.0),
    ]
    colors = ["#6b7280", "#2563eb", "#16a34a"]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=8, color=colors, alpha=0.85, edgecolor="black")
    for i, (m, s) in enumerate(zip(means, stds)):
        if math.isfinite(m):
            ax.text(x[i], m + 0.05 * max(means), f"{m:.3f}\n±{s:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("3D RMS 误差 (m)")
    ax.set_title("模型补偿效果对比（含 Baseline 与 LOSO 误差棒）")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURE_OUTPUT_DIR / "模型对比柱状图.png", dpi=160)
    plt.close(fig)


def _plot_predicted_vs_true_by_fold(lr_rows: List[dict], rf_rows: List[dict]) -> None:
    """按 fold 着色绘制 预测误差 vs 真实误差 散点。"""

    def _extract(rows: List[dict]) -> Dict[str, List[tuple]]:
        out: Dict[str, List[tuple]] = defaultdict(list)
        for r in rows:
            ex = _safe_float(r["pred_error_x"])
            ey = _safe_float(r["pred_error_y"])
            ez = _safe_float(r["pred_error_z"])
            tx = _safe_float(r["true_x"]) - _safe_float(r["spp_x"])
            ty = _safe_float(r["true_y"]) - _safe_float(r["spp_y"])
            tz = _safe_float(r["true_z"]) - _safe_float(r["spp_z"])
            pred3d = math.sqrt(ex ** 2 + ey ** 2 + ez ** 2)
            true3d = math.sqrt(tx ** 2 + ty ** 2 + tz ** 2)
            out[r.get("fold_index", "")].append((true3d, pred3d, r.get("scenario_name", "")))
        return out

    lr_groups = _extract(lr_rows)
    rf_groups = _extract(rf_rows)

    all_vals = []
    for g in (lr_groups, rf_groups):
        for items in g.values():
            for t, p, _ in items:
                all_vals.extend([t, p])
    max_val = max(max(all_vals, default=1.0), 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, groups, title in [
        (axes[0], lr_groups, "线性回归"),
        (axes[1], rf_groups, "随机森林"),
    ]:
        for idx, fold_key in enumerate(
            sorted(groups.keys(), key=lambda k: int(k) if str(k).lstrip("-").isdigit() else 9999)
        ):
            items = groups[fold_key]
            xs = [t for t, _, _ in items]
            ys = [p for _, p, _ in items]
            scenario_label = items[0][2] if items else "?"
            ax.scatter(
                xs,
                ys,
                s=28,
                alpha=0.6,
                color=FOLD_COLORS[idx % len(FOLD_COLORS)],
                edgecolors="none",
                label=f"fold {fold_key} ({scenario_label})",
            )
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=1.0, label="y=x")
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(0, max_val * 1.05)
        ax.set_xlabel("真实误差 3D (m)")
        ax.set_ylabel("预测误差 3D (m)")
        ax.set_title(f"{title}：OOF 预测 vs 真实误差")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_OUTPUT_DIR / "预测与真实误差对比.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    print("请使用 run_enhance.py 运行完整流程。")
