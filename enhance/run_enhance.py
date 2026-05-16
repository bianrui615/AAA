"""
run_enhance.py

提高部分统一运行入口。

执行命令：
    python enhance/run_enhance.py

自动完成：
1. 创建 outputs/enhance/ 目录结构；
2. 构建至少 3 个场景并生成连续定位数据；
3. 构建 机器学习数据集.csv；
4. LOSO 交叉验证训练 LinearRegression / RandomForestRegressor；
5. 写出 OOF 补偿预测 CSV；
6. 评估对比 + 可视化（含 Baseline 与 LOSO 误差棒）；
7. 输出技术报告；
8. 输出训练流程诊断报告；
9. 终端打印关键指标。
"""

from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from enhance.compensate import run_compensation
from enhance.dataset_builder import build_dataset
from enhance.enhance_config import BASE_OUTPUT_DIR, FEATURE_COLUMNS, FIGURE_OUTPUT_DIR, SCENARIOS
from enhance.evaluate_models import evaluate_and_visualize
from enhance.train_models import train_models


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and math.isfinite(value) else "NaN"


def write_technical_report(
    n_total: int,
    n_folds: int,
    cv_summary_metrics: Dict[str, Dict[str, Dict[str, float]]],
    baseline_metrics: Dict[str, float],
    output_path: Path,
) -> None:
    """生成技术报告 技术报告.txt（LOSO 版）。"""
    scenario_descriptions = {
        "scenario1": "静态接收机",
        "scenario2": "动态直线运动",
        "scenario3": "动态折线轨迹",
    }
    lr_summary = cv_summary_metrics.get("线性回归", {})
    rf_summary = cv_summary_metrics.get("随机森林", {})
    baseline_rms = baseline_metrics.get("rmse_3d", math.nan)
    lr_rms_after = lr_summary.get("rmse_3d_after", {}).get("mean", math.nan)
    lr_rms_after_std = lr_summary.get("rmse_3d_after", {}).get("std", math.nan)
    rf_rms_after = rf_summary.get("rmse_3d_after", {}).get("mean", math.nan)
    rf_rms_after_std = rf_summary.get("rmse_3d_after", {}).get("std", math.nan)
    lr_imp = lr_summary.get("improvement_percent", {}).get("mean", math.nan)
    lr_imp_std = lr_summary.get("improvement_percent", {}).get("std", math.nan)
    rf_imp = rf_summary.get("improvement_percent", {}).get("mean", math.nan)
    rf_imp_std = rf_summary.get("improvement_percent", {}).get("std", math.nan)

    with output_path.open("w", encoding="utf-8-sig") as f:
        f.write("北斗 SPP 定位解算系统 — 提高部分技术报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. 提高部分目标\n")
        f.write("-" * 40 + "\n")
        f.write(
            "利用机器学习方法（线性回归与随机森林）对基础模块的 SPP "
            "定位结果进行误差建模与补偿，\n"
            "并通过 LOSO 交叉验证科学评估模型在新场景上的泛化能力。\n\n"
        )

        f.write("2. 数据来源说明\n")
        f.write("-" * 40 + "\n")
        f.write("数据优先来源于 basic/gui_scenario_runner.py 运行的三场景结果。\n")
        f.write("使用 enhance/enhance_config.py 中定义的 3 个 BDS-3 CNAV 场景：\n")
        for index, scenario in enumerate(SCENARIOS, start=1):
            description = scenario_descriptions.get(scenario.name, "连续定位场景")
            f.write(
                f"  场景 {index}：{scenario.nav_file_path}（{description}，"
                f"{scenario.start_time:%Y-%m-%d} 至 {scenario.end_time:%Y-%m-%d}，"
                f"输出名 {scenario.name}）\n"
            )
        f.write("各场景基于广播星历和模拟伪距误差模型产生定位结果。\n\n")

        f.write("3. 为什么不用 .obs 文件\n")
        f.write("-" * 40 + "\n")
        f.write(
            "本项目课程要求不读取真实观测文件，所有伪距由模拟误差模型生成，\n"
            "保证实验可复现且不受真实观测数据缺失的影响。\n\n"
        )

        f.write("4. 伪距模拟与基础定位流程说明\n")
        f.write("-" * 40 + "\n")
        f.write(
            "每个历元：解析星历 → 计算卫星 ECEF 坐标 → 生成模拟伪距\n"
            "（含 SISRE、电离层、对流层、接收机钟差、噪声）→ 迭代最小二乘 SPP 解算。\n\n"
        )

        f.write("5. 特征工程说明\n")
        f.write("-" * 40 + "\n")
        f.write(f"共使用 {len(FEATURE_COLUMNS)} 个数值特征，包括：\n")
        f.write(f"{', '.join(FEATURE_COLUMNS)}\n")
        f.write(
            "这些特征涵盖了卫星数量、几何精度因子（PDOP/GDOP）、\n"
            "伪距统计量、高度角信息以及解算过程中的可观测参数。\n"
            "为避免数据泄漏，本文未使用伪距模拟误差的真值（如 sisre_error、\n"
            "iono_error、tropo_error 等）作为输入特征，只使用接收端可观测\n"
            "或解算过程中可获得的特征进行误差预测。\n\n"
        )

        f.write("6. 标签定义\n")
        f.write("-" * 40 + "\n")
        f.write(
            "error_x = true_x - spp_x\n"
            "error_y = true_y - spp_y\n"
            "error_z = true_z - spp_z\n"
            "即真实接收机坐标与 SPP 解算坐标在各轴上的偏差。\n\n"
        )

        f.write("7. 线性回归模型原理简述\n")
        f.write("-" * 40 + "\n")
        f.write(
            "线性回归假设特征与三维误差之间存在线性关系，\n"
            "通过最小二乘法拟合系数矩阵，使预测残差平方和最小。\n"
            "本实现内嵌 StandardScaler，使各特征对系数的影响在数值上可比。\n\n"
        )

        f.write("8. 随机森林模型原理简述\n")
        f.write("-" * 40 + "\n")
        f.write(
            "随机森林是由多棵决策树组成的集成模型，\n"
            "通过 Bagging 与特征随机选择降低过拟合风险。\n"
            "本实现已收紧默认超参（max_depth=6, min_samples_leaf=5, "
            "max_features='sqrt'），以缓解小样本下的过拟合。\n\n"
        )

        f.write("9. 训练评估方案（LOSO 交叉验证）\n")
        f.write("-" * 40 + "\n")
        f.write(f"总样本数：{n_total}\n")
        f.write(f"场景数（LOSO 折数）：{n_folds}\n")
        f.write(
            "划分方式：LeaveOneGroupOut，按 scenario_name 分组。每个场景轮流\n"
            "作为测试集，其余场景做训练集，共训练并评估 {n} 次。\n"
            "Pipeline (StandardScaler + 模型) 保证 scaler 只用训练 fold 拟合，\n"
            "杜绝测试统计量泄漏到训练流程。最终发布模型用全部 {n_total} 样本\n"
            "重新拟合，仅用于推理。\n".format(n=n_folds, n_total=n_total)
        )
        f.write("除模型评估外，额外报告 Baseline (不补偿，零模型) 作为对照基线。\n\n")

        f.write("10. 补偿公式\n")
        f.write("-" * 40 + "\n")
        f.write(
            "compensated_x = spp_x + pred_error_x\n"
            "compensated_y = spp_y + pred_error_y\n"
            "compensated_z = spp_z + pred_error_z\n\n"
        )

        f.write("11. LOSO 评估结果\n")
        f.write("-" * 40 + "\n")
        f.write(f"Baseline (不补偿)        3D RMS = {_fmt(baseline_rms)} m\n")
        f.write(
            f"线性回归 (LOSO mean±std) 3D RMS = {_fmt(lr_rms_after)} ± "
            f"{_fmt(lr_rms_after_std)} m，改善 {_fmt(lr_imp, 2)}% ± {_fmt(lr_imp_std, 2)}%\n"
        )
        f.write(
            f"随机森林 (LOSO mean±std) 3D RMS = {_fmt(rf_rms_after)} ± "
            f"{_fmt(rf_rms_after_std)} m，改善 {_fmt(rf_imp, 2)}% ± {_fmt(rf_imp_std, 2)}%\n\n"
        )

        f.write("12. 两个模型结果对比\n")
        f.write("-" * 40 + "\n")
        if math.isfinite(lr_rms_after) and math.isfinite(rf_rms_after):
            if rf_rms_after < lr_rms_after:
                f.write("随机森林 LOSO 平均 RMS 较低。")
            else:
                f.write("线性回归 LOSO 平均 RMS 较低。")
            f.write(
                f"  但 N={n_folds} 折下 std 估计极不稳定，差异是否显著需更多场景"
                "方能确认。\n"
            )
        lr_beats = math.isfinite(lr_rms_after) and lr_rms_after < baseline_rms
        rf_beats = math.isfinite(rf_rms_after) and rf_rms_after < baseline_rms
        if not lr_beats and not rf_beats:
            f.write(
                "\n⚠ 两个模型 LOSO 平均 RMS 均高于 Baseline，说明在当前特征/数据规模下，\n"
                "模型未能从 SPP 残差中学到有效误差信号。详见 训练流程评估报告.md。\n"
            )
        f.write(
            "\n详细每折指标见 outputs/enhance/ml/CV每折指标.csv；\n"
            "汇总指标见 outputs/enhance/ml/CV汇总指标.csv / CV汇总指标.txt。\n\n"
        )

        f.write("13. 优势、不足和改进方向\n")
        f.write("-" * 40 + "\n")
        f.write(
            "优势：\n"
            "  - LOSO 评估直接度量模型对新场景的泛化能力，符合工程实际；\n"
            "  - Pipeline + Baseline 对比避免数据泄漏与虚假提升。\n\n"
            "不足：\n"
            "  - 仅 3 个场景，LOSO std 估计不稳定，模型对比缺乏统计显著性；\n"
            "  - 当前特征均为历元级聚合，未利用卫星级（每卫星仰角/残差等）细节；\n"
            "  - 数据为模拟生成，真实环境的多路径/遮挡影响未体现。\n\n"
            "改进方向：\n"
            "  - 扩展到 ≥ 8 个场景（不同 NAV / 接收机位置 / 时间段）；\n"
            "  - 加入卫星级特征 + 残差序列特征；\n"
            "  - 尝试 Ridge / XGBoost / LightGBM；\n"
            "  - 在嵌套 CV 内做超参调优避免泄漏。\n\n"
        )
        f.write("14. 当前版本局限性说明\n")
        f.write("-" * 40 + "\n")
        f.write(
            "当前提高部分完全基于模拟伪距（BDS-3 CNAV 广播星历 + 伪距误差模型），\n"
            "所有定位数据均为仿真生成，未使用真实 RINEX OBS 观测文件。\n"
            "主要局限：\n"
            "  - 模拟误差模型与真实观测环境存在差异；\n"
            "  - 多路径效应、信号遮挡等复杂环境干扰未在当前模型中体现。\n"
            "后续扩展：\n"
            "  - module4.py 已通过 pseudorange_source 参数预留 OBS 接口；\n"
            "  - 接入真实 OBS 后可重新训练模型，验证泛化能力。\n"
        )


def write_diagnosis_report(output_path: Path) -> None:
    """写入 训练流程评估报告.md：对 8 个问题的诊断。"""
    content = """# enhance/train_models.py 训练流程评估报告

> 本文件由 `run_enhance.py` 在每次运行后自动重新生成。
> 评估对象：本仓库 `enhance/train_models.py` 当前实现及其与 3 场景数据的匹配情况。

## Q1. LinearRegression 和 RandomForestRegressor 是如何训练的？

- **LinearRegression**：使用 sklearn `LinearRegression()` 默认参数（最小二乘 OLS），无正则化；
  通过 `Pipeline([StandardScaler(), LinearRegression()])` 同时做标准化。14 维特征 →
  3 维误差标签（error_x/y/z），sklearn 对每个输出独立拟合一组系数。
- **RandomForestRegressor**：当前实现使用 `n_estimators=200, max_depth=6, min_samples_leaf=5,
  max_features='sqrt', random_state=2026, n_jobs=-1`（在改造前为 max_depth=10、
  min_samples_leaf=1，对 146 训练样本严重过拟合）。原生支持多输出。
- **可选 GridSearchCV**：函数签名保留 `enable_grid_search` 参数但默认关闭；LOSO 流程下
  超参调优应放进嵌套 CV，避免泄漏，未自动启用。

## Q2. 训练集/测试集是如何划分的？

- 改造前（已废弃）：`_scenario_based_split` 单次按 `test_ratio=0.3` 切场景，
  N=3 场景下退化为「2 场景训练 / 1 场景测试」，单次划分，random_state=2026。
- **改造后**：使用 `sklearn.model_selection.LeaveOneGroupOut`，groups = `scenario_name`，
  共 3 折，每个场景轮流做测试，OOF（out-of-fold）预测覆盖全部 219 样本。
- 最终发布模型在 CV 结束后用全部数据 fit 一次，仅用于推理。

## Q3. 按 scenario_name 整场景划分的优点与缺点

**优点**
1. 杜绝时序数据泄漏：同场景 300s 间隔的相邻历元在 PDOP / 卫星几何 / 伪距统计上
   高度相关，随机切分会让测试样本被训练近邻"插值"得到虚高 R²。
2. 评估更贴近部署：模型上线后面对的就是新场景。
3. 符合 GNSS 时序数据的最佳实践（同一时段不能跨训练/测试）。

**缺点**
1. 训练样本被强制丢弃整整一个场景，对小数据集而言信息利用率低。
2. **评估方差极大**（见 Q4）。
3. 不适合做模型选择/超参调优（单次评估方差大于真实模型差异）。
4. 各场景内部存在系统差异（NAV 文件、卫星几何）时，模型可能在见过的 2 场景上
   完美拟合却对第三场景毫无泛化能力——这正是原版 -6.57%/-1.66% 负改善的根因。

## Q4. 只有 3 个场景时拿 1 个场景做测试是否评估不稳定？

**极不稳定，几乎没有统计意义。**

1. **测试集有效样本量远低于名义 73**：场景内 73 历元每 300s 一个，PDOP/卫星几何
   高度时序相关，有效自由度（ESS）估计仅 5–15。
2. **不同测试场景的结果可能差异巨大**：原版固定 `random_state=2026` 选 scenario3 做测试，
   换成 scenario1 或 scenario2 时"改善百分比"可能从 -10% 跳到 +20%。
3. **改造前 -6.57% (LR) 与 -1.66% (RF) 的差距完全可能落在测试场景选择的随机区间内**，
   无法据此判断 RF 是否真的优于 LR。
4. **缺乏统计检验**：无置信区间、无 paired t-test、无 bootstrap，结论不可证伪。
   ⇒ **改造方案**：用 LOSO-CV，3 折轮流测试，报告 mean ± std；并明确指出 N=3 时
   std 估计本身极不稳定。

## Q5. 当前特征数量和样本数量是否匹配？

- 维度：14 特征，3 维标签。训练样本 146（LOSO 平均），测试 73。
- **线性回归视角**：每输出独立拟合 15 参数（14 系数 + 截距），146/15 ≈ 9.7 样本/参数，
  刚过经验下限（10），仍偏低。
- **随机森林视角**：原 max_depth=10 + min_samples_leaf=1 + 146 样本 → 树几乎逐样本
  记忆；改造后 max_depth=6 + min_samples_leaf=5 显著降低记忆能力。bagging 仅部分缓解。
- **有效样本量更悲观**：同场景 73 历元强相关，146 训练样本的"独立信息"接近 2 个
  场景的"样本聚簇"，远低于 146。
- **维度灾难**：14 维空间 146 个点极度稀疏（√[14]{146} ≈ 1.4）。
- **结论**：对当前默认超参的 RF 严重不足；对 LR 勉强够用但仍偏少。

## Q6. RandomForestRegressor(n_estimators=200, max_depth=10) 是否可能过拟合？

**原版极可能严重过拟合**，但这并不是 -1.66% 负改善的主因。

1. **过拟合证据**：max_depth=10 + min_samples_leaf=1 + 146 训练样本，单棵树足以
   逐样本记忆；训练 R² 接近 1，OOF/测试 R² 低甚至为负。
2. **bagging 不能消除**：200 棵树平均降低方差，但树间偏置/记忆模式相似。
3. **主因其实是分布偏移**：测试场景的 NAV 文件、卫星几何与训练场景不同，模型完美
   拟合训练分布也无能为力。RF 在分布外只能做近邻外推，预测被夹紧到训练集见过的
   误差范围内。
4. **改造方案**：max_depth=4–6, min_samples_leaf=5–10, max_features='sqrt'，并由 LOSO
   CV 同时报告训练 RMS 与验证 RMS（CV每折指标.csv 的 `train_rmse_3d` 列），过拟合
   程度可以一眼看出来。

## Q7. 是否需要标准化、交叉验证、留一场景验证或更多场景？

| 措施 | 必要性 | 处置 |
|---|---|---|
| **StandardScaler** | LinearRegression 必须，RF 不需要 | ✅ 已加入 Pipeline |
| **K-Fold CV** | 必要 | ✅ 已替换单次划分 |
| **GroupKFold (按 scenario)** | 必要 | ✅ 改造为 LeaveOneGroupOut |
| **LOSO（Leave-One-Scenario-Out）** | 当前数据强烈推荐 | ✅ N=3 时 LOSO = GroupKFold(3) |
| **更多场景** | 极强烈推荐 | ⚠ 本方案未实施，但报告中明确建议 ≥ 8 个 |
| **Baseline (不补偿) 对比** | 必要 | ✅ 已加入：模型必须明显优于零模型才算成功 |

## Q8. 更合理的训练评估方案（本仓库已落实）

1. **LOSO 交叉验证替代单次划分**：每个场景轮流测试，OOF 覆盖全样本。
2. **预处理放进 sklearn Pipeline**：StandardScaler 只用训练 fold 拟合，避免泄漏；
   切换到 Ridge / XGBoost 时零成本复用。
3. **多模型 + 多指标 + Baseline**：报告 RMS_after、改善百分比、MAE、R²、训练-验证
   差距，并与 Baseline (不补偿) 做硬对比。
4. **评估与发布解耦**：CV 用于估计泛化误差，最终模型用全量数据 fit 仅用于推理。
5. **明确统计局限**：N=3 折时显式提示 std 估计不稳定，避免误判 RF/LR 优劣。
6. **后续工作**：扩展场景数（≥ 8）；加入卫星级特征；嵌套 CV 内做超参调优；尝试 Ridge /
   XGBoost；接入真实 OBS 验证。

---

## 输出文件位置（每次运行后刷新）

- `outputs/enhance/ml/LOSO_CV_划分汇总.txt`
- `outputs/enhance/ml/CV每折指标.csv`
- `outputs/enhance/ml/CV汇总指标.csv` / `CV汇总指标.txt`
- `outputs/enhance/ml/补偿效果统计.txt`
- `outputs/enhance/ml/模型对比汇总.csv`
- `outputs/enhance/ml/predictions/线性回归_补偿预测.csv`（OOF，219 行）
- `outputs/enhance/ml/predictions/随机森林_补偿预测.csv`（OOF，219 行）
- `outputs/enhance/models/线性回归模型.joblib`（全量数据训练）
- `outputs/enhance/models/随机森林模型.joblib`（全量数据训练）
- `outputs/enhance/figures/*.png`
"""
    output_path.write_text(content, encoding="utf-8-sig")


def main() -> int:
    print("=" * 60)
    print("北斗 SPP 定位解算系统 — 提高部分（LOSO-CV）")
    print("=" * 60)
    start_time = datetime.now()

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[run_enhance] 输出目录：{BASE_OUTPUT_DIR.resolve()}")

    try:
        dataset_path = build_dataset()
    except Exception as exc:
        print(f"[run_enhance] 数据集构建失败：{exc}")
        return 1

    try:
        train_result = train_models(dataset_path)
    except Exception as exc:
        print(f"[run_enhance] 模型训练失败：{exc}")
        return 1

    n_total = train_result["n_total"]
    n_folds = train_result["n_folds"]
    cv_summary_metrics = train_result["cv_summary_metrics"]
    baseline_metrics = train_result["baseline_metrics"]

    try:
        prediction_paths = run_compensation(train_result)
    except Exception as exc:
        print(f"[run_enhance] 误差补偿失败：{exc}")
        return 1

    try:
        eval_paths = evaluate_and_visualize(
            prediction_paths,
            cv_summary_metrics=cv_summary_metrics,
            baseline_metrics=baseline_metrics,
            feature_columns=FEATURE_COLUMNS,
            n_total=n_total,
            n_folds=n_folds,
        )
    except Exception as exc:
        print(f"[run_enhance] 评估可视化失败：{exc}")
        return 1

    report_path = BASE_OUTPUT_DIR / "技术报告.txt"
    write_technical_report(
        n_total=n_total,
        n_folds=n_folds,
        cv_summary_metrics=cv_summary_metrics,
        baseline_metrics=baseline_metrics,
        output_path=report_path,
    )

    diagnosis_path = BASE_OUTPUT_DIR / "训练流程评估报告.md"
    write_diagnosis_report(diagnosis_path)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n[run_enhance] 全部流程完成，耗时 {elapsed:.1f} 秒。")

    print("\n" + "=" * 60)
    print(f"提高部分运行结果汇总（LOSO-CV, n_folds={n_folds}, n_samples={n_total}）")
    print("=" * 60)
    baseline_rms = baseline_metrics["rmse_3d"]
    lr_after = cv_summary_metrics.get("线性回归", {}).get("rmse_3d_after", {})
    rf_after = cv_summary_metrics.get("随机森林", {}).get("rmse_3d_after", {})
    lr_imp = cv_summary_metrics.get("线性回归", {}).get("improvement_percent", {})
    rf_imp = cv_summary_metrics.get("随机森林", {}).get("improvement_percent", {})
    print(f"Baseline (不补偿)        RMS = {baseline_rms:.4f} m")
    print(
        f"线性回归 (LOSO mean±std) RMS = {lr_after.get('mean', math.nan):.4f} ± "
        f"{lr_after.get('std', math.nan):.4f} m  "
        f"改善 {lr_imp.get('mean', math.nan):+.2f}% ± {lr_imp.get('std', math.nan):.2f}%"
    )
    print(
        f"随机森林 (LOSO mean±std) RMS = {rf_after.get('mean', math.nan):.4f} ± "
        f"{rf_after.get('std', math.nan):.4f} m  "
        f"改善 {rf_imp.get('mean', math.nan):+.2f}% ± {rf_imp.get('std', math.nan):.2f}%"
    )
    lr_beats = math.isfinite(lr_after.get("mean", math.nan)) and lr_after["mean"] < baseline_rms
    rf_beats = math.isfinite(rf_after.get("mean", math.nan)) and rf_after["mean"] < baseline_rms
    if lr_beats or rf_beats:
        better = "随机森林" if (rf_beats and (not lr_beats or rf_after["mean"] < lr_after["mean"])) else "线性回归"
        print(f"较优模型（LOSO mean RMS 更低且优于 Baseline）：{better}")
    else:
        print("[WARN] 两个模型 LOSO 平均 RMS 均高于 Baseline，详见 训练流程评估报告.md")

    print("\n输出文件路径：")
    files = [
        dataset_path,
        train_result["lr_path"],
        train_result["rf_path"],
        train_result["summary_path"],
        train_result["fold_metrics_csv"],
        train_result["summary_metrics_csv"],
        train_result["summary_metrics_txt"],
        prediction_paths["线性回归"],
        prediction_paths["随机森林"],
        eval_paths["summary_csv"],
        eval_paths["stats_txt"],
        FIGURE_OUTPUT_DIR / "线性回归_误差曲线.png",
        FIGURE_OUTPUT_DIR / "随机森林_误差曲线.png",
        FIGURE_OUTPUT_DIR / "模型对比柱状图.png",
        FIGURE_OUTPUT_DIR / "预测与真实误差对比.png",
        report_path,
        diagnosis_path,
    ]
    for fp in files:
        print(f"  {fp}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
