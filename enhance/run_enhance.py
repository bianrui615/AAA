"""
run_enhance.py

提高部分统一运行入口。

执行命令：
    python enhance/run_enhance.py

自动完成：
1. 创建 outputs/enhance/ 目录结构；
2. 构建至少 3 个场景并生成连续定位数据；
3. 构建 ml_dataset.csv；
4. 划分训练集/测试集；
5. 训练 LinearRegression 与 RandomForestRegressor；
6. 误差预测与坐标补偿；
7. 输出对比统计与可视化；
8. 输出技术报告；
9. 终端打印关键指标。
"""

from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from enhance.compensate import run_compensation
from enhance.dataset_builder import build_dataset
from enhance.enhance_config import BASE_OUTPUT_DIR, FEATURE_COLUMNS
from enhance.evaluate_models import evaluate_and_visualize
from enhance.train_models import train_models


def write_technical_report(
    n_total: int,
    n_train: int,
    n_test: int,
    lr_rms_before: float,
    lr_rms_after: float,
    rf_rms_before: float,
    rf_rms_after: float,
    output_path: Path,
) -> None:
    """生成技术报告 ml_technical_report.txt。"""
    with output_path.open("w", encoding="utf-8-sig") as f:
        f.write("北斗 SPP 定位解算系统 — 提高部分技术报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. 提高部分目标\n")
        f.write("-" * 40 + "\n")
        f.write(
            "利用机器学习方法（线性回归与随机森林）对基础模块的 SPP "
            "定位结果进行误差建模与补偿，\n"
            "验证数据驱动方法在 GNSS 定位精度提升中的可行性。\n\n"
        )

        f.write("2. 数据来源说明\n")
        f.write("-" * 40 + "\n")
        f.write(
            "数据由 enhance/dataset_builder.py 调用 basic/ 模块生成。\n"
            "使用 nav/tarc0910.26b_cnav 作为 BDS-3 CNAV 导航文件，\n"
            "基于广播星历和伪距模拟模型产生定位结果。\n\n"
        )

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
            "或解算过程中可获得的特征进行误差预测，使模型补偿结果更符合\n"
            "实际工程应用。\n\n"
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
            "优点：可解释性强、训练速度快；\n"
            "缺点：无法捕捉非线性规律。\n\n"
        )

        f.write("8. 随机森林模型原理简述\n")
        f.write("-" * 40 + "\n")
        f.write(
            "随机森林是由多棵决策树组成的集成模型，\n"
            "通过 Bagging 与特征随机选择降低过拟合风险，\n"
            "对高维特征和非线性关系具有较强建模能力。\n\n"
        )

        f.write("9. 训练集/测试集划分\n")
        f.write("-" * 40 + "\n")
        f.write(f"总样本数：{n_total}\n")
        f.write(f"训练集：{n_train}（70%）\n")
        f.write(f"测试集：{n_test}（30%）\n")
        f.write("划分方式：sklearn.train_test_split，random_state=2026\n\n")

        f.write("10. 补偿公式\n")
        f.write("-" * 40 + "\n")
        f.write(
            "compensated_x = spp_x + pred_error_x\n"
            "compensated_y = spp_y + pred_error_y\n"
            "compensated_z = spp_z + pred_error_z\n\n"
        )

        f.write("11. 补偿前后误差对比\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"线性回归：RMS 误差 {lr_rms_before:.6f} m → {lr_rms_after:.6f} m\n"
        )
        f.write(
            f"随机森林：RMS 误差 {rf_rms_before:.6f} m → {rf_rms_after:.6f} m\n\n"
        )

        f.write("12. 两个模型结果对比\n")
        f.write("-" * 40 + "\n")
        if math.isfinite(lr_rms_after) and math.isfinite(rf_rms_after):
            if rf_rms_after < lr_rms_after:
                f.write(
                    "随机森林补偿后 RMS 误差更低，整体效果优于线性回归。\n"
                )
            else:
                f.write(
                    "线性回归补偿后 RMS 误差更低，整体效果优于随机森林。\n"
                )
        f.write(
            "详细统计请参见 outputs/enhance/ml_compensation_statistics.txt\n\n"
        )

        f.write("13. 优势、不足和改进方向\n")
        f.write("-" * 40 + "\n")
        f.write(
            "优势：\n"
            "  - 无需额外硬件或外部差分数据，仅利用已有星历和模拟伪距即可实现；\n"
            "  - 模型训练与推理速度快，易于部署。\n\n"
            "不足：\n"
            "  - 当前数据为模拟生成，与真实观测存在差距；\n"
            "  - 特征工程较简单，未充分利用卫星几何结构细节；\n"
            "  - 场景数量有限，模型泛化能力待验证。\n\n"
            "改进方向：\n"
            "  - 引入真实观测数据验证；\n"
            "  - 尝试 XGBoost、LightGBM、神经网络等更复杂模型；\n"
            "  - 利用时序特征或滑动窗口提升预测稳定性。\n"
        )


def main() -> int:
    print("=" * 60)
    print("北斗 SPP 定位解算系统 — 提高部分")
    print("=" * 60)
    start_time = datetime.now()

    # 1. 创建目录
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[run_enhance] 输出目录：{BASE_OUTPUT_DIR.resolve()}")

    # 2. 构建数据集（同时运行场景并收集数据）
    try:
        dataset_path = build_dataset()
    except Exception as exc:
        print(f"[run_enhance] 数据集构建失败：{exc}")
        return 1

    # 3. 训练模型
    try:
        train_result = train_models(dataset_path)
    except Exception as exc:
        print(f"[run_enhance] 模型训练失败：{exc}")
        return 1

    n_train = train_result["n_train"]
    n_test = train_result["n_test"]
    n_total = n_train + n_test

    # 4. 补偿预测
    try:
        prediction_paths = run_compensation(train_result)
    except Exception as exc:
        print(f"[run_enhance] 误差补偿失败：{exc}")
        return 1

    # 5. 评估与可视化
    try:
        eval_paths = evaluate_and_visualize(
            prediction_paths,
            n_train=n_train,
            n_test=n_test,
            feature_columns=FEATURE_COLUMNS,
        )
    except Exception as exc:
        print(f"[run_enhance] 评估可视化失败：{exc}")
        return 1

    # 6. 读取评估指标用于技术报告和终端打印
    # 从 prediction CSV 重新计算 RMS
    import csv as csv_mod

    def _read_rms(csv_path: Path) -> tuple:
        before, after = [], []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                b = float(row.get("error_before", "nan"))
                a = float(row.get("error_after", "nan"))
                if math.isfinite(b):
                    before.append(b)
                if math.isfinite(a):
                    after.append(a)
        rms_b = math.sqrt(sum(v ** 2 for v in before) / len(before)) if before else math.nan
        rms_a = math.sqrt(sum(v ** 2 for v in after) / len(after)) if after else math.nan
        return rms_b, rms_a

    lr_rms_before, lr_rms_after = _read_rms(prediction_paths["linear_regression"])
    rf_rms_before, rf_rms_after = _read_rms(prediction_paths["random_forest"])

    # 7. 技术报告
    report_path = BASE_OUTPUT_DIR / "ml_technical_report.txt"
    write_technical_report(
        n_total=n_total,
        n_train=n_train,
        n_test=n_test,
        lr_rms_before=lr_rms_before,
        lr_rms_after=lr_rms_after,
        rf_rms_before=rf_rms_before,
        rf_rms_after=rf_rms_after,
        output_path=report_path,
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n[run_enhance] 全部流程完成，耗时 {elapsed:.1f} 秒。")

    # 8. 终端汇总
    print("\n" + "=" * 60)
    print("提高部分运行结果汇总")
    print("=" * 60)
    print(f"数据集样本数：{n_total}")
    print(f"训练集数量：{n_train}")
    print(f"测试集数量：{n_test}")
    print(f"线性回归补偿前 RMS：{lr_rms_before:.6f} m")
    print(f"线性回归补偿后 RMS：{lr_rms_after:.6f} m")
    print(f"随机森林补偿前 RMS：{rf_rms_before:.6f} m")
    print(f"随机森林补偿后 RMS：{rf_rms_after:.6f} m")
    if math.isfinite(lr_rms_after) and math.isfinite(rf_rms_after):
        if rf_rms_after < lr_rms_after:
            print(f"效果更好模型：RandomForestRegressor（RMS 更低）")
        else:
            print(f"效果更好模型：LinearRegression（RMS 更低）")
    print("\n输出文件路径：")
    files = [
        dataset_path,
        train_result["lr_path"],
        train_result["rf_path"],
        train_result["summary_path"],
        prediction_paths["linear_regression"],
        prediction_paths["random_forest"],
        eval_paths["summary_csv"],
        eval_paths["stats_txt"],
        BASE_OUTPUT_DIR / "figures" / "error_curve_linear_regression.png",
        BASE_OUTPUT_DIR / "figures" / "error_curve_random_forest.png",
        BASE_OUTPUT_DIR / "figures" / "model_comparison_bar.png",
        BASE_OUTPUT_DIR / "figures" / "predicted_vs_true_error.png",
        report_path,
    ]
    for fp in files:
        print(f"  {fp}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
