"""
compensate.py

加载训练好的模型，对测试集进行误差预测，
计算补偿后坐标，并输出预测结果 CSV。
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from enhance.enhance_config import (
    BASE_OUTPUT_DIR,
    PREDICTION_OUTPUT_DIR,
)


def compensate_and_save(
    model_path: Path,
    model_name: str,
    X_test: np.ndarray,
    test_metadata: List[dict],
) -> Path:
    """加载模型，预测误差，补偿坐标，保存预测结果 CSV。

    test_metadata 中每个元素至少包含：
    scenario_name, epoch_time, spp_x, spp_y, spp_z, true_x, true_y, true_z
    """
    print(f"[compensate] 正在使用 {model_name} 进行误差补偿...")
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    PREDICTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PREDICTION_OUTPUT_DIR / f"{model_name}_补偿预测.csv"

    fieldnames = [
        "scenario_name",
        "epoch_time",
        "true_x",
        "true_y",
        "true_z",
        "spp_x",
        "spp_y",
        "spp_z",
        "pred_error_x",
        "pred_error_y",
        "pred_error_z",
        "compensated_x",
        "compensated_y",
        "compensated_z",
        "error_before",
        "error_after",
        "improvement_m",
        "improvement_percent",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, meta in enumerate(test_metadata):
            spp_x = float(meta["spp_x"])
            spp_y = float(meta["spp_y"])
            spp_z = float(meta["spp_z"])
            true_x = float(meta["true_x"])
            true_y = float(meta["true_y"])
            true_z = float(meta["true_z"])

            pred_ex = float(y_pred[i, 0])
            pred_ey = float(y_pred[i, 1])
            pred_ez = float(y_pred[i, 2])

            compensated_x = spp_x + pred_ex
            compensated_y = spp_y + pred_ey
            compensated_z = spp_z + pred_ez

            error_before = math.sqrt(
                (spp_x - true_x) ** 2
                + (spp_y - true_y) ** 2
                + (spp_z - true_z) ** 2
            )
            error_after = math.sqrt(
                (compensated_x - true_x) ** 2
                + (compensated_y - true_y) ** 2
                + (compensated_z - true_z) ** 2
            )
            improvement_m = error_before - error_after
            improvement_percent = (
                (improvement_m / error_before * 100.0) if error_before > 1e-9 else 0.0
            )

            writer.writerow(
                {
                    "scenario_name": meta.get("scenario_name", ""),
                    "epoch_time": meta.get("epoch_time", ""),
                    "true_x": true_x,
                    "true_y": true_y,
                    "true_z": true_z,
                    "spp_x": spp_x,
                    "spp_y": spp_y,
                    "spp_z": spp_z,
                    "pred_error_x": pred_ex,
                    "pred_error_y": pred_ey,
                    "pred_error_z": pred_ez,
                    "compensated_x": compensated_x,
                    "compensated_y": compensated_y,
                    "compensated_z": compensated_z,
                    "error_before": error_before,
                    "error_after": error_after,
                    "improvement_m": improvement_m,
                    "improvement_percent": improvement_percent,
                }
            )

    print(f"[compensate] {model_name} 预测结果已保存：{csv_path}")
    return csv_path


def run_compensation(train_result: Dict[str, any]) -> Dict[str, Path]:
    """对两个模型分别执行补偿并返回输出路径。"""
    X_test = train_result["X_test"]
    test_metadata = train_result["test_metadata"]

    lr_path = compensate_and_save(
        train_result["lr_path"],
        "线性回归",
        X_test,
        test_metadata,
    )
    rf_path = compensate_and_save(
        train_result["rf_path"],
        "随机森林",
        X_test,
        test_metadata,
    )

    return {"线性回归": lr_path, "随机森林": rf_path}


if __name__ == "__main__":
    print("请使用 run_enhance.py 运行完整流程。")
