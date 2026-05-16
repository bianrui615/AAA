"""
compensate.py

消费 train_models.train_models() 返回的 OOF (out-of-fold) 预测，
计算补偿后坐标并写入预测结果 CSV。

注意：LOSO-CV 流程下，预测已经在交叉验证过程中产生（每个样本在自己
作为测试 fold 时被预测一次），因此这里不再加载模型再次 predict。
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from enhance.enhance_config import PREDICTION_OUTPUT_DIR


PREDICTION_FIELDNAMES = [
    "scenario_name",
    "epoch_time",
    "fold_index",
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


def _safe_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def write_oof_csv(model_name: str, oof_rows: List[dict]) -> Path:
    """将单个模型的 OOF 预测写成 补偿预测 CSV。"""
    PREDICTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PREDICTION_OUTPUT_DIR / f"{model_name}_补偿预测.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDNAMES)
        writer.writeheader()
        for rec in oof_rows:
            spp_x = _safe_float(rec.get("spp_x"))
            spp_y = _safe_float(rec.get("spp_y"))
            spp_z = _safe_float(rec.get("spp_z"))
            true_x = _safe_float(rec.get("true_x"))
            true_y = _safe_float(rec.get("true_y"))
            true_z = _safe_float(rec.get("true_z"))

            pred_ex = _safe_float(rec.get("pred_error_x"))
            pred_ey = _safe_float(rec.get("pred_error_y"))
            pred_ez = _safe_float(rec.get("pred_error_z"))

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
                    "scenario_name": rec.get("scenario_name", ""),
                    "epoch_time": rec.get("epoch_time", ""),
                    "fold_index": rec.get("fold_index", ""),
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
    print(f"[compensate] {model_name} OOF 预测结果已保存：{csv_path}")
    return csv_path


def run_compensation(train_result: Dict[str, Any]) -> Dict[str, Path]:
    """对所有模型的 OOF 预测写 CSV 并返回路径映射。"""
    oof_predictions = train_result.get("oof_predictions", {})
    if not oof_predictions:
        raise RuntimeError("train_result 缺少 oof_predictions 字段，无法生成补偿 CSV。")

    out: Dict[str, Path] = {}
    for model_name, rows in oof_predictions.items():
        out[model_name] = write_oof_csv(model_name, rows)
    return out


if __name__ == "__main__":
    print("请使用 run_enhance.py 运行完整流程。")
