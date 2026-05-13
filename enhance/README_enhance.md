# 北斗 SPP 定位解算系统 — 提高部分

## 简介

提高部分在基础部分（`basic/`）之上，利用机器学习方法对单点定位（SPP）结果进行误差建模与补偿，验证数据驱动方法在 GNSS 定位精度提升中的可行性。

## 额外依赖

```powershell
pip install scikit-learn joblib pandas matplotlib numpy
```

> 说明：`numpy` 与 `matplotlib` 为基础部分已需要的依赖，`scikit-learn`、`joblib` 为提高部分新增依赖。

## 文件结构

```
enhance/
  __init__.py
  enhance_config.py       # 场景配置、特征列、标签列、输出路径
  dataset_builder.py      # 构建机器学习数据集（运行场景 + 收集特征与标签）
  train_models.py         # 训练 LinearRegression 与 RandomForestRegressor
  compensate.py           # 模型预测误差并补偿坐标
  evaluate_models.py      # 评估对比、统计报告、可视化图表
  run_enhance.py          # 统一运行入口
  README_enhance.md       # 本文件
```

## 运行方式

```powershell
python enhance/run_enhance.py
```

该命令会自动完成：
1. 创建 `outputs/enhance/` 目录结构；
2. 构建至少 3 个不同场景并生成连续定位结果；
3. 构建 `ml/ml_dataset.csv`；
4. 划分训练集（70%）与测试集（30%）；
5. 训练 `LinearRegression`；
6. 训练 `RandomForestRegressor`；
7. 分别进行误差预测和坐标补偿；
8. 输出两个模型的预测结果 CSV；
9. 输出模型对比统计与可视化图表；
10. 输出技术报告。

## 输出文件

所有输出统一放在 `outputs/enhance/` 下：

| 文件/目录 | 说明 |
|-----------|------|
| `ml/scenarios/scenario_1/...` | 场景 1 连续定位结果 |
| `ml/scenarios/scenario_2/...` | 场景 2 连续定位结果 |
| `ml/scenarios/scenario_3/...` | 场景 3 连续定位结果 |
| `ml/ml_dataset.csv` | 机器学习数据集（特征 + 标签） |
| `models/linear_regression_model.joblib` | 线性回归模型文件 |
| `models/random_forest_model.joblib` | 随机森林模型文件 |
| `ml/train_test_split_summary.txt` | 训练/测试集划分摘要 |
| `ml/predictions/linear_regression_predictions.csv` | 线性回归补偿预测结果 |
| `ml/predictions/random_forest_predictions.csv` | 随机森林补偿预测结果 |
| `ml/model_comparison_summary.csv` | 模型对比指标 CSV |
| `ml/ml_compensation_statistics.txt` | 中文补偿统计报告 |
| `figures/error_curve_linear_regression.png` | 线性回归补偿前后误差曲线 |
| `figures/error_curve_random_forest.png` | 随机森林补偿前后误差曲线 |
| `figures/model_comparison_bar.png` | 模型对比柱状图 |
| `figures/predicted_vs_true_error.png` | 预测误差 vs 真实误差散点图 |
| `ml/ml_technical_report.txt` | 技术报告 |

## 设计要点

- **不修改 `basic/` 核心逻辑**：提高部分仅调用 `basic/` 中的公开函数，不破坏基础流程。
- **不使用 `.obs` 文件**：所有数据仍基于 BDS-3 CNAV 文件和伪距模拟模型生成。
- **多场景设计**：通过改变仿真时间、接收机坐标、随机种子和高度角阈值构造不同场景，提升模型泛化能力验证。
- **三维误差标签**：预测 `error_x`、`error_y`、`error_z`，可直接对定位坐标进行补偿。
- **可复现性**：所有随机过程均指定 `random_state` 或 `seed`。
