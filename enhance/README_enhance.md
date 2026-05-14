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
  dataset_builder.py      # 构建机器学习数据集（优先读取基础三场景结果，必要时运行场景）
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
2. 优先读取 `outputs/basic/gui_scenario_runner/scenario_1..3/` 的基础三场景结果，必要时重新运行增强场景；
3. 构建 `ml/机器学习数据集.csv`；
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
| `ml/scenarios/scenario1/...` | 场景 1 连续定位结果 |
| `ml/scenarios/scenario2/...` | 场景 2 连续定位结果 |
| `ml/scenarios/scenario3/...` | 场景 3 连续定位结果 |
| `ml/机器学习数据集.csv` | 机器学习数据集（特征 + 标签） |
| `models/线性回归模型.joblib` | 线性回归模型文件 |
| `models/随机森林模型.joblib` | 随机森林模型文件 |
| `ml/训练测试集划分汇总.txt` | 训练/测试集划分摘要 |
| `ml/predictions/线性回归_补偿预测.csv` | 线性回归补偿预测结果 |
| `ml/predictions/随机森林_补偿预测.csv` | 随机森林补偿预测结果 |
| `ml/模型对比汇总.csv` | 模型对比指标 CSV |
| `ml/补偿效果统计.txt` | 中文补偿统计报告 |
| `figures/线性回归_误差曲线.png` | 线性回归补偿前后误差曲线 |
| `figures/随机森林_误差曲线.png` | 随机森林补偿前后误差曲线 |
| `figures/模型对比柱状图.png` | 模型对比柱状图 |
| `figures/预测与真实误差对比.png` | 预测误差 vs 真实误差散点图 |
| `ml/技术报告.txt` | 技术报告 |

## 设计要点

- **不修改 `basic/` 核心逻辑**：提高部分仅调用 `basic/` 中的公开函数，不破坏基础流程。
- **不使用 `.obs` 文件**：所有数据仍基于 BDS-3 CNAV 文件和伪距模拟模型生成。
- **多场景设计**：场景名统一为 `scenario1`、`scenario2`、`scenario3`；基础三场景 GUI 的输出目录仍为 `outputs/basic/gui_scenario_runner/scenario_1..3/`。
- **数据来源优先级**：如果基础三场景 CSV 已存在，增强部分直接读取并补充机器学习特征；如果不存在，则按 `enhance_config.py` 重新解算并保存到 `outputs/enhance/ml/scenarios/scenario1..3/`。
- **三维误差标签**：预测 `error_x`、`error_y`、`error_z`，可直接对定位坐标进行补偿。
- **可复现性**：所有随机过程均指定 `random_state` 或 `seed`。
