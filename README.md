# 北斗 SPP 全流程定位解算系统

本项目实现一个纯 Python 的北斗 RINEX NAV 定位解算流程：

1. 解析 RINEX 导航文件，并生成模拟伪距观测值。
2. 根据广播星历计算卫星 ECEF 坐标和卫星钟差。
3. 使用卫星钟差、简化电离层模型和 Saastamoinen 对流层模型修正伪距。
4. 使用迭代最小二乘完成伪距单点定位。
5. 完成连续定位、精度分析、可视化、图形界面和测试。

核心 GNSS 算法均为手写实现，不调用 RTKLIB、gnsspy、georinex 等第三方 GNSS 定位库。

## 环境依赖

```powershell
pip install numpy matplotlib PyQt5
```

## 命令行运行

```powershell
python basic/module5.py
```

默认使用 `nav/tarc0910.26b_cnav` 作为导航文件。可在 `basic/module5.py` 中修改 `NAV_FILE_PATH` 等参数。

## 图形界面

```powershell
python basic/rinex_gui.py
```

图形界面支持导入 NAV 文件、设置解算参数和接收机坐标、运行定位、查看结果表格与图像，并导出生成的结果文件。

GUI 默认打开 `nav/` 目录，支持选择 `*.26b_cnav`、`*.cnav` 等格式的 RINEX NAV 文件。

## 主要输出文件

模块一：
- `outputs/basic/module1_ephemeris_list.csv`
- `outputs/basic/module1_nav_parse_summary.txt`

模块二：
- `outputs/basic/module2_satellite_position_clock.csv`
- `outputs/basic/module2_satellite_position_summary.txt`
- `outputs/basic/module2_pseudorange_correction_debug.csv`（调试文件，展示 rho、卫星钟差、各项误差与模拟伪距之间的关系，不作为模块三 SPP 解算输入）

模块三：
- `outputs/basic/module3_pseudorange_single_epoch.csv`
- `outputs/basic/module3_spp_result_single_epoch.txt`

模块四：
- `outputs/basic/module4_continuous_position_results.csv`
- `outputs/basic/module4_error_statistics.txt`
- `outputs/basic/module4_error_curve.png`
- `outputs/basic/module4_trajectory.png`
- `outputs/basic/module4_satellite_dop_curve.png`

模块五：
- `outputs/basic/module5_system_test_report.txt`

## 提高部分

项目提供独立的机器学习误差补偿模块，位于 `enhance/` 目录下。

### 运行命令

```powershell
python enhance/run_enhance.py
```

### 额外依赖

```powershell
pip install scikit-learn joblib
```

（若代码中使用 pandas，也需安装：`pip install pandas`）

### 功能说明

提高部分会同时训练两个模型：
- **LinearRegression**（线性回归）
- **RandomForestRegressor**（随机森林）

对基础模块的 SPP 定位结果进行误差预测与坐标补偿，并输出补偿前后效果对比。

### 输出目录

提高部分所有输出统一放在 `outputs/enhance/`：
- `ml_dataset.csv` — 机器学习数据集
- `models/linear_regression_model.joblib` — 线性回归模型
- `models/random_forest_model.joblib` — 随机森林模型
- `predictions/linear_regression_predictions.csv` — 线性回归补偿预测结果
- `predictions/random_forest_predictions.csv` — 随机森林补偿预测结果
- `model_comparison_summary.csv` — 模型对比指标
- `ml_compensation_statistics.txt` — 补偿统计报告
- `figures/error_curve_linear_regression.png` — 误差曲线
- `figures/error_curve_random_forest.png` — 误差曲线
- `figures/model_comparison_bar.png` — 模型对比柱状图
- `figures/predicted_vs_true_error.png` — 预测误差散点图
- `ml_technical_report.txt` — 技术报告

> 注意：`basic/` 为基础部分，`enhance/` 为提高部分，两者输出目录相互独立。

## 导航文件

RINEX NAV 导航文件统一放在项目根目录下的 `nav/` 文件夹中：
- `nav/tarc0910.26b_cnav`（默认文件）

文件格式为 RINEX 3.x 北斗导航电文（CNAV），解析器同时兼容 `.26b` 等历史扩展名。
