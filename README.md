# 北斗 SPP 全流程定位解算系统

本项目是一个纯 Python 实现的北斗三号（BDS-3）单点定位（SPP）全流程解算系统，包含基础定位流程与机器学习误差补偿两个层次。

## 项目特点

- **输入格式**：默认解析 BDS-3 CNAV 导航电文（如 `.26b_cnav`、`.cnav`），不依赖 `.obs` 观测文件。
- **伪距模拟**：基于广播星历和误差模型生成模拟伪距，不读取真实观测伪距。
- **算法自主**：核心 GNSS 算法（星历解析、卫星位置计算、迭代最小二乘 SPP）均为手写实现，不调用 RTKLIB、georinex 等第三方 GNSS 库。
- **双层架构**：
  - `basic/`：基础定位模块（模块一至五），完成从 NAV 解析到连续定位的全流程；
  - `enhance/`：提高部分，利用线性回归与随机森林对 SPP 定位误差进行建模与补偿。

## 项目结构

```
├── basic/                      # 基础定位模块
│   ├── module1.py              # 模块一：NAV 导航文件解析 + 模拟伪距生成
│   ├── module2.py              # 模块二：卫星位置/钟差计算
│   ├── module3.py              # 模块三：伪距生成与单点定位解算
│   ├── module4.py              # 模块四：连续定位与结果分析
│   ├── module5.py              # 模块五：系统整合与测试（主入口）
│   └── rinex_gui.py            # 图形界面
├── enhance/                    # 提高部分（机器学习误差补偿）
│   ├── enhance_config.py       # 场景与特征配置
│   ├── dataset_builder.py      # 构建机器学习数据集
│   ├── train_models.py         # 模型训练
│   ├── compensate.py           # 误差预测与坐标补偿
│   ├── evaluate_models.py      # 效果评估与可视化
│   ├── run_enhance.py          # 提高部分统一入口
│   └── README_enhance.md       # 提高部分详细说明
├── nav/                        # 导航文件存放目录
│   ├── tarc0910.26b_cnav       # 默认 BDS-3 CNAV 文件
│   └── tarc1210.26b_cnav ... tarc1300.26b_cnav
├── outputs/                    # 所有输出目录
│   ├── basic/                  # 基础部分输出
│   └── enhance/                # 提高部分输出
├── README.md                   # 本文件
└── requirements.txt            # 依赖列表
```

## 环境依赖

### 基础部分

```powershell
pip install numpy matplotlib
```

图形界面（可选）需要额外安装：

```powershell
pip install PyQt5
```

### 提高部分

```powershell
pip install scikit-learn joblib
```

## 快速开始

### 1. 运行基础部分

```powershell
python basic/module5.py
```

基础部分主入口为 `basic/module5.py`，默认使用 `nav/tarc0910.26b_cnav`。运行后会自动执行：

1. 解析 NAV 导航文件（按 BDS-3 CNAV 格式）；
2. 计算卫星位置与钟差；
3. 生成模拟伪距并进行单历元 SPP 解算；
4. 执行连续定位与结果分析；
5. 输出系统测试报告。

可在 `basic/module5.py` 中修改 `NAV_FILE_PATH`、`RECEIVER_TRUE_POSITION`、`SIMULATION_START_TIME` 等参数。

### 2. 运行图形界面（可选）

```powershell
python basic/rinex_gui.py
```

GUI 支持导入 NAV 文件、设置解算参数与接收机坐标、运行定位、查看结果表格与图像，并导出生成的结果文件。

### 3. 运行提高部分

```powershell
python enhance/run_enhance.py
```

提高部分会优先读取 `outputs/basic/gui_scenario_runner/scenario_1..3/` 中的基础三场景结果；若不存在，则调用基础模块按 `enhance/enhance_config.py` 重新生成场景数据。随后训练 `LinearRegression` 与 `RandomForestRegressor` 两个模型，对 SPP 坐标进行误差补偿，并输出补偿效果对比、可视化图表与技术报告。

详细说明见 `enhance/README_enhance.md`。

## 基础部分输出文件

运行 `python basic/module5.py` 或在 `basic/rinex_gui.py` 中选择 NAV 文件后，输出按 NAV 文件名存放在 `outputs/basic/<nav文件名点号改下划线>/`。例如 `nav/tarc0910.26b_cnav` 的输出目录为 `outputs/basic/tarc0910_26b_cnav/`。

**模块一**
- `outputs/basic/<nav目录>/module1_导航电文解析调试.csv` — 星历解析调试数据
- `outputs/basic/<nav目录>/module1_模拟伪距.csv` — 模拟伪距明细
- `outputs/basic/<nav目录>/module1_导航电文解析汇总.txt` — 解析摘要

**模块二**
- `outputs/basic/<nav目录>/module2_卫星位置与钟差.csv` — 卫星位置与钟差
- `outputs/basic/<nav目录>/module2_卫星位置汇总.txt` — 计算摘要
- `outputs/basic/<nav目录>/module2_伪距修正调试.csv` — 伪距修正调试文件（仅用于展示 rho、卫星钟差、各项误差与模拟伪距之间的关系）

**模块三**
- `outputs/basic/<nav目录>/module3_单历元伪距.csv` — 单历元伪距明细
- `outputs/basic/<nav目录>/module3_单历元定位结果.txt` — 单历元 SPP 解算报告

**模块四**
- `outputs/basic/<nav目录>/module4_连续定位结果.csv` — 连续定位结果
- `outputs/basic/<nav目录>/module4_误差统计.txt` — 误差统计
- `outputs/basic/<nav目录>/module4_误差曲线.png` — 误差曲线
- `outputs/basic/<nav目录>/module4_轨迹图.png` — 经纬度轨迹图
- `outputs/basic/<nav目录>/module4_真实与估计轨迹对比.png` — 真实轨迹与估计轨迹对比
- `outputs/basic/<nav目录>/module4_卫星DOP曲线.png` — 卫星数与 DOP 曲线
- `outputs/basic/<nav目录>/module4_DOP与误差分析.png` — DOP 与误差关系分析

**模块五**
- `outputs/basic/<nav目录>/module5_系统测试报告.txt` — 系统整合与测试报告

此外，三场景 GUI 批量解算（`python basic/gui_scenario_runner.py`）会输出：
- `outputs/basic/gui_scenario_runner/gui_three_scenario_summary.csv`
- `outputs/basic/gui_scenario_runner/gui_three_scenario_report.txt`
- `outputs/basic/gui_scenario_runner/scenario_1/...`
- `outputs/basic/gui_scenario_runner/scenario_2/...`
- `outputs/basic/gui_scenario_runner/scenario_3/...`

## 提高部分输出文件

运行 `python enhance/run_enhance.py` 后，输出统一存放在 `outputs/enhance/`，其中 txt/csv 结果位于 `outputs/enhance/ml/`：

- `ml/机器学习数据集.csv` — 机器学习数据集
- `ml/scenarios/scenario1/module4_连续定位结果.csv` — 增强场景 1 连续定位结果
- `ml/scenarios/scenario2/module4_连续定位结果.csv` — 增强场景 2 连续定位结果
- `ml/scenarios/scenario3/module4_连续定位结果.csv` — 增强场景 3 连续定位结果
- `models/线性回归模型.joblib` — 线性回归模型
- `models/随机森林模型.joblib` — 随机森林模型
- `ml/predictions/线性回归_补偿预测.csv` — 线性回归补偿结果
- `ml/predictions/随机森林_补偿预测.csv` — 随机森林补偿结果
- `ml/模型对比汇总.csv` — 模型对比指标
- `ml/补偿效果统计.txt` — 补偿统计报告
- `figures/线性回归_误差曲线.png` — 线性回归误差曲线
- `figures/随机森林_误差曲线.png` — 随机森林误差曲线
- `figures/模型对比柱状图.png` — 模型对比柱状图
- `figures/预测与真实误差对比.png` — 预测误差散点图
- `ml/技术报告.txt` — 技术报告

## 导航文件

RINEX NAV 导航文件统一放在项目根目录下的 `nav/` 文件夹中：
- `nav/tarc0910.26b_cnav`（默认文件）
- `nav/tarc1210.26b_cnav` 至 `nav/tarc1300.26b_cnav`（多场景与扩展测试文件）

文件格式为 RINEX 3.x 北斗三号 CNAV 导航电文。解析器不再根据后缀判断文件类型，`.26b_cnav`、`.cnav`、无后缀文件均统一按 CNAV 格式解析。

## 注意事项

1. **不读取 .obs 文件**：本项目使用模拟伪距，不依赖真实观测文件。
2. **数据格式**：默认按 BDS-3 CNAV 格式解析，仅保留北斗三号卫星（PRN >= 19）。
3. **输出隔离**：`outputs/basic/` 与 `outputs/enhance/` 相互独立，分别存放基础部分与提高部分的输出结果。
4. **时间系统**：本项目使用 BDS-3 CNAV 文件进行仿真，程序内部将导航文件 toc/toe 与用户设置的仿真历元统一按 BDT 时间系统处理。由于本项目不读取 OBS 文件，不涉及真实观测文件中的 GPST、UTC 与 BDT 转换，因此当前时间处理在仿真系统内部是自洽的。若后续扩展到真实 OBS 解算，需要增加严格的时间系统转换模块。
5. **收敛阈值**：SPP 迭代以三维坐标改正量范数作为收敛判据，默认收敛阈值为 `1e-2 m`（1 cm）。当连续两次迭代的三维坐标改正量小于该阈值时，认为解算收敛。
