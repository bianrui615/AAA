# 北斗 SPP 定位解算系统 — 提高部分（机器学习误差补偿）

## 简介

在基础部分（`basic/`）已实现的 SPP 单点定位之上，本模块用机器学习对定位误差建模并补偿。采用 **LOSO（Leave-One-Scenario-Out）交叉验证 + Pipeline 标准化 + Baseline 对比** 的科学评估流程，确保结论可信。

## 依赖

```powershell
pip install scikit-learn joblib pandas matplotlib numpy
# GUI 可选：
pip install PyQt5
```

## 文件结构与作用

```
enhance/
  enhance_config.py    # 场景列表（默认 9 个）、14 维特征列、3 维标签列、所有输出路径常量
  dataset_builder.py   # 跑场景 → 模拟伪距 → SPP 解算 → 抽取特征 → 写 机器学习数据集.csv
                       # （场景已有 basic CSV 时直接读取，否则用 NAV 文件自动解算）
  train_models.py      # 核心：LOSO 交叉验证训练 LR 与 RF，输出 OOF 预测 / 每折指标 /
                       # mean±std 汇总 / Baseline 对比；CV 后用全量数据训练最终发布模型
  compensate.py        # 消费 train_models 的 OOF 预测，写 219 行（或更多）补偿 CSV
  evaluate_models.py   # 读补偿 CSV，输出 补偿效果统计.txt / 模型对比汇总.csv 与 4 张图表
  run_enhance.py       # 统一 CLI 入口，串联以上 5 步并写技术报告 / 训练流程评估报告
  enhance_gui.py       # PyQt5 GUI；含「场景配置」标签页（勾选 / 编辑 / 新增 / 删除场景）
  README_enhance.md    # 本文件
```

| 文件 | 一句话作用 |
|---|---|
| `enhance_config.py` | 配置中心：9 个默认场景 + 特征列表 + 输出路径，**改场景就改这里** |
| `dataset_builder.py` | 把场景跑成 CSV，是机器学习数据的"工厂" |
| `train_models.py` | 训练 + 评估的核心，LOSO 全部逻辑都在这里 |
| `compensate.py` | 把模型预测的误差加回 SPP 坐标，生成补偿后结果 |
| `evaluate_models.py` | 把 CSV 整理成报告 + 图表 |
| `run_enhance.py` | 一键跑全流程的 CLI 入口 |
| `enhance_gui.py` | 图形界面，支持自定义场景 |

## 运行方式

### 方式一：CLI（最简单，用默认 9 场景）

```powershell
python enhance\run_enhance.py
```

约 5–10 秒跑完。终端会打印：
- 9 个场景的 LOSO 折划分
- Baseline（不补偿）RMS
- 线性回归 / 随机森林 各自 LOSO mean±std
- 是否优于 Baseline
- 所有输出文件路径

### 方式二：GUI（支持自定义场景）

```powershell
python enhance\enhance_gui.py
```

GUI 右侧标签页：
- **场景配置**：表格展示 9 个默认场景，每行有勾选框
  - 勾选/取消勾选 → 控制是否参与 LOSO（折数 = 勾选场景数，最少 2）
  - **新增场景**：弹窗填表，NAV 下拉自动扫描 `nav/` 目录
  - **编辑选中** / **删除选中** / **全选** / **全不选** / **恢复默认**
- **数据集预览**：跑完显示数据集前 100 行
- **训练指标**：LOSO mean±std 汇总
- **误差对比图** / **散点图**：可视化
- **报告**：技术报告 + 训练流程评估报告

左侧操作流程：先在「场景配置」选好场景 → 点击「一键运行全部」即跑通整链路。

## 怎么检验结果

### 1. 看终端输出

健康的输出长这样（默认 9 场景）：
```
LOSO-CV, n_folds=9, n_samples=656
Baseline (不补偿)        RMS = 7.4174 m
线性回归 (LOSO mean±std) RMS = 7.8305 ± 1.5535 m  改善 -17.75% ± 46.20%
随机森林 (LOSO mean±std) RMS = 7.1243 ± 1.7140 m  改善 +0.73% ± 3.41%
较优模型（LOSO mean RMS 更低且优于 Baseline）：随机森林
```

判断指标：
- **`随机森林 mean RMS < Baseline`** → 模型有效。
- **`std` 越小** → 评估越稳定（场景越多 std 越收敛）。
- **`改善 > 0%`** → 平均而言比"不补偿"好。

如果两个模型 mean RMS 都 ≥ Baseline，会打印 `[WARN]`，并提示查看 `训练流程评估报告.md`。

### 2. 看 CSV / TXT 报告

| 文件 | 内容 |
|---|---|
| `outputs/enhance/ml/CV每折指标.csv` | 每折逐行：fold / 测试场景 / RMS_before / RMS_after / 改善% / 训练 RMS / MAE / R² |
| `outputs/enhance/ml/CV汇总指标.csv` | Baseline / LR / RF 三行，含 `beats_baseline` 标志 |
| `outputs/enhance/ml/CV汇总指标.txt` | 同上但中文格式化 |
| `outputs/enhance/ml/补偿效果统计.txt` | 详细统计 + 每折表 + 结论 |
| `outputs/enhance/ml/技术报告.txt` | 完整技术报告（流程、模型、结果、改进方向） |
| `outputs/enhance/ml/训练流程评估报告.md` | 对训练评估方案 8 个问题的诊断报告 |

**重点看 `CV每折指标.csv` 中的 `train_rmse_3d` 列**：
- `train_rmse_3d` ≈ `rmse_3d_after` → 模型泛化良好
- `train_rmse_3d` << `rmse_3d_after` → 存在过拟合

### 3. 看图

| 图 | 怎么解读 |
|---|---|
| `outputs/enhance/figures/模型对比柱状图.png` | 3 根柱（Baseline / LR / RF）含误差棒，柱越矮越好 |
| `outputs/enhance/figures/线性回归_误差曲线.png` | 按 fold 着色（每折一种颜色），实线 = 补偿后，虚线 = 补偿前，黑色虚线 = Baseline |
| `outputs/enhance/figures/随机森林_误差曲线.png` | 同上 |
| `outputs/enhance/figures/预测与真实误差对比.png` | 散点越靠近 y=x 越好 |

### 4. 单独单元测试（按模块）

```powershell
# 只构建数据集（不训练）
python -c "from enhance.dataset_builder import build_dataset; build_dataset()"

# 只训练 + 评估（数据集已存在）
python -c "from enhance.train_models import train_models; from pathlib import Path; train_models(Path('outputs/enhance/ml/机器学习数据集.csv'))"

# 加载训练好的模型做预测
python -c "import joblib; m = joblib.load('outputs/enhance/models/随机森林模型.joblib'); print(m.predict([[12,15,2.1,3.5,0.5,5,0,2.2e7,5e6,2.0e7,4e6,45,15,75]]))"
```

## 输出文件清单

所有输出统一放在 `outputs/enhance/` 下。

```
outputs/enhance/
├── ml/
│   ├── scenarios/scenario4..9/         # enhance 自生成的场景结果
│   ├── 机器学习数据集.csv              # 训练数据源（feature + label）
│   ├── LOSO_CV_划分汇总.txt           # 每折训练/测试场景明细
│   ├── CV每折指标.csv                  # 9 折 × 2 模型的详细指标
│   ├── CV汇总指标.csv / .txt          # mean±std + 是否优于 baseline
│   ├── 补偿效果统计.txt                # 中文统计报告
│   ├── 模型对比汇总.csv                # OOF + LOSO 综合对比
│   ├── 技术报告.txt                    # 完整技术报告
│   ├── 训练流程评估报告.md             # 8 个诊断问题的回答
│   └── predictions/
│       ├── 线性回归_补偿预测.csv       # OOF 预测（含 fold_index 列）
│       └── 随机森林_补偿预测.csv
├── models/
│   ├── 线性回归模型.joblib             # 用全量数据 fit 的 Pipeline
│   └── 随机森林模型.joblib
└── figures/
    ├── 线性回归_误差曲线.png           # 按 fold 着色
    ├── 随机森林_误差曲线.png
    ├── 模型对比柱状图.png              # 含 Baseline + LOSO 误差棒
    └── 预测与真实误差对比.png          # 按 fold 着色散点
```

## 设计要点

- **不修改 `basic/` 核心逻辑**：仅调用 `basic/module1/module3` 等公开函数。
- **不使用 `.obs` 文件**：所有数据基于 BDS-3 CNAV 广播星历 + 伪距模拟模型生成。
- **LOSO 交叉验证**：按 `scenario_name` 分组，每个场景轮流作测试集，避免同场景相邻历元的时序泄漏，直接度量"对新场景的泛化能力"。
- **Pipeline 内置 StandardScaler**：每折只用训练数据 fit scaler，杜绝测试统计量泄漏。
- **Baseline 对比**：把"不补偿"作为零模型，模型必须明显优于零模型才算成功。
- **评估与发布解耦**：CV 用于估计泛化误差，最终模型用全量 656 样本重新拟合，仅用于推理。
- **场景可配置**：默认 9 个场景在 `enhance_config.py`；GUI「场景配置」标签页可勾选/编辑/新增/删除，不需要改代码。
- **可复现性**：所有随机过程指定 `random_state` / `seed`。

## 常见问题

**Q: 想加更多场景怎么办？**
A: 两种方式：
  1. 改 `enhance_config.py:42-153` 的 `SCENARIOS` 列表，重启 CLI；
  2. GUI 中点击「场景配置」标签页 → 「新增场景」，填表后直接生效。

**Q: 第一次跑很慢？**
A: scenario4-9 没有 basic CSV 缓存，需要从 NAV + 模拟伪距重新解算，约 4–5 秒；之后跑会直接复用 `outputs/enhance/ml/scenarios/scenario*/` 下的缓存（如果该路径存在）。如果想强制重跑，删除对应目录即可。

**Q: 模型表现差怎么办？**
A: 看 `训练流程评估报告.md`。当前结论：14 维历元级聚合特征 + 9 场景下，RandomForest 刚好打过 Baseline，LinearRegression 还不行。改进方向：继续扩到 15+ 场景；加入卫星级特征（每颗卫星仰角/残差等）；尝试 Ridge / XGBoost。
