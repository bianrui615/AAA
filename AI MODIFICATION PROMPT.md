# 北斗 SPP 定位项目修改任务提示词

> 把这份文档完整地发给 AI 编程助手（Claude / GPT / Cursor 等），它会按顺序执行 6 项任务。
> 任务之间相互独立，可以分批执行，也可以一次性完成。

-----

## 0. 项目背景（必读）

这是一个**已经能运行**的北斗三号 SPP 单点定位 Python 项目，结构如下：

```
basic/
├── module1.py      # NAV 解析 + 卫星位置/钟差 + 模拟伪距生成（约 1250 行，最重）
├── module2.py      # 卫星位置/钟差输出层（约 380 行，组织层）
├── module3.py      # 单点定位最小二乘 SPP 解算（约 580 行）
├── module4.py      # 连续定位与可视化（约 520 行）
├── module5.py      # 系统整合 + 多场景测试（约 710 行，主入口）
└── rinex_gui.py    # PyQt 图形界面（约 1360 行）
enhance/
├── enhance_config.py    # ML 场景/特征配置
├── dataset_builder.py   # 构建 ML 数据集
├── train_models.py      # 训练线性回归 + 随机森林
├── compensate.py        # 误差预测与坐标补偿
├── evaluate_models.py   # 评估与可视化
└── run_enhance.py       # ML 部分主入口
nav/                # BDS-3 CNAV 导航文件（多个）
outputs/basic/      # 基础部分输出
outputs/enhance/    # ML 部分输出
```

**重要约束**：

- 项目**不使用 OBS 文件**，伪距由 `module1.simulate_pseudorange` 用高斯模型生成
- 核心 GNSS 算法**必须手写**，禁止调用 RTKLIB、georinex 等第三方 GNSS 库
- 允许使用：numpy、matplotlib、scikit-learn、joblib、PyQt5/6
- 代码风格：中文注释、类型注解、`from __future__ import annotations`
- 输出目录约定：基础部分写到 `outputs/basic/`，ML 部分写到 `outputs/enhance/`

**当前问题**：

- 模拟伪距生成时电离层（高斯均值 10 m）、对流层（高斯均值 4 m）、卫星钟差（约 60 m 级别）作为误差源加进去，但 SPP 解算时**完全没有扣除这些确定性偏差**，全部压到接收机钟差里，导致定位 RMS 约 60 m，明显偏高
- ML 训练样本只有 26 个，效果差
- 模块 4 没有 DOP 与精度关系的显式可视化
- 提高部分（ML）没有独立 GUI
- 缺少设计报告和实验报告文档

-----

## 任务 1：在 SPP 解算中加入 Saastamoinen 对流层修正、电离层修正、卫星钟差修正

### 目标

让喂给最小二乘的伪距是”修正后的伪距”，把 SPP 定位 RMS 误差从约 60 m 降到 10 m 以下。

### 实现要点

#### 1.1 在 `basic/module3.py` 顶部新增三个修正函数

```python
import math
from typing import Tuple

# 单位：米。简化的"标准大气"对流层模型，已经叫做 Saastamoinen 简化形式
def saastamoinen_tropospheric_delay(
    elevation_deg: float,
    receiver_height_m: float = 0.0,
    pressure_hpa: float = 1013.25,
    temperature_k: float = 288.15,
    humidity: float = 0.5,
) -> float:
    """
    Saastamoinen 对流层延迟模型（简化版）。
    
    返回某高度角下的对流层延迟，单位米。
    
    公式：
        T = 0.002277 / sin(elevation) * (P + (1255/T + 0.05)*e - tan²(zenith))
    其中：
        P: 气压 (hPa)，标准 1013.25
        T: 温度 (K)，标准 288.15
        e: 水汽压 (hPa)，由相对湿度计算 e = humidity * 6.108 * exp(...)
    
    高度角 < 5° 时强制按 5° 计算，避免数值发散。
    """
    el_rad = math.radians(max(elevation_deg, 5.0))
    # 简化的水汽压
    e_vapor = humidity * 6.108 * math.exp(
        (17.15 * (temperature_k - 273.15) - 4684.0) / (temperature_k - 38.45)
    )
    zenith_rad = math.pi / 2.0 - el_rad
    delay = (0.002277 / math.sin(el_rad)) * (
        pressure_hpa
        + (1255.0 / temperature_k + 0.05) * e_vapor
        - math.tan(zenith_rad) ** 2
    )
    return delay


def simple_ionospheric_delay(
    elevation_deg: float,
    zenith_delay_m: float = 10.0,
) -> float:
    """
    简化电离层延迟模型：基于高度角的余割映射函数。
    
    I(el) = zenith_delay / sin(elevation)
    
    返回单位米。高度角 < 5° 时按 5° 计算。
    """
    el_rad = math.radians(max(elevation_deg, 5.0))
    return zenith_delay_m / math.sin(el_rad)


def apply_pseudorange_corrections(
    raw_pseudorange: float,
    satellite_clock_bias_s: float,
    elevation_deg: float,
    receiver_height_m: float = 0.0,
    enable_satellite_clock: bool = True,
    enable_tropospheric: bool = True,
    enable_ionospheric: bool = True,
    iono_zenith_delay_m: float = 10.0,
) -> Tuple[float, dict]:
    """
    对单颗卫星的伪距进行确定性修正：
        P_corrected = P_raw + c·dt_sat - T_iono - T_tropo
    
    返回 (修正后伪距, 修正量明细字典)。
    
    各开关默认 True，允许通过参数关闭某项修正以做对比实验。
    """
    C = 299_792_458.0
    sat_clock_m = C * satellite_clock_bias_s if enable_satellite_clock else 0.0
    tropo_m = (
        saastamoinen_tropospheric_delay(elevation_deg, receiver_height_m)
        if enable_tropospheric
        else 0.0
    )
    iono_m = (
        simple_ionospheric_delay(elevation_deg, iono_zenith_delay_m)
        if enable_ionospheric
        else 0.0
    )
    corrected = raw_pseudorange + sat_clock_m - tropo_m - iono_m
    return corrected, {
        "satellite_clock_correction_m": sat_clock_m,
        "tropospheric_correction_m": tropo_m,
        "ionospheric_correction_m": iono_m,
    }
```

#### 1.2 修改 `module3.solve_spp` 的签名和调用方式

在 `solve_spp` 入口处增加一个新参数 `apply_corrections: bool = True`，以及辅助参数：

- `satellite_clock_biases: Optional[Dict[str, float]] = None`  每颗卫星的钟差（秒），由 module4 计算后传入
- `satellite_elevations: Optional[Dict[str, float]] = None`  每颗卫星的高度角（度）

在最小二乘迭代之前，若 `apply_corrections=True`，对每个 `pseudoranges[sat_id]` 调用 `apply_pseudorange_corrections` 得到修正后的伪距，**用修正后的字典进入迭代**。

不要破坏向后兼容：参数默认 `apply_corrections=False` 也能正常跑（退化为现有行为）。或者默认 `True` 但允许调用方传 `False` 关掉。**推荐默认 True**。

#### 1.3 修改 `module4.run_continuous_positioning`

在每个历元构造完 `satellite_positions` 和 `pseudoranges` 之后：

- 调用 `compute_satellite_clock_bias` 算出每颗卫星的 `dt_sat`，组成 `satellite_clock_biases` 字典
- 用 `compute_elevation`（来自 module1）算出每颗卫星的高度角，组成 `satellite_elevations` 字典
- 把这两个字典传给 `solve_spp`，并设 `apply_corrections=True`

#### 1.4 module5 多场景测试中保持开启

在 module5 调用 module4 时无须改动（module4 内部默认开启就行）。

#### 1.5 GUI 加一个复选框（可选）

在 `rinex_gui.py` 的”解算参数”分组里加一个 `QCheckBox("启用伪距修正（卫星钟差/对流层/电离层）")`，默认勾选；状态传递给 `PositioningWorker` 再传给 `solve_spp`。

### 验收标准

- `python basic/module5.py` 跑完后，`outputs/basic/module4_error_statistics.txt` 里 **RMS 三维误差应该 < 15 m**（理想 < 5 m）
- `outputs/basic/module4_continuous_position_results.csv` 里新增 3 列：`sat_clock_correction_m`、`tropo_correction_m`、`iono_correction_m`（取所有卫星的均值即可）
- 单元测试：在 `tests/test_corrections.py` 里加 3 个最小测试：高度角 90° 时对流层 ≈ 2.3 m，高度角 30° 时 ≈ 4.6 m，电离层在 90° 时 = 10 m。

-----

## 任务 2：模块 4 增加 “DOP 与定位精度关系” 显式分析

### 目标

直观展示 PDOP / 卫星数与定位误差的相关性，回答题目模块 4 第 2 条：“分析卫星数量、DOP 值与定位精度的关系”。

### 实现要点

#### 2.1 在 `basic/module4.py` 的 `plot_results` 中追加新图

新增一张图 `module4_dop_error_analysis.png`，包含 4 个子图（2×2 布局）：

1. 左上：PDOP vs error_3d 散点图 + 线性拟合直线 + 标注相关系数 r
1. 右上：GDOP vs error_3d 散点图 + 线性拟合 + r
1. 左下：satellite_count vs error_3d 散点图（用箱线图或散点 + 平均线）
1. 右下：误差直方图（三维定位误差分布）

参考代码：

```python
import numpy as np

def _plot_dop_error_analysis(results, output_path):
    valid = [r for r in results if r["status"] == "成功"]
    if len(valid) < 3:
        return
    errors = np.array([float(r["error_3d"]) for r in valid])
    pdops = np.array([float(r["PDOP"]) for r in valid])
    gdops = np.array([float(r["GDOP"]) for r in valid])
    counts = np.array([int(r["satellite_count"]) for r in valid])

    # Pearson 相关系数
    def _corr(x, y):
        if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    r_pdop = _corr(pdops, errors)
    r_gdop = _corr(gdops, errors)
    r_count = _corr(counts, errors)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 子图 1：PDOP vs error
    axes[0, 0].scatter(pdops, errors, alpha=0.6)
    if not math.isnan(r_pdop):
        p = np.polyfit(pdops, errors, 1)
        x_line = np.linspace(pdops.min(), pdops.max(), 50)
        axes[0, 0].plot(x_line, np.polyval(p, x_line), "r--", linewidth=1.2)
    axes[0, 0].set_xlabel("PDOP")
    axes[0, 0].set_ylabel("三维定位误差 (m)")
    axes[0, 0].set_title(f"PDOP vs 误差 (r={r_pdop:.3f})")
    axes[0, 0].grid(True, alpha=0.3)

    # 子图 2：GDOP vs error（类似）
    # 子图 3：卫星数 vs error
    # 子图 4：误差直方图 hist

    plt.tight_layout()
    plt.savefig(output_path / "module4_dop_error_analysis.png", dpi=160)
    plt.close()
    
    return {"r_pdop_error": r_pdop, "r_gdop_error": r_gdop, "r_count_error": r_count}
```

#### 2.2 把相关系数写入 `module4_error_statistics.txt`

在 `save_error_statistics` 函数里追加：

```
DOP 与定位精度相关性分析
========================================
PDOP 与三维误差相关系数 r = 0.823
GDOP 与三维误差相关系数 r = 0.811
卫星数与三维误差相关系数 r = -0.456
（r > 0 表示正相关，r 绝对值 > 0.5 视为显著相关）

定性结论：
PDOP 与误差正相关（卫星几何越差，定位精度越低），符合理论预期。
卫星数与误差负相关（卫星越多，定位精度越好），符合冗余观测理论。
```

### 验收标准

- `outputs/basic/module4_dop_error_analysis.png` 文件生成且包含 4 个子图
- `module4_error_statistics.txt` 里有相关系数和定性结论
- GUI 误差曲线 Tab 旁边增加一个 “DOP 分析” Tab，显示这张图（可选）

-----

## 任务 3：附加题增加 ML 样本量

### 目标

当前每个场景仅 13 个历元，3 个场景共 26 个样本（训练 18 / 测试 8）。增加到每场景 100~300 个，总样本 500+。

### 实现要点

#### 3.1 修改 `enhance/enhance_config.py` 的场景配置

```python
SCENARIOS: List[ScenarioConfig] = [
    ScenarioConfig(
        name="scenario_1_default",
        nav_file_path="nav/tarc0910.26b_cnav",
        receiver_true_position=(-2267800.0, 5009340.0, 3221000.0),
        start_time=datetime(2026, 4, 1, 0, 0, 0),
        end_time=datetime(2026, 4, 1, 2, 0, 0),    # 改：2 小时
        interval_seconds=30,                         # 改：30 秒采样
        random_seed=2026,
        elevation_mask_deg=0.0,
    ),
    ScenarioConfig(
        name="scenario_2_different_seed",
        nav_file_path="nav/tarc1210.26b_cnav",     # 换不同的 NAV 文件
        receiver_true_position=(-2350000.0, 5100000.0, 3150000.0),
        start_time=datetime(2026, 5, 1, 0, 0, 0),
        end_time=datetime(2026, 5, 1, 2, 0, 0),
        interval_seconds=30,
        random_seed=42,
        elevation_mask_deg=0.0,
    ),
    ScenarioConfig(
        name="scenario_3_elevation_mask",
        nav_file_path="nav/tarc1230.26b_cnav",
        receiver_true_position=(-2200000.0, 4950000.0, 3300000.0),
        start_time=datetime(2026, 5, 3, 0, 0, 0),
        end_time=datetime(2026, 5, 3, 2, 0, 0),
        interval_seconds=30,
        random_seed=2027,
        elevation_mask_deg=10.0,                     # 启用高度角截止
    ),
]
```

每场景：2 小时 × 3600 / 30 = 240 个历元 → 3 个场景 720 个样本 → 训练 504 / 测试 216。

#### 3.2 注意星历有效期

NAV 文件里星历有效期通常 ±2 小时。如果 `end_time - start_time` 太长，会跨多个星历区间，确保 `select_ephemeris` 会自动切换最近星历——查一下现有实现是否支持。如果不支持，需要修复。

#### 3.3 加入随机森林超参数调优

在 `enhance/train_models.py` 里增加一个可选的 GridSearchCV 流程（默认关闭，可通过参数开启）：

```python
from sklearn.model_selection import GridSearchCV

def train_models(dataset_path, test_size=0.3, random_state=2026, 
                 enable_grid_search=False):
    # ... 现有代码 ...
    
    if enable_grid_search:
        param_grid = {
            "n_estimators": [100, 200, 400],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
        }
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=2026, n_jobs=-1),
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        rf_grid.fit(X_train, y_train)
        rf_model = rf_grid.best_estimator_
        print(f"[GridSearch] 最佳参数：{rf_grid.best_params_}")
    else:
        rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, ...)
        rf_model.fit(X_train, y_train)
```

### 验收标准

- `outputs/enhance/ml_dataset.csv` 行数 >= 500
- `outputs/enhance/ml_compensation_statistics.txt` 中**随机森林**的 RMS 改善 > 30%（任务 1 修正完成后，再做 ML 补偿应该更有效）
- 训练时间不超过 2 分钟

-----

## 任务 4：为附加题（ML）做独立 GUI

### 目标

新文件 `enhance/enhance_gui.py`，提供 ML 训练与误差补偿的图形化操作界面。

### 实现要点

#### 4.1 界面结构（PyQt5/6 兼容）

```
+--------------------------------------------------------+
| 左侧控制面板（固定 400 px）          | 右侧标签页区     |
|--------------------------------------|------------------|
| [1] 数据集设置                       | Tab1: 数据集预览 |
|   - 数据集 CSV 路径（浏览）          |   表格 + 统计    |
|   - 训练/测试比例 SpinBox            |                  |
|   - 数据集随机种子                   | Tab2: 训练指标   |
|                                      |   R² / MSE 等    |
| [2] 模型设置                         |                  |
|   - 模型选择 (下拉)                  | Tab3: 误差对比图 |
|     线性回归 / 随机森林 / 两个都跑   |   补偿前后曲线   |
|   - 随机森林参数                     |                  |
|     n_estimators SpinBox             | Tab4: 散点图     |
|     max_depth SpinBox                |   预测 vs 真实   |
|     启用 GridSearch CheckBox         |                  |
|                                      | Tab5: 报告       |
| [3] 操作按钮                         |   读 TXT 报告    |
|   [构建数据集] (调 dataset_builder)  |                  |
|   [训练模型]                         |                  |
|   [补偿与评估]                       |                  |
|   [一键运行全部]                     |                  |
|                                      |                  |
| [4] 日志框 (QTextEdit)               |                  |
+--------------------------------------------------------+
```

#### 4.2 必须实现的交互功能

1. **构建数据集**：调用 `enhance.dataset_builder.build_dataset`，进度条显示
1. **训练模型**：调用 `enhance.train_models.train_models`，把训练结果存到 `self.train_result`
1. **补偿与评估**：调用 `enhance.compensate.run_compensation` 和 `enhance.evaluate_models.evaluate_and_visualize`
1. **可视化嵌入**：用 `FigureCanvasQTAgg` 把 matplotlib 图嵌到 Tab 里，而不是只给文件路径
1. **日志同步**：所有 print 重定向到 GUI 日志框（参考 basic/rinex_gui.py 的做法）

#### 4.3 参考已有 GUI

直接复用 `basic/rinex_gui.py` 的 PyQt 兼容代码（顶部 try-import 块）、`MplCanvas` 类、`PositioningWorker` 后台线程模式——把这些模式照搬过来，只把业务逻辑换成 ML 流程。

#### 4.4 提供命令行入口

```python
# enhance/enhance_gui.py 末尾
def main() -> int:
    app = QApplication(sys.argv)
    window = EnhanceMainWindow()
    window.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
```

启动方式：`python enhance/enhance_gui.py`

### 验收标准

- `python enhance/enhance_gui.py` 能正常启动
- 一键运行按钮能完成全流程并在 GUI 内显示所有图表
- 程序无报错退出
- 与 `basic/rinex_gui.py` 风格一致

-----

## 任务 5：清理重复代码

### 目标

当前 `module1.simulate_pseudorange`、`module3.generate_simulated_pseudorange_record` 都实现了同一套高斯误差模型，应该只保留一处。

### 实现要点

#### 5.1 找出所有重复点

至少存在以下重复：

- `module1.py:836-876` `simulate_pseudorange`：使用 random.Random，输出 `iono_error/tropo_error` 字段
- `module3.py:85-150` `generate_simulated_pseudorange_record`：同样的误差模型，但字段名是 `ionosphere_error/troposphere_error`

#### 5.2 重构方案

**保留 `module1.simulate_pseudorange` 作为唯一实现**（因为它是数据流的起点），把 `module3` 里的两个函数改为薄包装：

```python
# module3.py 中
from basic.module1 import simulate_pseudorange

def generate_simulated_pseudorange_record(
    epoch_time, sat_id, sat_position, receiver_true_position,
    health=0.0, rng=None, receiver_clock_error=None,
):
    """生成单颗卫星模拟伪距记录。
    
    薄包装：底层调用 module1.simulate_pseudorange，
    输出添加 epoch_time、sat_id、sat_position 等元数据字段。
    """
    if rng is None:
        rng = random.Random()
    rho = geometric_distance(sat_position, receiver_true_position)
    sim = simulate_pseudorange(rho, rng)
    if receiver_clock_error is not None:
        # 覆盖随机生成的接收机钟差
        old_clk = sim["receiver_clock_error"]
        sim["receiver_clock_error"] = receiver_clock_error
        sim["pseudorange"] = sim["pseudorange"] - old_clk + receiver_clock_error
    
    return {
        "epoch_time": epoch_time.isoformat(sep=" "),
        "sat_id": sat_id,
        "sat_x": sat_position[0],
        "sat_y": sat_position[1],
        "sat_z": sat_position[2],
        "health": health,
        "rho": rho,
        # 兼容旧字段名
        "ionosphere_error": sim["iono_error"],
        "troposphere_error": sim["tropo_error"],
        # 新字段
        **sim,
        "simulated_pseudorange": sim["pseudorange"],
    }
```

#### 5.3 字段命名统一

整个项目里电离层误差字段应该叫 `iono_error`，对流层 `tropo_error`。如果 module2/module3 的下游代码使用了旧名 `ionosphere_error/troposphere_error`，要么改下游、要么在包装函数里同时提供两个名字以兼容。**推荐统一改为 `iono_error/tropo_error`**，跑通所有现有测试。

#### 5.4 删除冗余文件

检查 `outputs/basic/corrected_pseudorange.csv` 是否仍在生成；如果是历史遗留（不是任务 1 新生成的），删除生成代码。

### 验收标准

- `simulate_pseudorange` 误差模型只在 `module1.py` 中一处定义
- `module3.generate_simulated_pseudorange_record` 不再重复书写 `rng.gauss(0.0, 0.5)` 等行
- 所有现有功能（`module5.py` 与 `run_enhance.py`）跑通，数值结果不变（设同样 seed）

-----

## 任务 6：编写设计报告与实验报告

### 目标

按题目 PDF 第 8 节《实验报告要求》和第 3 节《文档要求》产出两份正式文档，存放为：

- `docs/设计报告.md`（约 2000~3000 字）
- `docs/实验报告.md`（约 5000~8000 字，含程序清单可放附录）

也可以生成 `.docx`，但 Markdown 文件更便于版本控制和后续修改。

### 6.1 设计报告 目录结构

```markdown
# 北斗定位解算全流程软件系统 — 设计报告

## 1. 需求分析
### 1.1 课程要求概述
### 1.2 功能性需求
   - 解析 RINEX NAV 文件
   - 计算卫星位置与钟差
   - 生成模拟伪距（项目特化，替代 OBS 文件读取）
   - 单点定位最小二乘解算
   - 连续定位与精度分析
   - GUI 交互
   - ML 误差补偿（附加题）
### 1.3 非功能性需求
   - 模块化、可扩展
   - 中文界面与注释
   - 跨平台（Windows / Linux）
   - 输出可视化与报告
### 1.4 约束与假设
   - 不读取 OBS 文件，使用伪距生成模型
   - 仅处理 BDS-3 卫星
   - 接收机为单频接收机

## 2. 系统功能设计
### 2.1 总体架构图
   [文字描述或 ASCII art 架构图]
### 2.2 模块划分
   - 表格：模块名 / 职责 / 输入 / 输出 / 依赖
### 2.3 关键流程图
   - 数据流图：NAV 文件 → 解析 → 卫星位置 → 模拟伪距 → SPP → 连续定位 → ML 补偿
   - 单点定位流程图（迭代最小二乘）
   - 多场景测试流程图
### 2.4 GUI 交互流程
   - 基础部分 GUI（rinex_gui.py）的标签页结构
   - 提高部分 GUI（enhance_gui.py）的工作流
### 2.5 数据结构设计
   - BroadcastEphemeris、SppSolution、AnalysisSummary、ScenarioConfig 各 dataclass 字段说明
   - 字典/列表组织：nav_data、pseudoranges、satellite_positions

## 3. 算法设计
### 3.1 卫星位置计算（开普勒方程 + 摄动修正 + GEO 特殊处理）
### 3.2 卫星钟差计算（多项式 + 相对论效应）
### 3.3 伪距修正（Saastamoinen 对流层、简化电离层、卫星钟差）
### 3.4 迭代最小二乘 SPP
### 3.5 DOP 计算
### 3.6 ECEF ↔ BLH 坐标转换
### 3.7 ML 模型选择依据（LinearRegression vs RandomForest）

## 4. 测试方案设计
### 4.1 单元测试
### 4.2 多场景测试（3 个内置场景）
### 4.3 ML 补偿对比测试
```

**架构图请用 ASCII 字符画或 Mermaid 语法（GitHub 支持）画清楚 5 个模块的关系**：

```
┌─────────────────────────────────────────────────────────┐
│                      basic/module5.py                    │
│                       （系统整合层）                       │
└─────┬──────────┬────────────┬────────────┬───────────────┘
      │          │            │            │
      ▼          ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ module1  │ │ module2  │ │ module3  │ │ module4  │
│ NAV 解析 │ │ 卫星位置 │ │ SPP 解算 │ │ 连续定位 │
│ 伪距生成 │ │ 钟差计算 │ │ 修正函数 │ │ 可视化   │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
```

### 6.2 实验报告 目录结构

```markdown
# 北斗定位解算全流程软件系统 — 实验报告

## 1. 实验目的
   [复述题目实验目的，并明确说明实验最终完成的功能]

## 2. 实验方法和步骤
### 2.1 实验方法
   - 软件开发方法（迭代式开发）
   - 验证方法（与真实坐标对比、多场景测试）
### 2.2 实验步骤
   - 第 1-2 周：需求分析、查阅 ICD/RINEX 规范、设计文档
   - 第 3-6 周：实现 5 个模块 + GUI + ML
   - 第 7-8 周：调试、测试、验收

## 3. 实验软件总体设计
### 3.1 软件结构设计（模块结构图 + 模块说明）
### 3.2 数据结构设计（全局变量、dataclass、CSV 字段约定）

## 4. 详细设计及实现
### 4.1 模块一：NAV 解析与伪距生成
   - 算法原理：RINEX 3 CNAV 格式、广播星历参数定义
   - 流程图（伪代码或文字描述）
   - 关键代码片段
   - 运行界面/输出截图（贴 1-2 张 PNG）
### 4.2 模块二：卫星位置与钟差计算
   - 算法原理：开普勒方程迭代解、摄动修正公式、GEO 卫星 -5° 倾角修正
   - 流程图
   - 关键代码片段
   - **伪距修正算法详解**：Saastamoinen 公式推导、电离层简化模型推导
### 4.3 模块三：SPP 单点定位
   - 算法原理：观测方程线性化、设计矩阵 H、法方程 (HᵀH)δ = Hᵀv
   - 迭代收敛条件
   - DOP 公式
   - 流程图
### 4.4 模块四：连续定位与误差分析
   - 逐历元循环逻辑
   - 误差统计公式
   - 可视化图表说明（贴所有 PNG 截图）
   - **DOP 与精度关系分析**：相关系数解读
### 4.5 模块五：系统整合与 GUI
   - GUI 截图（贴 2-3 张）
   - 多场景测试结果
### 4.6 附加题：ML 误差补偿
   - 特征工程依据
   - 数据集构建流程
   - LinearRegression / RandomForest 原理与对比
   - 补偿效果分析
   - **ML 部分 GUI 截图**
### 4.7 调试过程中遇到的错误与解决方案
   **每个学生不得雷同**，至少描述 3~5 个真实遇到的问题：
   - 例如：GEO 卫星坐标算出来偏离地心轨道几万公里 → 加入 -5° 倾角修正
   - 例如：高纬度场景下迭代不收敛 → 检查初值
   - 例如：matplotlib 中文乱码 → 设置 font.sans-serif
   - 例如：PyQt5/6 兼容性 → 写 try-import 块
   - 例如：星历跨周边界时间归一化错误
   - 每条要写明问题现象、分析过程、解决方案。

## 5. 测试结果与误差分析
### 5.1 单场景测试结果（截图 + 数值表）
### 5.2 多场景对比表
### 5.3 ML 补偿前后对比
### 5.4 误差来源分析
   - 接收机钟差吸收量
   - 残余电离层/对流层误差
   - 几何精度因子（DOP）影响

## 6. 结论
   - 是否达到设计要求（按 5 个模块和附加题逐项打钩）
   - 项目特点与亮点
   - 不足之处
   - 改进建议

## 7. 结束语（心得体会）
   - 学到了什么（GNSS 理论 / 工程实践 / 团队协作？）
   - 遇到的困难
   - 对北斗系统的理解加深

## 8. 程序清单
   - 列出全部源文件、行数、主要函数
   - 也可以放到附录 A
```

#### 6.3 写作要求

- **每章内容要充实**，不能只列标题
- **配图至少 8 张**：架构图、流程图、模块输出截图、GUI 截图、误差曲线、DOP 散点、ML 对比
- 公式用 LaTeX 语法（`$...$` 行内，`$$...$$` 行间），后续可导出为 PDF / Word
- 调试错误记录**必须个人化**，写真实经历，不要套话
- 程序清单部分可以简略，写出每个 .py 文件的功能和主要函数即可

#### 6.4 自动生成截图（可选辅助）

在生成报告之前先运行：

```bash
python basic/module5.py
python enhance/run_enhance.py
```

确保 `outputs/basic/` 和 `outputs/enhance/` 下的所有 PNG 都存在，然后在报告 Markdown 里用相对路径引用：

```markdown
![模块四误差曲线](../outputs/basic/module4_error_curve.png)
```

### 验收标准

- `docs/设计报告.md` 文件存在，字数 >= 2000
- `docs/实验报告.md` 文件存在，字数 >= 5000，包含至少 8 张配图引用
- 两份文档目录结构与题目 PDF 第 8 节要求完全对应
- 调试错误记录章节包含至少 3 个具体技术问题及其解决方案

-----

## 任务执行顺序建议

1. **先做任务 1**（伪距修正）—— 影响所有数值结果，越早做越好
1. **再做任务 5**（去重）—— 清理代码方便后续维护
1. **再做任务 2**（DOP 分析）—— 小改动，立刻让模块 4 完整
1. **再做任务 3**（ML 增样本）—— 任务 1 完成后效果差异会更明显
1. **再做任务 4**（ML GUI）—— 独立工作量大，放后面
1. **最后做任务 6**（文档）—— 所有功能确定后再写最准确

-----

## 通用要求

1. **不要破坏现有功能**：所有任务完成后 `python basic/module5.py` 和 `python enhance/run_enhance.py` 必须依然能跑通
1. **保留中文注释和类型注解**风格
1. **每个新增函数都要有 docstring**，说明参数、返回值、单位
1. **不引入新的第三方 GNSS 库**（RTKLIB、georinex、pyproj 都禁用）
1. **测试通过后再提交**：用相同的 seed 跑两遍，确保结果一致
1. **commit 信息清晰**，建议每个任务一个 commit

完成后请给出修改清单和测试结果摘要。