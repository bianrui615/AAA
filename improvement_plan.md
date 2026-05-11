   # 北斗定位解算系统：提高部分实施计划（修订版）

## 一、题目修订后的目标

当前题目已调整为：**不用读取 RINEX OBS 观测文件，只需要基于 RINEX NAV 广播星历和伪距生成模型完成定位解算**。

因此，后续提高部分不再把 OBS 解析作为缺口；项目主线应保持为：

1. 解析 RINEX NAV 导航文件；
2. 根据广播星历计算北斗卫星位置与钟差；
3. 使用伪距生成模型生成每颗卫星的模拟伪距；
4. 对伪距进行钟差、电离层、对流层等修正或等效建模；
5. 使用迭代最小二乘完成 SPP 定位；
6. 完成多历元连续定位、误差统计、可视化、GUI 和测试报告；
7. 可选完成 AI 定位误差预测与补偿附加题。

基础题优先目标：在当前伪距仿真模型下，使系统测试 RMS 和最大三维误差稳定小于 1 m，并保证输出内容能够说明算法流程和误差来源。

---

## 二、现状与缺口分析

| 项目 | 当前状态 | 是否仍需做 |
|---|---|---|
| NAV 解析 | 已完成，`module1_nav_parser.py` 可解析北斗广播星历 | 保留，后续只需整理报告说明 |
| 卫星位置与钟差 | 已完成，`module2_satellite_position_clock.py` 已实现广播星历轨道、钟差、相对论修正 | 保留，可后续补充中间结果说明 |
| 伪距生成模型 | 已完成，`module3_spp_solver.py` 可生成模拟伪距并保存误差项 | 需要小幅增强：参数化噪声、可选粗差注入 |
| 高度角截止阈值 | 已完成，CLI/GUI/输出均已支持 | 不再作为缺口，只需做参数对比 |
| 伪距粗差剔除 | 已完成，已接入 SPP；当前已改为 MAD 优先的稳健判别 | 后续补充可控粗差测试 |
| 连续定位分析 | 已完成，可输出 CSV、误差统计和 3 类图 | 需要补充参数扫描/对比报告 |
| GUI | 已完成基本交互、实时表格、轨迹回放、误差曲线 | 保留，不作为精度提高重点 |
| AI 误差预测与补偿 | 未完成，属于附加题 | 仅在需要附加分时实现 |

---

## 三、基础提高部分（优先完成）

### P1：确定默认解算参数

**目的**：在当前伪距生成模型下稳定达到 1 m 内精度。

当前测试结论：

- `ELEVATION_MASK_DEG = 10.0` 时，RMS 可小于 1 m，但最大误差可能超过 1 m；
- `ELEVATION_MASK_DEG = 0.0` 时，保留全部健康卫星，几何强度更好，当前系统测试可稳定小于 1 m。

**实施要求**：

1. 默认使用 `ELEVATION_MASK_DEG = 0.0`。
2. 在报告中说明：当前为伪距仿真模型，低高度卫星没有额外多路径误差，因此保留全部卫星更有利于降低 DOP。
3. GUI 保留高度角阈值输入，方便验收时演示 0°、5°、10°、15° 参数影响。

### P2：增加参数扫描报告

**目的**：证明参数选择不是随意调参，而是有实验依据。

建议新增脚本或函数 `run_parameter_sweep()`，输出：

| 参数 | 输出指标 |
|---|---|
| 高度角阈值：0°、5°、10°、15°、20° | 成功历元数、平均卫星数、PDOP、GDOP、平均误差、RMS、最大误差 |
| 随机种子：至少 3 组 | 验证误差统计稳定性 |
| 伪距噪声标准差 | 分析观测噪声对定位精度的影响 |

输出文件建议：

- `outputs/basic/parameter_sweep_results.csv`
- `outputs/basic/parameter_sweep_report.txt`
- `outputs/basic/parameter_sweep_error.png`

### P3：伪距生成模型参数化

**目的**：让伪距模型更像可控实验，而不是把噪声硬编码在函数里。

建议新增数据类：

```python
@dataclass
class PseudorangeModelConfig:
    sisre_std_m: float = 0.3
    ionosphere_scale: float = 1.0
    troposphere_scale: float = 1.0
    receiver_clock_bias_mean_m: float = 60.0
    receiver_clock_bias_std_m: float = 12.0
    observation_noise_std_m: float = 0.2
    outlier_probability: float = 0.0
    outlier_magnitude_m: float = 200.0
```

后续将 `generate_simulated_pseudorange_record()` 和 `generate_simulated_pseudorange_records()` 改为接收该配置。

注意：`outlier_probability` 默认必须为 `0.0`，只能用于粗差剔除测试，不能默认污染主流程。

### P4：完善粗差剔除验证

**目的**：证明粗差剔除功能有效，同时不影响正常数据。

建议做两组测试：

1. 正常伪距：`outlier_probability = 0.0`，粗差剔除数应接近 0，定位精度稳定。
2. 注入粗差：`outlier_probability = 0.05` 或人工指定一颗卫星加 200 m 偏差，粗差剔除应能降低误差。

输出中保留：

- `rejected_outliers`
- 剔除前/剔除后卫星数
- 剔除前/剔除后误差对比

### P5：可选的加权最小二乘

**目的**：进一步提高模型合理性和报告含金量。

在普通最小二乘基础上增加可选 WLS：

- 权重可按高度角设置，例如 `weight = sin(elevation)^2`；
- 或按伪距模型中的噪声方差设置；
- 默认仍使用普通最小二乘，避免引入额外不稳定性；
- 输出普通 LS 与 WLS 的误差对比。

这不是基础必需项，但比直接上 AI 更适合作为“算法提高”。

---

## 四、AI 附加题计划（需要附加分时再做）

### AI-1：多场景数据生成

至少生成 3 组不同场景，每组包含多个历元：

1. 开阔场景：全部健康卫星，默认噪声；
2. 受限场景：提高高度角阈值或限制最多使用若干颗高仰角卫星；
3. 噪声增强场景：增大观测噪声或加入少量可控粗差；
4. 可选：不同接收机位置场景。

每个历元保存特征和标签。

建议特征：

- `num_sats`
- `pdop`, `gdop`
- `mean_pseudorange`, `std_pseudorange`
- `mean_elevation`, `min_elevation`, `max_elevation`
- `mean_iono_delay`, `mean_trop_delay`
- `receiver_clock_bias_m`
- `mean_sat_clock_bias_s`
- `rejected_outliers`

### AI-2：标签必须改为误差分量

原计划中“预测三维误差 `error_3d` 后直接修正 X/Y/Z”是不正确的。

正确标签应为三分量误差，例如：

```text
dx = solved_x - true_x
dy = solved_y - true_y
dz = solved_z - true_z
```

或转换到站心坐标：

```text
dE, dN, dU
```

模型预测三分量后，才能进行坐标补偿：

```text
X_compensated = X_solved - dx_pred
Y_compensated = Y_solved - dy_pred
Z_compensated = Z_solved - dz_pred
```

`error_3d` 只能作为评价指标，不能作为直接补偿量。

### AI-3：模型训练

可实现两种模型并比较：

1. 线性回归：便于解释；
2. 随机森林：适合非线性误差关系。

训练要求：

- 使用 70% / 30% 训练测试划分；
- 输出 RMSE、MAE、R2；
- 分别输出 `dx/dy/dz` 或 `dE/dN/dU` 的指标；
- 保存模型、特征名、标准化器。

建议文件：

- `ml_data_generator.py`
- `ml_train_model.py`
- `outputs/basic/ml_features_train.csv`
- `outputs/basic/ml_features_test.csv`
- `outputs/basic/ml_model.pkl`
- `outputs/basic/ml_training_report.txt`

### AI-4：补偿集成

新增 `ml_compensate.py`：

1. 加载模型和特征配置；
2. 从单历元定位结果和伪距明细中构建同样顺序的特征；
3. 预测三分量误差；
4. 修正坐标并重新计算经纬高；
5. 输出补偿前后误差统计。

输出字段：

- `X_compensated`, `Y_compensated`, `Z_compensated`
- `lat_compensated`, `lon_compensated`, `height_compensated`
- `error_3d_compensated`
- `dx_pred`, `dy_pred`, `dz_pred`

### AI-5：效果验证与报告

输出：

- `outputs/basic/ml_compensation_comparison.csv`
- `outputs/basic/ml_error_comparison.png`
- `outputs/basic/ml_error_bar_comparison.png`
- `docs/optional_ai_report.md`

报告必须说明：

1. 数据如何生成；
2. 特征为什么与误差相关；
3. 标签为什么用三分量而不是 `error_3d` 标量；
4. 模型训练指标；
5. 补偿前后 RMS、均值误差、最大误差对比；
6. 不足：仿真数据训练出的模型只对当前伪距模型有效，不能直接等同真实接收机效果。

---

## 五、建议执行顺序

```text
已完成并维护：
1. 高度角截止阈值
2. 伪距粗差剔除

基础提高优先：
3. 参数扫描报告
4. 伪距生成模型参数化
5. 粗差剔除专项验证
6. 可选加权最小二乘

附加题：
7. 多场景 AI 数据生成
8. 三分量误差模型训练
9. AI 补偿集成
10. 补偿效果可视化与 AI 技术报告
```

---

## 六、当前不建议做的事

1. 不再新增 OBS 解析模块，除非题目再次恢复 OBS 要求。
2. 不要默认向主流程注入大粗差；粗差只用于专项测试。
3. 不要用 `error_3d` 标量直接修正坐标。
4. 不要为了附加题过早改动主流程结构；先保证基础流程稳定、报告清楚、误差小于 1 m。
