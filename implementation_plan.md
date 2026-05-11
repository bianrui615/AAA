# 北斗定位解算全流程软件系统 — 实施计划

## 一、现状评估

当前代码库已具备功能完备的核心算法实现，测试可达到 <1m RMS 精度：

| 模块 | 现有文件 | 当前状态 | 与目标差距 |
|------|---------|---------|-----------|
| 模块1 | `module1_nav_parser.py` | NAV解析完成，输出星历CSV+摘要 | 缺：模拟伪距生成、可见星筛选、`simulated_obs.csv`、`available_sats.csv` |
| 模块2 | `module2_satellite_position_clock.py` | 卫星位置/钟差计算完成 | 缺：显式摄动分量输出(δu,δr,δi)、延迟修正CSV、`corrected_pseudorange.csv` |
| 模块3 | `module3_spp_solver.py` | 迭代最小二乘+SPP完成 | 缺：逐迭代详细日志(`iteration_log.txt`)、HDOP/VDOP/TDOP、`omc_residuals.csv` |
| 模块4 | `module4_continuous_analysis.py` | 连续定位+3张图+统计完成 | 缺：ENU误差分解、ENU方向误差曲线图、分方向统计 |
| 模块5 | `module5_main_system_test.py` + `rinex_gui.py` | 硬编码参数Pipeline+GUI | 缺：CLI参数解析、`README.md`、多场景测试脚本 |

**核心策略**：保留并复用全部已验证的核心算法（NAV解析、开普勒轨道、迭代最小二乘、伪距模型），按用户要求的模块边界和输出格式进行重构与补充。

---

## 二、实施阶段

### Phase 1: 模块一重构 — NAV解析 + 模拟伪距生成 + 可见星筛选

**目标**：让模块一成为"数据入口"，输出原始模拟观测数据。

**当前问题**：伪距生成目前在模块三，需前移至模块一。

**实现方案**：
1. 在 `module1_nav_parser.py` 中新增函数：
   - `compute_approximate_sat_positions(nav_data, epoch)` — 基于星历计算概略位置（复用模块二算法，但简化为不输出摄动分量）
   - `filter_visible_satellites(sat_positions, receiver_pos, mask_deg=15.0)` — 计算高度角并筛选
   - `generate_simulated_observations(nav_data, receiver_pos, duration, interval, seed)` — 按误差模型生成伪距
2. 新增输出文件：
   - `nav_params.csv` — 保留现有 `module1_ephemeris_list.csv` 内容，扩展为包含全部所需星历参数字段
   - `simulated_obs.csv` — 列：`epoch, prn, rho, sisre, iono, trop, rcv_clock, noise, pseudorange, elevation`
   - `available_sats.csv` — 列：`epoch, prn, elevation_deg`
3. 接口函数签名按用户要求：
   - `parse_nav(file_path) -> dict[str, list[BroadcastEphemeris]]`
   - `generate_simulated_obs(nav_data, true_pos, duration, interval) -> pd.DataFrame`

**关键技术决策**：模块一计算卫星概略位置时，直接调用模块二的核心算法（`calculate_satellite_position_clock`），但通过参数控制不输出摄动细节。模块二在独立运行时会重新计算并输出完整摄动分量。这避免了代码重复，同时满足"模块一输出概略位置、模块二输出精确位置+摄动修正"的教学展示需求。

### Phase 2: 模块二重构 — 卫星位置/钟差 + 显式传播延迟修正

**目标**：让模块二成为"修正模块"，输出详细的修正分量和修正后伪距。

**当前问题**：延迟修正（Saastamoinen对流层、简化电离层）目前内嵌在模块三的伪距生成函数中，需抽离到模块二。

**实现方案**：
1. 从 `module3_spp_solver.py` 中抽离延迟模型函数到 `module2_satellite_position_clock.py`：
   - `simple_ionosphere_delay(sat_pos, rcv_pos) -> float`
   - `saastamoinen_troposphere_delay(sat_pos, rcv_pos) -> (dry, wet, total)` — 需拆分为干/湿分量
2. 新增函数：
   - `correct_pseudorange(raw_pseudorange, sat_clock_bias_m, iono_m, trop_dry_m, trop_wet_m) -> corrected_pseudorange`
3. 扩展 `SatelliteState` dataclass，增加 `delta_u`, `delta_r`, `delta_i` 字段
4. 新增输出文件：
   - `sat_position.csv` — 扩展现有 `module2_satellite_position_clock.csv`，增加 `delta_u, delta_r, delta_i` 列
   - `sat_clock.csv` — 新增：`epoch, prn, polynomial_corr, relativity_corr, total_clock_bias`
   - `delay_correction.csv` — 新增：`epoch, prn, trop_dry, trop_wet, iono, total_delay`
   - `corrected_pseudorange.csv` — 新增：`epoch, prn, P_raw, P_corrected`
5. 接口函数签名按用户要求：
   - `calc_sat_position(nav_params, epoch, prn) -> (X,Y,Z, corrections)`
   - `calc_sat_clock(nav_params, epoch, prn) -> bias`
   - `calc_trop_delay(lat, h, el) -> delay`
   - `calc_iono_delay(el, az, lat, lon, epoch) -> delay`

**关键技术决策**：模块一生成的 `simulated_obs.csv` 中的 `pseudorange` 字段即为 `P_raw`（含误差但未做系统修正）。模块二读取 `simulated_obs.csv` 和 `nav_params.csv`，计算每颗卫星的精确位置、钟差、传播延迟，输出 `P_corrected = P_raw - (sat_clock_bias_m + iono_delay + trop_delay)`。这里的 `iono_delay` 和 `trop_delay` 是模型计算值（非模块一中用于生成伪距的随机实现值），因此修正后会残留随机误差，这正是SPP的实际工作原理。

### Phase 3: 模块三增强 — 详细迭代日志 + 完整DOP + OMC残差

**目标**：让模块三的定位过程完全可追溯、可验收。

**当前问题**：迭代过程仅保存在内存中，无逐迭代日志文件；DOP仅有PDOP/GDOP。

**实现方案**：
1. 重构 `solve_spp` 函数，增加迭代过程记录：
   - 每次迭代记录：残差向量范数 `||v||`、参数改正量 `(dX, dY, dZ, dt)`、当前位置估计
   - 收敛时记录最终设计矩阵条件数
2. 新增HDOP/VDOP/TDOP计算：
   - 在 `_compute_dops` 中增加：将 `(G^T G)^{-1}` 转换到ENU坐标系后提取各方向方差
   - `HDOP = sqrt(q_EE + q_NN)`, `VDOP = sqrt(q_UU)`, `TDOP = sqrt(q_tt)`
3. 新增输出文件：
   - `iteration_log.txt` — 每历元详细迭代日志（必须可独立验收）
   - `dop_values.csv` — 列：`epoch, n_sats, GDOP, PDOP, HDOP, VDOP, TDOP`
   - `positioning_result.csv` — 列：`epoch, X, Y, Z, B, L, H, dt, residual_rms`
   - `omc_residuals.csv` — 列：`epoch, prn, O_minus_C`
4. 接口函数签名按用户要求：
   - `least_squares_positioning(sat_pos, pseudorange, initial_pos, threshold, max_iter) -> (X,Y,Z,dt,log)`
   - `ecef_to_blh(X,Y,Z) -> (B,L,H)` — 已有，保持
   - `calc_dop(G) -> (GDOP,PDOP,HDOP,VDOP,TDOP)` — 扩展

**关键技术决策**：迭代日志使用字符串列表在 `solve_spp` 内部累积，收敛或失败后统一写入文件。HDOP/VDOP计算需要先将协因数矩阵从ECEF转换到ENU（使用接收机概略位置的BLH构建旋转矩阵），这是标准GNSS算法，需手写实现。

### Phase 4: 模块四增强 — ENU误差分解 + ENU误差曲线

**目标**：增加ENU局部分析能力，替换现有纯3D分析。

**当前问题**：仅有3D误差，无ENU方向分解。

**实现方案**：
1. 新增 `xyz_to_enu(xyz_err, blh) -> (dE, dN, dU)` 函数：
   - 基于接收机位置的BLH构建ECEF→ENU旋转矩阵
   - 公式来源标注：武汉大学《GPS测量与数据处理》
2. 修改连续定位循环，对每个成功历元计算ENU误差
3. 修改统计函数，输出分方向统计：
   - RMS/Mean/Max for dE, dN, dU, d3D
4. 修改/新增可视化：
   - 图1：ENU三方向误差曲线（替换或增加现有error_curve.png）
   - 图2：经纬度轨迹散点图（保留现有）
   - 图3：卫星可见数与DOP双轴曲线（保留现有）
5. 新增/修改输出文件：
   - `error_analysis.csv` — 列：`epoch, dE, dN, dU, d3D`
   - `accuracy_stats.txt` — 包含分方向统计表
   - `fig_error_curve.png`, `fig_trajectory.png`, `fig_dop.png`

**关键技术决策**：保留现有3张图的同时，将误差曲线图改为ENU三方向子图（1行3列或3行1列），更便于分析各方向误差特性。保留原有 `module4_*.png` 命名以避免GUI兼容性问题，同时新增 `fig_error_curve.png` 等按用户命名规范的图表。

### Phase 5: 模块五整合 — CLI命令行 + README + 测试脚本

**目标**：将硬编码Pipeline改造为可配置、可测试的完整系统。

**当前问题**：参数全部硬编码在 `module5_main_system_test.py` 中。

**实现方案**：
1. 新建 `main.py` 作为命令行入口：
   - 使用 `argparse` 支持：`--nav`, `--true-pos`, `--duration`, `--interval`, `--max-iter`, `--threshold`, `--elevation-mask`, `--output-dir`, `--seed`
   - 默认参数与现有 `module5_main_system_test.py` 保持一致
2. 保留 `module5_main_system_test.py` 作为向后兼容的入口（或将其内容迁移到 `main.py`）
3. 更新 `rinex_gui.py` 以调用新的模块接口
4. 编写 `README.md`：
   - 环境依赖与安装步骤
   - 命令行使用示例
   - 模块说明
   - 文件输出说明
5. 编写测试脚本 `test_scenarios.py`：
   - 场景1（开阔）：默认参数，全部健康卫星
   - 场景2（遮挡）：`elevation_mask_deg=15.0`，模拟部分卫星被遮挡
   - 场景3（多路径/噪声增强）：增大观测噪声标准差，模拟恶劣环境
   - 每个场景运行后输出对比统计

**关键技术决策**：`main.py` 将成为新的主要入口，`module5_main_system_test.py` 可保留为内部测试文件或重命名为 `module5_system_integration.py`。CLI设计遵循用户要求的 `--nav xxx.nav --true-pos x,y,z` 风格。

---

## 三、文件变更清单

### 修改文件
| 文件 | 变更内容 |
|------|---------|
| `module1_nav_parser.py` | 新增伪距生成、可见星筛选函数；扩展输出文件；增加pandas依赖使用 |
| `module2_satellite_position_clock.py` | 抽入延迟模型；扩展SatelliteState；新增4个输出文件 |
| `module3_spp_solver.py` | 增加迭代日志、HDOP/VDOP/TDOP、OMC残差输出；抽离延迟模型到模块二 |
| `module4_continuous_analysis.py` | 增加ENU转换、ENU统计、ENU误差曲线 |
| `module5_main_system_test.py` | 调整Pipeline以匹配新模块接口；保持向后兼容 |
| `rinex_gui.py` | 适配新输出文件路径 |
| `requirements.txt` | 增加 `pandas` |

### 新增文件
| 文件 | 说明 |
|------|------|
| `main.py` | CLI主入口 |
| `README.md` | 项目说明文档 |
| `test_scenarios.py` | 三组场景测试脚本 |

---

## 四、验证检查点

按用户验收要求，每阶段完成后验证：

- [ ] 模块1：`nav_params.csv` + `simulated_obs.csv`（含误差分量明细）+ `available_sats.csv`
- [ ] 模块2：`sat_position.csv`（含δu,δr,δi）+ `sat_clock.csv` + `delay_correction.csv` + `corrected_pseudorange.csv`
- [ ] 模块3：`iteration_log.txt`（含逐次迭代残差）+ `dop_values.csv`（含HDOP,VDOP,TDOP）+ `positioning_result.csv`（含BLH）+ `omc_residuals.csv`
- [ ] 模块4：`error_analysis.csv`（含dE,dN,dU）+ `accuracy_stats.txt`（分方向统计）+ 3张图
- [ ] 模块5：`python main.py --nav xxx.nav --true-pos x,y,z` 一键运行

---

## 五、风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| 模块边界重构引入回归错误 | 每阶段完成后运行完整Pipeline，与现有输出比对精度指标 |
| HDOP/VDOP/TDOP计算复杂 | 使用已知参考点验证：通过构造特定几何构型（如正上方卫星）验证DOP值合理性 |
| ENU旋转矩阵实现错误 | 用单位向量验证：将(1,0,0)ECEF误差转换到ENU，应在东方向得到非零值 |
| CLI参数解析与GUI冲突 | CLI和GUI使用同一套默认配置字典，保持参数语义一致 |

---

## 六、推荐方案

**单一推荐方案**：基于现有代码进行"重构+增强"。

理由：
1. 现有核心算法（开普勒轨道、迭代最小二乘、伪距模型、对流层/电离层模型）已经过测试，可达到 <1m RMS 精度
2. 用户的要求主要是"模块边界重组"和"输出文件规范化"，而非算法重新实现
3. 完全重写会引入不必要的风险，且浪费已验证的代码资产
4. 重构方案可在保持精度的同时，快速达到课程验收要求

如果用户对模块边界有严格到不能复用现有算法的程度的要求，可在实施中调整。但基于现有代码重构是最高效、最可靠的路径。
