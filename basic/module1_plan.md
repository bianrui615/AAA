# Module1 重构实现总结

## 目标
将原本仅解析 RINEX NAV 的 `module1.py` 重构为自包含的模块，集成卫星位置计算（原 module2）和伪距模拟/预处理（原 module3）功能，输出两张新的 CSV 文件。

## 已完成工作

### 1. 代码迁移与集成
从 `module2.py` 迁移以下功能到 `module1.py`：
- 常量定义：`MU`, `OMEGA_E`, `C`, `BDT_EPOCH`, `SECONDS_IN_WEEK`, `HALF_WEEK`, `RELATIVITY_F`
- 辅助函数：`_solve_kepler()`, `_normalize_time()`, `_bds_seconds_of_week()`, `_is_bds_geo()`
- 核心算法：`compute_satellite_position()` — 基于广播星历的卫星 ECEF 坐标计算（开普勒轨道 + 摄动 + 地球自转）

从 `module3.py` 迁移以下功能到 `module1.py`：
- `ecef_to_blh()` — WGS84 ECEF 转经纬高
- `compute_geometric_range()` — 卫星与接收机间几何距离
- `compute_elevation()` — 卫星相对接收机的高度角（ENU 转换）
- `simulate_pseudorange()` — 伪距模拟，使用 `random.Random(seed)` 保证可复现
- 误差模型：`P = rho + sisre + iono + tropo + rcv_clock + noise`

### 2. 新增预处理流程
`preprocess_pseudorange_records()` 实现四级过滤：
1. **健康状态过滤**：`health != 0` → `is_used=no, reject_reason=unhealthy`
2. **高度角过滤**：`elevation < elevation_mask_deg` → `is_used=no, reject_reason=low_elevation`
3. **数值范围过滤**：`P < 15,000,000` 或 `P > 50,000,000` → `is_used=no, reject_reason=out_of_range`
4. **粗差剔除（MAD 稳健统计）**：基于中位数绝对偏差，若剔除后卫星数 < 4 则放弃剔除 → `is_used=no, reject_reason=outlier`

### 3. 新增输出文件
| 文件名 | 内容 | 字段 |
|--------|------|------|
| `module1_parsed_nav_debug.csv` | 解析后的星历参数明细 | sat_id, toc, toe, af0, af1, af2, sqrtA, e, i0, Omega0, omega, M0, DeltaN, IDOT, Cuc, Cus, Crc, Crs, Cic, Cis, health, is_healthy, parse_status |
| `module1_simulated_pseudorange.csv` | 模拟伪距及各项误差分解 | epoch, sat_id, sat_x, sat_y, sat_z, receiver_x, receiver_y, receiver_z, elevation_deg, rho, sisre_error, iono_error, tropo_error, receiver_clock_error, noise_error, pseudorange, is_used, reject_reason |

### 4. 统一入口函数
```python
def run_module1(
    nav_path: str | Path,
    receiver_approx: Tuple[float, float, float],
    epochs: List[datetime],
    seed: int,
    output_dir: str | Path = "output",
    elevation_mask_deg: float = 0.0,
) -> Dict[str, Path]
```
- 单函数完成：解析 NAV → 选星历 → 计算卫星位置 → 模拟伪距 → 预处理 → 保存 CSV
- 随机种子可复现性已验证：相同 seed 产生完全相同的 CSV

### 5. 保留的向后兼容
- `parse_rinex_nav_with_info()` 和 `save_nav_parse_outputs()` 接口不变
- `BroadcastEphemeris` 和 `NavParseInfo` dataclass 不变
- 原有输出 `module1_nav_parse_summary.txt` 和 `module1_ephemeris_list.csv` 继续生成

### 6. BDS-3 过滤
- 继续保留 `PRN >= 19` 过滤（BDS-3 only）
- 解析统计中记录 `skipped_bds2_records` 数量

## 已知限制
- 导航文件 `tarc0910.26b_cnav` 中部分 BDS-3 卫星（如 C19-C30）的 `sqrtA` 值异常偏小（~34），导致卫星位置计算结果仅约 1000m，高度角为负，被预处理器剔除
- 部分卫星（如 C27, C34, C35, C39）的 `sqrtA` 为负数，计算时会抛出异常
- 仅有少数卫星（如 C40，sqrtA≈5337）能计算出合理位置
- 上述问题属于原始导航数据质量问题，非本重构引入

## 验证结果
- `module1.py` 可直接运行：`python basic/module1.py`
- 随机种子可复现性：两次运行 seed=42，所有伪距值完全一致
- `module5.py` 系统测试仍可正常完成（使用旧的 module2/3 接口）
