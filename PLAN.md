# 北斗 SPP 系统完善计划

## Summary

将现有脚本式原型整理为标准 `beidou_spp` 包，同时保留原有 `module1_*.py` 到 `module5_*.py` 作为兼容入口。补齐 CLI、标准输出文件、Markdown 报告、pytest 测试、GUI 导出功能，并把伪距模拟、伪距修正、SPP、连续定位的字段名统一到课程验收要求。

## Key Changes

- 新建标准结构：`src/beidou_spp/`、`tests/`、`docs/`、`data/`、`outputs/`，并补齐 `README.md`、`requirements.txt`。
- 将现有功能拆分迁移：
  - `module1_nav_parser.py` → `rinex/nav_parser.py`
  - 伪距模拟 → `rinex/pseudorange_simulator.py`
  - 卫星位置/钟差 → `gnss/satellite_position.py`、`gnss/satellite_clock.py`
  - 电离层/对流层/卫星钟差伪距修正 → `gnss/corrections.py`
  - DOP → `gnss/dop.py`
  - SPP → `positioning/spp_solver.py`
  - 坐标转换 → `positioning/coordinates.py`
  - 连续定位、统计、绘图、报告 → `analysis/`
- 保留旧脚本为 thin wrapper，内部调用新包，避免老师或现有 GUI 运行旧文件时报错。

## Module Fixes

- 模块 1：
  - 输出 `parsed_nav_debug.csv` 和 `simulated_pseudorange.csv`。
  - 伪距模拟从模块 3 移到模块 1 对应的 `pseudorange_simulator.py`。
  - 支持 `--seed`，保证可复现。
  - 预处理明确输出健康筛选、高度角筛选、粗差筛选、历元对齐后的卫星数量。
- 模块 2：
  - 输出 `satellite_debug.csv` 和 `corrected_pseudorange.csv`。
  - 将 Saastamoinen 对流层、简化电离层、卫星钟差修正集中到 `corrections.py`。
  - 在关键公式旁补充变量含义、单位注释。
- 模块 3：
  - 输出标准 `spp_epoch_result.csv`，字段使用 `x_m/y_m/z_m/lat_deg/lon_deg/height_m/receiver_clock_bias_m/receiver_clock_bias_s/num_sats/GDOP/PDOP/HDOP/VDOP/TDOP/converged/iterations/message`。
  - 补齐 HDOP、VDOP、TDOP。
  - 失败历元统一记录原因，不抛出到主流程。
- 模块 4：
  - 输出标准 `positioning_results.csv`、`accuracy_report.md`。
  - 生成 `trajectory.png`、`position_error.png`、`dop_and_sat_count.png`，同时保留旧图名兼容 GUI。
  - 报告中加入卫星数量、DOP、定位误差关系分析。
- 模块 5：
  - 新增 CLI：`python -m beidou_spp.cli --nav data/sample.nav --output outputs --max-iter 10 --threshold 1e-4 --elevation-mask 15`。
  - GUI 改为优先 PySide6，兼容 PyQt6；保留 PyQt5 仅作为本机 fallback。
  - GUI 增加“导出结果和报告”按钮。
  - 生成 `docs/test_report.md` 和 `outputs/test_report.md`。
  - 新增 pytest 单元测试。

## Public Interfaces

- CLI 参数固定为：
  - `--nav`
  - `--output`
  - `--receiver-ecef x,y,z`
  - `--start`
  - `--end`
  - `--interval`
  - `--seed`
  - `--max-iter`
  - `--threshold`
  - `--elevation-mask`
- 核心入口函数：
  - `parse_nav_file(nav_path) -> dict`
  - `simulate_pseudorange(...) -> pandas.DataFrame`
  - `compute_satellite_states(...) -> pandas.DataFrame`
  - `correct_pseudorange(...) -> pandas.DataFrame`
  - `solve_epoch_spp(...) -> SppSolution`
  - `run_pipeline(config) -> PipelineResult`
- 输出目录统一为用户传入的 `--output`，默认 `outputs/`。

## Test Plan

- `tests/test_rinex_parser.py`
  - 验证能解析 `tarc0910.26b` 或复制到 `data/sample.nav` 的样例。
  - 验证至少解析出 4 颗北斗卫星，字段不为空。
- `tests/test_coordinates.py`
  - 验证 ECEF ↔ BLH 基本精度。
  - 验证高度、经纬度范围合法。
- `tests/test_dop.py`
  - 使用固定设计矩阵验证 GDOP、PDOP、HDOP、VDOP、TDOP 为有限正数。
- `tests/test_spp_solver.py`
  - 用人工构造的 4 颗以上卫星和伪距验证 SPP 收敛。
  - 验证少于 4 颗卫星时返回失败消息。
- 集成测试：
  - 使用至少 3 组 RINEX 文件：`tarc0910.26b`、`tarc1200.26b`、`tarc1250.26b`。
  - 输出 `docs/test_report.md`，记录每组数据的成功历元、RMS、最大误差、平均 DOP。

## Assumptions

- 默认接收机 ECEF 坐标继续使用当前值：`(-2267800.0, 5009340.0, 3221000.0)`。
- 默认随机种子使用 `2026`。
- 默认输出目录使用 `outputs/`，旧 `output/` 不删除。
- 旧脚本继续可运行，新包结构作为正式验收入口。
- 不引入 RTKLIB、gnsspy、georinex 等 GNSS 定位库；只使用 `numpy`、`pandas`、`matplotlib`、`PySide6/PyQt6`、`pytest`。
