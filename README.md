# 北斗 SPP 全流程定位解算系统

本项目实现一个纯 Python 的北斗 RINEX NAV 定位解算流程：

1. 解析 RINEX 导航文件，并生成模拟伪距观测值。
2. 根据广播星历计算卫星 ECEF 坐标和卫星钟差。
3. 使用卫星钟差、简化电离层模型和 Saastamoinen 对流层模型修正伪距。
4. 使用迭代最小二乘完成伪距单点定位。
5. 完成连续定位、精度分析、可视化、命令行入口、图形界面和测试。

核心 GNSS 算法均为手写实现，不调用 RTKLIB、gnsspy、georinex 等第三方 GNSS 定位库。

## 命令行运行

```powershell
python -m beidou_spp.cli --nav data/sample.nav --output outputs --max-iter 10 --threshold 1e-4 --elevation-mask 15
```

常用参数：

- `--receiver-ecef x,y,z`
- `--start YYYY-MM-DDTHH:MM:SS`
- `--end YYYY-MM-DDTHH:MM:SS`
- `--interval 秒数`
- `--seed 整数随机种子`

## 图形界面

```powershell
python rinex_gui.py
```

图形界面支持导入 NAV 文件、设置解算参数和接收机坐标、运行定位、查看结果表格与图像，并导出生成的结果文件。

## 主要输出文件

- `parsed_nav_debug.csv`
- `simulated_pseudorange.csv`
- `satellite_debug.csv`
- `corrected_pseudorange.csv`
- `spp_epoch_result.csv`
- `positioning_results.csv`
- `accuracy_report.md`
- `trajectory.png`
- `position_error.png`
- `dop_and_sat_count.png`
- `test_report.md`
