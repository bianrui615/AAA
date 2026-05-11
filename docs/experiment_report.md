# 实验报告

默认接收机 ECEF 坐标：

`(-2267800.0, 5009340.0, 3221000.0)`

默认样例文件：

`data/sample.nav`

运行命令：

```powershell
python -m beidou_spp.cli --nav data/sample.nav --output outputs --max-iter 10 --threshold 1e-4 --elevation-mask 15
```

主要结果写入 `outputs/`，包括定位 CSV、精度报告和三张结果图。
