# Experiment Report

Default receiver ECEF coordinate:

`(-2267800.0, 5009340.0, 3221000.0)`

Default sample file:

`data/sample.nav`

Run command:

```powershell
python -m beidou_spp.cli --nav data/sample.nav --output outputs --max-iter 10 --threshold 1e-4 --elevation-mask 15
```

Main results are written to `outputs/`, including positioning CSV files,
accuracy report and three figures.

