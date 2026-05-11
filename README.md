# Beidou SPP Full-Flow System

This project implements a pure-Python Beidou RINEX NAV positioning workflow:

1. RINEX navigation parsing and simulated pseudorange generation.
2. Satellite ECEF position and satellite clock calculation from broadcast ephemeris.
3. Pseudorange correction using satellite clock, simple ionosphere and Saastamoinen troposphere models.
4. Single Point Positioning using iterative least squares.
5. Continuous positioning, accuracy analysis, visualization, CLI, GUI and tests.

The core GNSS algorithms are handwritten. RTKLIB, gnsspy, georinex and similar GNSS positioning libraries are not used.

## CLI

```powershell
python -m beidou_spp.cli --nav data/sample.nav --output outputs --max-iter 10 --threshold 1e-4 --elevation-mask 15
```

Important options:

- `--receiver-ecef x,y,z`
- `--start YYYY-MM-DDTHH:MM:SS`
- `--end YYYY-MM-DDTHH:MM:SS`
- `--interval seconds`
- `--seed integer`

## GUI

```powershell
python rinex_gui.py
```

The GUI supports NAV import, solver parameters, receiver coordinates, running positioning, viewing tables/plots and exporting generated results.

## Main Outputs

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

