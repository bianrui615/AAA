# Design Report

The project is organized around five required modules:

1. RINEX NAV parsing and pseudorange simulation.
2. Satellite position, clock and propagation corrections.
3. Single-epoch SPP least-squares positioning.
4. Continuous positioning, analysis and visualization.
5. CLI, GUI and tests.

Core GNSS formulas are implemented in Python with `numpy` used only for matrix
operations. No RTKLIB, gnsspy, georinex or other GNSS positioning library is
used.

