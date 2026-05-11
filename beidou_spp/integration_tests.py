"""用于生成 docs/test_report.md 的集成测试场景运行器。"""

from __future__ import annotations

import math
from pathlib import Path

from .analysis.report import write_test_report
from .config import PipelineConfig
from .pipeline import run_pipeline


def run_default_scenarios() -> Path:
    rows = []
    for nav in [Path("tarc0910.26b"), Path("tarc1200.26b"), Path("tarc1250.26b")]:
        if not nav.exists():
            continue
        output = Path("outputs") / nav.stem
        result = run_pipeline(PipelineConfig(nav=nav, output=output, elevation_mask=0.0))
        gdops = [
            float(row.get("GDOP", math.nan))
            for row in result.epoch_results
            if math.isfinite(float(row.get("GDOP", math.nan)))
        ]
        rows.append(
            {
                "nav_file": str(nav),
                "success_epochs": result.success_epochs,
                "total_epochs": result.total_epochs,
                "rms_error_m": result.rms_error_m,
                "max_error_m": result.max_error_m,
                "mean_gdop": sum(gdops) / len(gdops) if gdops else math.nan,
            }
        )
    return write_test_report(rows, Path("docs") / "test_report.md")


if __name__ == "__main__":
    print(run_default_scenarios())
