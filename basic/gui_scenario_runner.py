"""
gui_scenario_runner.py

Three-scenario GUI batch runner for BDS CNAV continuous positioning.

This file is intentionally independent from module5.py and rinex_gui.py. It
reuses the public positioning pipeline from module1/module4, but keeps all GUI
batch configuration, ENU trajectory handling, report generation, and threading
logic local to this entry point.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/basic") / "matplotlib_cache"))

try:
    from PyQt6.QtCore import QDate, QDateTime, QThread, QTime, Qt, pyqtSignal as Signal
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QDateTimeEdit,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    QT_BINDING = "PyQt6"
    QT_IMPORT_ERROR: Optional[Exception] = None
except ModuleNotFoundError as pyqt6_error:
    try:
        from PySide6.QtCore import QDate, QDateTime, QThread, QTime, Qt, Signal
        from PySide6.QtWidgets import (
            QApplication,
            QComboBox,
            QDateTimeEdit,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QProgressBar,
            QSpinBox,
            QSplitter,
            QStackedWidget,
            QTableWidget,
            QTableWidgetItem,
            QTabWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
        QT_BINDING = "PySide6"
        QT_IMPORT_ERROR = None
    except ModuleNotFoundError as pyside6_error:
        try:
            from PyQt5.QtCore import QDate, QDateTime, QThread, QTime, QTimer, Qt, pyqtSignal as Signal
            from PyQt5.QtWidgets import (
                QApplication,
                QComboBox,
                QDateTimeEdit,
                QDoubleSpinBox,
                QFileDialog,
                QFormLayout,
                QGridLayout,
                QGroupBox,
                QHBoxLayout,
                QHeaderView,
                QLabel,
                QLineEdit,
                QMainWindow,
                QMessageBox,
                QPushButton,
                QProgressBar,
                QSpinBox,
                QSplitter,
                QStackedWidget,
                QTableWidget,
                QTableWidgetItem,
                QTabWidget,
                QTextEdit,
                QVBoxLayout,
                QWidget,
            )
            QT_BINDING = "PyQt5"
            QT_IMPORT_ERROR = None
        except ModuleNotFoundError:
            QT_BINDING = ""
            QT_IMPORT_ERROR = pyqt6_error or pyside6_error

if QT_IMPORT_ERROR is None:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    from basic.module1 import parse_nav_file
    from basic.module3 import ECEF
    from basic.module4 import AnalysisSummary, calculate_summary, run_continuous_positioning


OUTPUT_ROOT = Path("outputs/basic/gui_scenario_runner")


def _qt_value(name: str, namespace: str | None = None) -> Any:
    if namespace and hasattr(Qt, namespace):
        member = getattr(getattr(Qt, namespace), name, None)
        if member is not None:
            return member
    return getattr(Qt, name)


def _header_resize_mode(name: str) -> Any:
    resize_mode = getattr(QHeaderView, "ResizeMode", None)
    if resize_mode is not None:
        return getattr(resize_mode, name)
    return getattr(QHeaderView, name)


def _qdatetime_to_datetime(value: Any) -> datetime:
    return datetime(
        value.date().year(),
        value.date().month(),
        value.date().day(),
        value.time().hour(),
        value.time().minute(),
        value.time().second(),
    )


def _datetime_to_qdatetime(value: datetime) -> Any:
    return QDateTime(
        QDate(value.year, value.month, value.day),
        QTime(value.hour, value.minute, value.second),
    )


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NaN"
    if not math.isfinite(numeric):
        return "NaN"
    return f"{numeric:.{digits}f}"


def blh_to_ecef(lat_deg: float, lon_deg: float, height_m: float) -> ECEF:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    semi_major_axis = 6378137.0
    flattening = 1.0 / 298.257223563
    eccentricity_squared = flattening * (2.0 - flattening)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    prime_vertical_radius = semi_major_axis / math.sqrt(
        1.0 - eccentricity_squared * sin_lat * sin_lat
    )
    x = (prime_vertical_radius + height_m) * cos_lat * math.cos(lon)
    y = (prime_vertical_radius + height_m) * cos_lat * math.sin(lon)
    z = (prime_vertical_radius * (1.0 - eccentricity_squared) + height_m) * sin_lat
    return x, y, z


def enu_to_ecef_delta(lat_deg: float, lon_deg: float, east: float, north: float, up: float) -> ECEF:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    dx = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
    dy = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
    dz = cos_lat * north + sin_lat * up
    return dx, dy, dz


def interpolate_enu_points(
    points: List[Tuple[float, float, float, float]],
    time_offset_s: float,
) -> Tuple[float, float, float]:
    if not points:
        return 0.0, 0.0, 0.0
    ordered = sorted(points, key=lambda row: row[0])
    if time_offset_s <= ordered[0][0]:
        return ordered[0][1], ordered[0][2], ordered[0][3]
    if time_offset_s >= ordered[-1][0]:
        return ordered[-1][1], ordered[-1][2], ordered[-1][3]

    for left, right in zip(ordered, ordered[1:]):
        t0, e0, n0, u0 = left
        t1, e1, n1, u1 = right
        if t0 <= time_offset_s <= t1:
            ratio = 0.0 if abs(t1 - t0) < 1e-12 else (time_offset_s - t0) / (t1 - t0)
            return (
                e0 + ratio * (e1 - e0),
                n0 + ratio * (n1 - n0),
                u0 + ratio * (u1 - u0),
            )
    return ordered[-1][1], ordered[-1][2], ordered[-1][3]


def build_linear_enu_trajectory(
    start_time: datetime,
    initial_ecef: ECEF,
    lat_deg: float,
    lon_deg: float,
    velocity_enu_mps: Tuple[float, float, float],
) -> Callable[[datetime], ECEF]:
    def trajectory(epoch_time: datetime) -> ECEF:
        dt = (epoch_time - start_time).total_seconds()
        de = velocity_enu_mps[0] * dt
        dn = velocity_enu_mps[1] * dt
        du = velocity_enu_mps[2] * dt
        dx, dy, dz = enu_to_ecef_delta(lat_deg, lon_deg, de, dn, du)
        return initial_ecef[0] + dx, initial_ecef[1] + dy, initial_ecef[2] + dz

    return trajectory


def build_polyline_enu_trajectory(
    start_time: datetime,
    initial_ecef: ECEF,
    lat_deg: float,
    lon_deg: float,
    trajectory_points: List[Tuple[float, float, float, float]],
) -> Callable[[datetime], ECEF]:
    def trajectory(epoch_time: datetime) -> ECEF:
        dt = (epoch_time - start_time).total_seconds()
        de, dn, du = interpolate_enu_points(trajectory_points, dt)
        dx, dy, dz = enu_to_ecef_delta(lat_deg, lon_deg, de, dn, du)
        return initial_ecef[0] + dx, initial_ecef[1] + dy, initial_ecef[2] + dz

    return trajectory


@dataclass
class ScenarioSettings:
    name: str
    rinex_file: str
    start_time: datetime
    end_time: datetime
    interval_seconds: int
    receiver_lat: float
    receiver_lon: float
    receiver_height: float
    trajectory_mode: str
    velocity_east_mps: float
    velocity_north_mps: float
    velocity_up_mps: float
    trajectory_points: List[Tuple[float, float, float, float]]
    random_seed: int
    elevation_mask_deg: float
    max_iter: int
    convergence_threshold: float


@dataclass
class ScenarioRunResult:
    settings: ScenarioSettings
    output_dir: Path
    status: str
    failure_reason: str
    results: List[dict]
    summary: AnalysisSummary


def failed_summary(failure_reason: str) -> AnalysisSummary:
    return AnalysisSummary(
        total_epochs=0,
        success_epochs=0,
        failed_epochs=0,
        average_satellite_count=math.nan,
        average_pdop=math.nan,
        average_gdop=math.nan,
        mean_error_3d=math.nan,
        rms_error_3d=math.nan,
        max_error_3d=math.nan,
        min_error_3d=math.nan,
        success_rate=0.0,
        evaluation=f"场景运行失败：{failure_reason}",
    )


def write_summary_csv(run_results: List[ScenarioRunResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario_name",
        "rinex_file",
        "trajectory_mode",
        "start_time",
        "end_time",
        "interval_seconds",
        "receiver_lat",
        "receiver_lon",
        "receiver_height",
        "total_epochs",
        "success_epochs",
        "failed_epochs",
        "success_rate",
        "average_satellite_count",
        "average_pdop",
        "average_gdop",
        "mean_error_3d",
        "rms_error_3d",
        "max_error_3d",
        "output_dir",
        "status",
        "failure_reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in run_results:
            s = item.settings
            summary = item.summary
            writer.writerow(
                {
                    "scenario_name": s.name,
                    "rinex_file": s.rinex_file,
                    "trajectory_mode": s.trajectory_mode,
                    "start_time": s.start_time.isoformat(sep=" "),
                    "end_time": s.end_time.isoformat(sep=" "),
                    "interval_seconds": s.interval_seconds,
                    "receiver_lat": s.receiver_lat,
                    "receiver_lon": s.receiver_lon,
                    "receiver_height": s.receiver_height,
                    "total_epochs": summary.total_epochs,
                    "success_epochs": summary.success_epochs,
                    "failed_epochs": summary.failed_epochs,
                    "success_rate": summary.success_rate,
                    "average_satellite_count": summary.average_satellite_count,
                    "average_pdop": summary.average_pdop,
                    "average_gdop": summary.average_gdop,
                    "mean_error_3d": summary.mean_error_3d,
                    "rms_error_3d": summary.rms_error_3d,
                    "max_error_3d": summary.max_error_3d,
                    "output_dir": str(item.output_dir),
                    "status": item.status,
                    "failure_reason": item.failure_reason,
                }
            )


def write_text_report(run_results: List[ScenarioRunResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    successes = [
        item
        for item in run_results
        if item.status == "成功" and math.isfinite(item.summary.mean_error_3d)
    ]
    min_error = min(successes, key=lambda item: item.summary.mean_error_3d) if successes else None
    max_error = max(successes, key=lambda item: item.summary.mean_error_3d) if successes else None

    with report_path.open("w", encoding="utf-8-sig") as file:
        file.write("三场景 GUI 批量连续定位评估报告\n")
        file.write("=" * 52 + "\n")
        file.write("测试目的：通过 GUI 分别配置三组 RINEX CNAV 数据、接收机位置和轨迹模式，\n")
        file.write("依次运行连续定位解算，并自动汇总成功率、误差、DOP 和轨迹跟踪表现。\n\n")

        for index, item in enumerate(run_results, 1):
            s = item.settings
            summary = item.summary
            file.write(f"场景 {index}：{s.name}\n")
            file.write("-" * 44 + "\n")
            file.write(f"RINEX 文件路径：{s.rinex_file}\n")
            file.write(f"起止时间：{s.start_time.isoformat(sep=' ')} 至 {s.end_time.isoformat(sep=' ')}\n")
            file.write(
                "接收机经纬高："
                f"lat={s.receiver_lat:.8f} deg, lon={s.receiver_lon:.8f} deg, "
                f"height={s.receiver_height:.3f} m\n"
            )
            file.write(f"轨迹模式：{s.trajectory_mode}\n")
            if s.trajectory_mode == "静态接收机":
                file.write("静态场景说明：所有历元真实接收机 ECEF 坐标保持不变。\n")
            elif s.trajectory_mode == "动态直线运动":
                file.write(
                    "动态直线速度："
                    f"E={s.velocity_east_mps:.4f} m/s, "
                    f"N={s.velocity_north_mps:.4f} m/s, "
                    f"U={s.velocity_up_mps:.4f} m/s\n"
                )
                file.write("跟踪效果说明：真实轨迹按 ENU 匀速位移转换为 ECEF，解算轨迹随历元连续跟踪。\n")
            elif s.trajectory_mode == "动态折线轨迹":
                file.write("动态折线控制点：(time_offset_s, dE_m, dN_m, dU_m)\n")
                for point in s.trajectory_points:
                    file.write(f"  {point}\n")
                file.write("跟踪效果说明：真实轨迹按控制点线性插值，再转换为 ECEF，与解算轨迹对比输出。\n")

            file.write(f"总历元数：{summary.total_epochs}\n")
            file.write(f"成功历元数：{summary.success_epochs}\n")
            file.write(f"失败历元数：{summary.failed_epochs}\n")
            file.write(f"成功率：{summary.success_rate * 100.0:.2f}%\n")
            file.write(f"平均误差：{_format_float(summary.mean_error_3d)} m\n")
            file.write(f"RMS 误差：{_format_float(summary.rms_error_3d)} m\n")
            file.write(f"最大误差：{_format_float(summary.max_error_3d)} m\n")
            file.write(f"平均 PDOP：{_format_float(summary.average_pdop)}\n")
            file.write(f"平均 GDOP：{_format_float(summary.average_gdop)}\n")
            file.write(f"输出目录：{item.output_dir}\n")
            if item.failure_reason:
                file.write(f"失败原因：{item.failure_reason}\n")
            file.write("\n")

        file.write("三场景结果对比\n")
        file.write("-" * 44 + "\n")
        for item in run_results:
            file.write(
                f"{item.settings.name}: 状态={item.status}, "
                f"成功率={item.summary.success_rate * 100.0:.2f}%, "
                f"平均误差={_format_float(item.summary.mean_error_3d)} m, "
                f"RMS={_format_float(item.summary.rms_error_3d)} m, "
                f"最大误差={_format_float(item.summary.max_error_3d)} m\n"
            )
        if min_error is not None:
            file.write(f"\n误差最小场景：{min_error.settings.name}，平均误差 {_format_float(min_error.summary.mean_error_3d)} m\n")
        if max_error is not None:
            file.write(f"误差最大场景：{max_error.settings.name}，平均误差 {_format_float(max_error.summary.mean_error_3d)} m\n")

        failed = [item for item in run_results if item.failure_reason]
        if failed:
            file.write("\n失败场景记录：\n")
            for item in failed:
                file.write(f"{item.settings.name}: {item.failure_reason}\n")
        else:
            file.write("\n三个场景均完成连续定位解算并生成评估图表。\n")


if QT_IMPORT_ERROR is None:

    class MplCanvas(FigureCanvas):
        def __init__(self, width: float = 5.0, height: float = 4.0) -> None:
            self.figure = Figure(figsize=(width, height), tight_layout=True)
            super().__init__(self.figure)


    class ScenarioConfigPage(QWidget):
        def __init__(self, index: int, defaults: Dict[str, Any]) -> None:
            super().__init__()
            self.index = index
            self._build_ui()
            self.apply_defaults(defaults)

        def _build_ui(self) -> None:
            layout = QVBoxLayout(self)

            file_group = QGroupBox("RINEX 文件")
            file_layout = QHBoxLayout(file_group)
            self.path_edit = QLineEdit()
            browse_button = QPushButton("选择 RINEX 文件")
            browse_button.clicked.connect(self.browse_file)
            file_layout.addWidget(self.path_edit)
            file_layout.addWidget(browse_button)
            layout.addWidget(file_group)

            time_group = QGroupBox("解算时间")
            time_layout = QFormLayout(time_group)
            self.start_edit = QDateTimeEdit()
            self.start_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
            self.start_edit.setCalendarPopup(True)
            self.end_edit = QDateTimeEdit()
            self.end_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
            self.end_edit.setCalendarPopup(True)
            self.interval_spin = QSpinBox()
            self.interval_spin.setRange(1, 86400)
            self.interval_spin.setSuffix(" s")
            time_layout.addRow("start_time", self.start_edit)
            time_layout.addRow("end_time", self.end_edit)
            time_layout.addRow("interval_seconds", self.interval_spin)
            layout.addWidget(time_group)

            receiver_group = QGroupBox("接收机经纬高")
            receiver_layout = QFormLayout(receiver_group)
            self.lat_spin = QDoubleSpinBox()
            self.lat_spin.setRange(-90.0, 90.0)
            self.lat_spin.setDecimals(8)
            self.lon_spin = QDoubleSpinBox()
            self.lon_spin.setRange(-180.0, 180.0)
            self.lon_spin.setDecimals(8)
            self.height_spin = QDoubleSpinBox()
            self.height_spin.setRange(-1000.0, 10000.0)
            self.height_spin.setDecimals(3)
            receiver_layout.addRow("latitude_deg", self.lat_spin)
            receiver_layout.addRow("longitude_deg", self.lon_spin)
            receiver_layout.addRow("height_m", self.height_spin)
            layout.addWidget(receiver_group)

            params_group = QGroupBox("解算参数")
            params_layout = QFormLayout(params_group)
            self.seed_spin = QSpinBox()
            self.seed_spin.setRange(0, 999999999)
            self.elevation_spin = QDoubleSpinBox()
            self.elevation_spin.setRange(0.0, 90.0)
            self.elevation_spin.setDecimals(1)
            self.max_iter_spin = QSpinBox()
            self.max_iter_spin.setRange(1, 100)
            self.threshold_spin = QDoubleSpinBox()
            self.threshold_spin.setRange(1e-9, 1000.0)
            self.threshold_spin.setDecimals(9)
            self.threshold_spin.setSingleStep(0.0001)
            params_layout.addRow("random_seed", self.seed_spin)
            params_layout.addRow("elevation_mask_deg", self.elevation_spin)
            params_layout.addRow("max_iter", self.max_iter_spin)
            params_layout.addRow("convergence_threshold", self.threshold_spin)
            layout.addWidget(params_group)

            trajectory_group = QGroupBox("轨迹模式")
            trajectory_layout = QVBoxLayout(trajectory_group)
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["静态接收机", "动态直线运动", "动态折线轨迹"])
            self.mode_combo.currentIndexChanged.connect(self._update_mode_stack)
            trajectory_layout.addWidget(self.mode_combo)

            self.mode_stack = QStackedWidget()
            static_page = QWidget()
            static_layout = QVBoxLayout(static_page)
            static_layout.addWidget(QLabel("静态模式：真实接收机位置在所有历元保持不变。"))

            linear_page = QWidget()
            linear_layout = QFormLayout(linear_page)
            self.ve_spin = QDoubleSpinBox()
            self.ve_spin.setRange(-1000.0, 1000.0)
            self.ve_spin.setDecimals(4)
            self.vn_spin = QDoubleSpinBox()
            self.vn_spin.setRange(-1000.0, 1000.0)
            self.vn_spin.setDecimals(4)
            self.vu_spin = QDoubleSpinBox()
            self.vu_spin.setRange(-1000.0, 1000.0)
            self.vu_spin.setDecimals(4)
            linear_layout.addRow("velocity_east_mps", self.ve_spin)
            linear_layout.addRow("velocity_north_mps", self.vn_spin)
            linear_layout.addRow("velocity_up_mps", self.vu_spin)

            polyline_page = QWidget()
            polyline_layout = QVBoxLayout(polyline_page)
            button_row = QHBoxLayout()
            add_button = QPushButton("添加轨迹点")
            remove_button = QPushButton("删除选中点")
            add_button.clicked.connect(self.add_trajectory_row)
            remove_button.clicked.connect(self.remove_selected_trajectory_row)
            button_row.addWidget(add_button)
            button_row.addWidget(remove_button)
            button_row.addStretch(1)
            self.trajectory_table = QTableWidget(0, 4)
            self.trajectory_table.setHorizontalHeaderLabels(["time_offset_s", "dE_m", "dN_m", "dU_m"])
            self.trajectory_table.horizontalHeader().setSectionResizeMode(_header_resize_mode("Stretch"))
            polyline_layout.addLayout(button_row)
            polyline_layout.addWidget(self.trajectory_table)

            self.mode_stack.addWidget(static_page)
            self.mode_stack.addWidget(linear_page)
            self.mode_stack.addWidget(polyline_page)
            trajectory_layout.addWidget(self.mode_stack)
            layout.addWidget(trajectory_group)
            layout.addStretch(1)

        def apply_defaults(self, defaults: Dict[str, Any]) -> None:
            self.path_edit.setText(defaults["rinex_file"])
            self.start_edit.setDateTime(_datetime_to_qdatetime(defaults["start_time"]))
            self.end_edit.setDateTime(_datetime_to_qdatetime(defaults["end_time"]))
            self.interval_spin.setValue(defaults["interval_seconds"])
            self.lat_spin.setValue(defaults["receiver_lat"])
            self.lon_spin.setValue(defaults["receiver_lon"])
            self.height_spin.setValue(defaults["receiver_height"])
            self.seed_spin.setValue(defaults["random_seed"])
            self.elevation_spin.setValue(defaults["elevation_mask_deg"])
            self.max_iter_spin.setValue(defaults["max_iter"])
            self.threshold_spin.setValue(defaults["convergence_threshold"])
            self.ve_spin.setValue(defaults.get("velocity_east_mps", 0.0))
            self.vn_spin.setValue(defaults.get("velocity_north_mps", 0.0))
            self.vu_spin.setValue(defaults.get("velocity_up_mps", 0.0))
            mode_index = self.mode_combo.findText(defaults["trajectory_mode"])
            self.mode_combo.setCurrentIndex(max(mode_index, 0))
            self.set_trajectory_points(defaults.get("trajectory_points", []))
            self._update_mode_stack()

        def browse_file(self) -> None:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "选择 RINEX CNAV 文件",
                str(Path("nav").resolve()),
                "RINEX CNAV (*.26b_cnav *.cnav *.*)",
            )
            if file_name:
                self.path_edit.setText(file_name)

        def _update_mode_stack(self) -> None:
            self.mode_stack.setCurrentIndex(self.mode_combo.currentIndex())

        def add_trajectory_row(self) -> None:
            row = self.trajectory_table.rowCount()
            self.trajectory_table.insertRow(row)
            defaults = ["0", "0", "0", "0"]
            if row > 0:
                prev = self._cell_float(row - 1, 0, 0.0)
                defaults[0] = str(int(prev + 300))
            for col, value in enumerate(defaults):
                self.trajectory_table.setItem(row, col, QTableWidgetItem(value))

        def remove_selected_trajectory_row(self) -> None:
            rows = sorted({idx.row() for idx in self.trajectory_table.selectedIndexes()}, reverse=True)
            for row in rows:
                self.trajectory_table.removeRow(row)

        def set_trajectory_points(self, points: List[Tuple[float, float, float, float]]) -> None:
            self.trajectory_table.setRowCount(0)
            for point in points:
                row = self.trajectory_table.rowCount()
                self.trajectory_table.insertRow(row)
                for col, value in enumerate(point):
                    self.trajectory_table.setItem(row, col, QTableWidgetItem(str(value)))

        def _cell_float(self, row: int, col: int, default: float) -> float:
            item = self.trajectory_table.item(row, col)
            if item is None or not item.text().strip():
                return default
            return float(item.text())

        def trajectory_points(self) -> List[Tuple[float, float, float, float]]:
            points = []
            for row in range(self.trajectory_table.rowCount()):
                points.append(
                    (
                        self._cell_float(row, 0, 0.0),
                        self._cell_float(row, 1, 0.0),
                        self._cell_float(row, 2, 0.0),
                        self._cell_float(row, 3, 0.0),
                    )
                )
            points.sort(key=lambda point: point[0])
            return points

        def settings(self) -> ScenarioSettings:
            return ScenarioSettings(
                name=f"场景 {self.index}",
                rinex_file=self.path_edit.text().strip(),
                start_time=_qdatetime_to_datetime(self.start_edit.dateTime()),
                end_time=_qdatetime_to_datetime(self.end_edit.dateTime()),
                interval_seconds=self.interval_spin.value(),
                receiver_lat=self.lat_spin.value(),
                receiver_lon=self.lon_spin.value(),
                receiver_height=self.height_spin.value(),
                trajectory_mode=self.mode_combo.currentText(),
                velocity_east_mps=self.ve_spin.value(),
                velocity_north_mps=self.vn_spin.value(),
                velocity_up_mps=self.vu_spin.value(),
                trajectory_points=self.trajectory_points(),
                random_seed=self.seed_spin.value(),
                elevation_mask_deg=self.elevation_spin.value(),
                max_iter=self.max_iter_spin.value(),
                convergence_threshold=self.threshold_spin.value(),
            )


    class ThreeScenarioWorker(QThread):
        log_message = Signal(str)
        progress_changed = Signal(int, int)
        scenario_finished = Signal(object)
        all_finished = Signal(object, object, object)

        def __init__(self, scenarios: List[ScenarioSettings], output_root: Path) -> None:
            super().__init__()
            self.scenarios = scenarios
            self.output_root = output_root

        def run(self) -> None:
            self.output_root.mkdir(parents=True, exist_ok=True)
            run_results: List[ScenarioRunResult] = []
            total_epochs = sum(
                max(int((s.end_time - s.start_time).total_seconds() // s.interval_seconds) + 1, 0)
                for s in self.scenarios
                if s.interval_seconds > 0 and s.end_time > s.start_time
            )
            completed_epochs = 0

            for index, settings in enumerate(self.scenarios, 1):
                output_dir = self.output_root / f"scenario_{index}"
                output_dir.mkdir(parents=True, exist_ok=True)
                self.log_message.emit(f"开始运行{settings.name}：{settings.rinex_file}")

                try:
                    rinex_path = Path(settings.rinex_file)
                    if not rinex_path.exists():
                        raise FileNotFoundError(f"RINEX 文件不存在：{rinex_path}")

                    nav_data, _ = parse_nav_file(rinex_path)
                    receiver_initial_ecef = blh_to_ecef(
                        settings.receiver_lat,
                        settings.receiver_lon,
                        settings.receiver_height,
                    )
                    receiver_initial_approx = (
                        receiver_initial_ecef[0] + 50.0,
                        receiver_initial_ecef[1] - 50.0,
                        receiver_initial_ecef[2] + 30.0,
                    )

                    receiver_trajectory: Optional[Callable[[datetime], ECEF]]
                    if settings.trajectory_mode == "动态直线运动":
                        receiver_trajectory = build_linear_enu_trajectory(
                            settings.start_time,
                            receiver_initial_ecef,
                            settings.receiver_lat,
                            settings.receiver_lon,
                            (
                                settings.velocity_east_mps,
                                settings.velocity_north_mps,
                                settings.velocity_up_mps,
                            ),
                        )
                    elif settings.trajectory_mode == "动态折线轨迹":
                        if not settings.trajectory_points:
                            raise ValueError("动态折线轨迹至少需要 1 个轨迹控制点")
                        receiver_trajectory = build_polyline_enu_trajectory(
                            settings.start_time,
                            receiver_initial_ecef,
                            settings.receiver_lat,
                            settings.receiver_lon,
                            settings.trajectory_points,
                        )
                    else:
                        receiver_trajectory = None

                    def on_epoch(row: dict, epoch_index: int, scenario_total: int) -> None:
                        overall = min(completed_epochs + epoch_index, total_epochs)
                        self.progress_changed.emit(overall, max(total_epochs, 1))

                    results, summary = run_continuous_positioning(
                        nav_data=nav_data,
                        start_time=settings.start_time,
                        end_time=settings.end_time,
                        interval_seconds=settings.interval_seconds,
                        receiver_true_position=receiver_initial_ecef,
                        receiver_trajectory=receiver_trajectory,
                        receiver_initial_approx=receiver_initial_approx,
                        output_dir=output_dir,
                        random_seed=settings.random_seed,
                        max_iter=settings.max_iter,
                        convergence_threshold=settings.convergence_threshold,
                        elevation_mask_deg=settings.elevation_mask_deg,
                        progress_callback=on_epoch,
                    )
                    status = "成功" if summary.success_epochs > 0 else "失败"
                    failure_reason = "" if summary.success_epochs > 0 else summary.evaluation
                    item = ScenarioRunResult(
                        settings=settings,
                        output_dir=output_dir,
                        status=status,
                        failure_reason=failure_reason,
                        results=results,
                        summary=summary,
                    )
                    self.log_message.emit(
                        f"{settings.name}完成：成功率 {summary.success_rate * 100.0:.2f}%，"
                        f"平均误差 {_format_float(summary.mean_error_3d)} m"
                    )
                except Exception as exc:
                    item = ScenarioRunResult(
                        settings=settings,
                        output_dir=output_dir,
                        status="失败",
                        failure_reason=str(exc),
                        results=[],
                        summary=failed_summary(str(exc)),
                    )
                    self.log_message.emit(f"{settings.name}失败：{exc}")

                run_results.append(item)
                scenario_epoch_count = 0
                if settings.interval_seconds > 0 and settings.end_time > settings.start_time:
                    scenario_epoch_count = int(
                        (settings.end_time - settings.start_time).total_seconds()
                        // settings.interval_seconds
                    ) + 1
                completed_epochs += scenario_epoch_count
                self.progress_changed.emit(min(completed_epochs, total_epochs), max(total_epochs, 1))
                self.scenario_finished.emit(item)

            summary_path = self.output_root / "gui_three_scenario_summary.csv"
            report_path = self.output_root / "gui_three_scenario_report.txt"
            write_summary_csv(run_results, summary_path)
            write_text_report(run_results, report_path)
            self.log_message.emit(f"三场景汇总 CSV：{summary_path}")
            self.log_message.emit(f"三场景评估报告：{report_path}")
            self.all_finished.emit(run_results, summary_path, report_path)


    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(f"三场景 RINEX CNAV 批量解算 - {QT_BINDING}")
            self.resize(1500, 920)
            self.worker: Optional[ThreeScenarioWorker] = None
            self.run_results: List[ScenarioRunResult] = []
            self._build_ui()

        def _build_ui(self) -> None:
            root = QWidget()
            root_layout = QVBoxLayout(root)

            splitter = QSplitter(_qt_value("Horizontal", "Orientation"))
            splitter.addWidget(self._build_config_panel())
            splitter.addWidget(self._build_results_panel())
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            root_layout.addWidget(splitter)

            bottom = QHBoxLayout()
            self.run_button = QPushButton("开始三场景解算")
            self.run_button.clicked.connect(self.start_run)
            self.progress_bar = QProgressBar()
            bottom.addWidget(self.run_button)
            bottom.addWidget(self.progress_bar, 1)
            root_layout.addLayout(bottom)
            self.setCentralWidget(root)

        def _build_config_panel(self) -> QWidget:
            panel = QWidget()
            panel.setMinimumWidth(470)
            layout = QVBoxLayout(panel)
            self.config_tabs = QTabWidget()
            defaults = self._default_scenarios()
            self.config_pages: List[ScenarioConfigPage] = []
            for index, values in enumerate(defaults, 1):
                page = ScenarioConfigPage(index, values)
                self.config_pages.append(page)
                self.config_tabs.addTab(page, f"场景 {index}")
            layout.addWidget(self.config_tabs)
            return panel

        def _build_results_panel(self) -> QWidget:
            panel = QWidget()
            layout = QVBoxLayout(panel)

            view_row = QHBoxLayout()
            view_row.addWidget(QLabel("查看"))
            self.view_combo = QComboBox()
            self.view_combo.addItems(["场景 1", "场景 2", "场景 3", "三场景对比"])
            self.view_combo.currentIndexChanged.connect(self.refresh_current_view)
            view_row.addWidget(self.view_combo)
            view_row.addStretch(1)
            layout.addLayout(view_row)

            self.summary_table = QTableWidget(0, 9)
            self.summary_table.setHorizontalHeaderLabels(
                ["场景", "状态", "总历元", "成功", "失败", "成功率", "平均误差", "RMS", "最大误差"]
            )
            self.summary_table.horizontalHeader().setSectionResizeMode(_header_resize_mode("Stretch"))
            layout.addWidget(self.summary_table)

            detail_splitter = QSplitter(_qt_value("Vertical", "Orientation"))
            self.result_table = QTableWidget(0, 9)
            self.result_table.setHorizontalHeaderLabels(
                ["epoch", "status", "sats", "true_X", "true_Y", "true_Z", "X", "Y", "Z"]
            )
            self.result_table.horizontalHeader().setSectionResizeMode(_header_resize_mode("ResizeToContents"))
            detail_splitter.addWidget(self.result_table)

            plot_holder = QWidget()
            plot_layout = QHBoxLayout(plot_holder)
            self.error_canvas = MplCanvas(width=5, height=4)
            self.trajectory_canvas = MplCanvas(width=5, height=4)
            plot_layout.addWidget(self.error_canvas)
            plot_layout.addWidget(self.trajectory_canvas)
            detail_splitter.addWidget(plot_holder)
            detail_splitter.setStretchFactor(0, 1)
            detail_splitter.setStretchFactor(1, 1)
            layout.addWidget(detail_splitter, 1)

            self.log_edit = QTextEdit()
            self.log_edit.setReadOnly(True)
            self.log_edit.setMinimumHeight(140)
            layout.addWidget(self.log_edit)
            return panel

        def _default_scenarios(self) -> List[Dict[str, Any]]:
            return [
                {
                    "rinex_file": "nav/tarc0910.26b_cnav",
                    "start_time": datetime(2026, 4, 1, 0, 0, 0),
                    "end_time": datetime(2026, 4, 1, 6, 0, 0),
                    "interval_seconds": 300,
                    "receiver_lat": 41.0,
                    "receiver_lon": 116.0,
                    "receiver_height": 35.0,
                    "trajectory_mode": "静态接收机",
                    "random_seed": 2026,
                    "elevation_mask_deg": 0.0,
                    "max_iter": 12,
                    "convergence_threshold": 1e-4,
                    "trajectory_points": [(0, 0, 0, 0), (21600, 0, 0, 0)],
                },
                {
                    "rinex_file": "nav/tarc1220.26b_cnav",
                    "start_time": datetime(2026, 5, 2, 0, 0, 0),
                    "end_time": datetime(2026, 5, 2, 6, 0, 0),
                    "interval_seconds": 300,
                    "receiver_lat": 31.2304,
                    "receiver_lon": 121.4737,
                    "receiver_height": 20.0,
                    "trajectory_mode": "动态直线运动",
                    "velocity_east_mps": 0.3,
                    "velocity_north_mps": 0.1,
                    "velocity_up_mps": 0.02,
                    "random_seed": 42,
                    "elevation_mask_deg": 0.0,
                    "max_iter": 12,
                    "convergence_threshold": 1e-4,
                    "trajectory_points": [(0, 0, 0, 0), (21600, 0, 0, 0)],
                },
                {
                    "rinex_file": "nav/tarc1230.26b_cnav",
                    "start_time": datetime(2026, 5, 3, 0, 0, 0),
                    "end_time": datetime(2026, 5, 3, 6, 0, 0),
                    "interval_seconds": 300,
                    "receiver_lat": 23.1291,
                    "receiver_lon": 113.2644,
                    "receiver_height": 20.0,
                    "trajectory_mode": "动态折线轨迹",
                    "velocity_east_mps": 0.0,
                    "velocity_north_mps": 0.0,
                    "velocity_up_mps": 0.0,
                    "random_seed": 2027,
                    "elevation_mask_deg": 0.0,
                    "max_iter": 12,
                    "convergence_threshold": 1e-4,
                    "trajectory_points": [
                        (0, 0, 0, 0),
                        (3600, 800, 300, 100),
                        (7200, 1500, 900, 300),
                        (10800, 2100, 1500, 600),
                        (14400, 2800, 2200, 900),
                        (18000, 3400, 2800, 1200),
                        (21600, 4000, 3500, 1500),
                    ],
                },
            ]

        def validate_settings(self, settings: List[ScenarioSettings]) -> bool:
            for item in settings:
                if not item.rinex_file:
                    QMessageBox.warning(self, "缺少 RINEX 文件", f"{item.name} 未选择 RINEX 文件。")
                    return False
                if item.end_time <= item.start_time:
                    QMessageBox.warning(self, "时间设置错误", f"{item.name} 的 end_time 必须晚于 start_time。")
                    return False
                if item.interval_seconds <= 0:
                    QMessageBox.warning(self, "采样间隔错误", f"{item.name} 的 interval_seconds 必须为正数。")
                    return False
            return True

        def start_run(self) -> None:
            settings = [page.settings() for page in self.config_pages]
            if not self.validate_settings(settings):
                return

            self.run_results = []
            self.summary_table.setRowCount(0)
            self.result_table.setRowCount(0)
            self.progress_bar.setValue(0)
            self.error_canvas.figure.clear()
            self.error_canvas.draw_idle()
            self.trajectory_canvas.figure.clear()
            self.trajectory_canvas.draw_idle()
            self.log_edit.clear()
            self.run_button.setEnabled(False)

            self.worker = ThreeScenarioWorker(settings, OUTPUT_ROOT)
            self.worker.log_message.connect(self.log)
            self.worker.progress_changed.connect(self.on_progress)
            self.worker.scenario_finished.connect(self.on_scenario_finished)
            self.worker.all_finished.connect(self.on_all_finished)
            self.worker.start()

        def on_progress(self, value: int, maximum: int) -> None:
            self.progress_bar.setMaximum(max(maximum, 1))
            self.progress_bar.setValue(value)

        def on_scenario_finished(self, item: ScenarioRunResult) -> None:
            self.run_results.append(item)
            self.refresh_summary_table()
            self.refresh_current_view()

        def on_all_finished(self, run_results: list, summary_path: Path, report_path: Path) -> None:
            self.run_results = list(run_results)
            self.refresh_summary_table()
            self.refresh_current_view()
            self.run_button.setEnabled(True)
            self.log(f"全部完成。summary={summary_path}, report={report_path}")
            QMessageBox.information(self, "完成", "三场景解算与自动评估已完成。")

        def log(self, message: str) -> None:
            self.log_edit.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

        def refresh_summary_table(self) -> None:
            self.summary_table.setRowCount(0)
            for item in self.run_results:
                summary = item.summary
                values = [
                    item.settings.name,
                    item.status,
                    str(summary.total_epochs),
                    str(summary.success_epochs),
                    str(summary.failed_epochs),
                    f"{summary.success_rate * 100.0:.2f}%",
                    _format_float(summary.mean_error_3d),
                    _format_float(summary.rms_error_3d),
                    _format_float(summary.max_error_3d),
                ]
                row = self.summary_table.rowCount()
                self.summary_table.insertRow(row)
                for col, text in enumerate(values):
                    self.summary_table.setItem(row, col, QTableWidgetItem(text))

        def refresh_current_view(self) -> None:
            index = self.view_combo.currentIndex()
            if index == 3:
                self.draw_comparison()
                self.result_table.setRowCount(0)
                return
            if index >= len(self.run_results):
                return
            self.populate_result_table(self.run_results[index].results)
            self.draw_scenario_plots(self.run_results[index])

        def populate_result_table(self, rows: List[dict]) -> None:
            self.result_table.setRowCount(0)
            for row_data in rows:
                row = self.result_table.rowCount()
                self.result_table.insertRow(row)
                values = [
                    row_data.get("epoch_time", ""),
                    row_data.get("status", ""),
                    row_data.get("satellite_count", ""),
                    _format_float(row_data.get("true_X")),
                    _format_float(row_data.get("true_Y")),
                    _format_float(row_data.get("true_Z")),
                    _format_float(row_data.get("X")),
                    _format_float(row_data.get("Y")),
                    _format_float(row_data.get("Z")),
                ]
                for col, value in enumerate(values):
                    self.result_table.setItem(row, col, QTableWidgetItem(str(value)))

        def _finite_float(self, value: Any) -> Optional[float]:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            return numeric if math.isfinite(numeric) else None

        def draw_scenario_plots(self, item: ScenarioRunResult) -> None:
            rows = item.results
            error_ax = self.error_canvas.figure.clear()
            error_ax = self.error_canvas.figure.add_subplot(111)
            errors = [
                self._finite_float(row.get("error_3d")) for row in rows
            ]
            x_values = list(range(1, len(rows) + 1))
            y_values = [value if value is not None else math.nan for value in errors]
            if x_values:
                error_ax.plot(x_values, y_values, marker="o", linewidth=1.2)
            error_ax.set_title(f"{item.settings.name} 误差曲线")
            error_ax.set_xlabel("历元序号")
            error_ax.set_ylabel("三维误差 (m)")
            error_ax.grid(True, alpha=0.3)
            self.error_canvas.draw_idle()

            traj_ax = self.trajectory_canvas.figure.clear()
            traj_ax = self.trajectory_canvas.figure.add_subplot(111)
            est_lons = [self._finite_float(row.get("lon")) for row in rows]
            est_lats = [self._finite_float(row.get("lat")) for row in rows]
            true_lons = [self._finite_float(row.get("true_lon")) for row in rows]
            true_lats = [self._finite_float(row.get("true_lat")) for row in rows]

            if any(value is not None for value in true_lons) and any(value is not None for value in true_lats):
                traj_ax.plot(true_lons, true_lats, marker="s", linestyle="--", label="真实轨迹")
            if any(value is not None for value in est_lons) and any(value is not None for value in est_lats):
                traj_ax.plot(est_lons, est_lats, marker="o", linewidth=1.2, label="解算轨迹")
            traj_ax.set_title(f"{item.settings.name} 轨迹对比")
            traj_ax.set_xlabel("经度 (deg)")
            traj_ax.set_ylabel("纬度 (deg)")
            traj_ax.grid(True, alpha=0.3)
            traj_ax.legend(loc="best")
            self.trajectory_canvas.draw_idle()

        def draw_comparison(self) -> None:
            error_ax = self.error_canvas.figure.clear()
            error_ax = self.error_canvas.figure.add_subplot(111)
            names = [item.settings.name for item in self.run_results]
            mean_errors = [
                item.summary.mean_error_3d if math.isfinite(item.summary.mean_error_3d) else math.nan
                for item in self.run_results
            ]
            if names:
                error_ax.bar(names, mean_errors)
            error_ax.set_title("三场景平均误差对比")
            error_ax.set_ylabel("平均三维误差 (m)")
            error_ax.tick_params(axis="x", rotation=20)
            error_ax.grid(True, axis="y", alpha=0.3)
            self.error_canvas.draw_idle()

            dop_ax = self.trajectory_canvas.figure.clear()
            dop_ax = self.trajectory_canvas.figure.add_subplot(111)
            pdops = [
                item.summary.average_pdop if math.isfinite(item.summary.average_pdop) else math.nan
                for item in self.run_results
            ]
            gdops = [
                item.summary.average_gdop if math.isfinite(item.summary.average_gdop) else math.nan
                for item in self.run_results
            ]
            x_values = list(range(len(names)))
            if names:
                dop_ax.plot(x_values, pdops, marker="o", label="PDOP")
                dop_ax.plot(x_values, gdops, marker="s", label="GDOP")
                dop_ax.set_xticks(x_values)
                dop_ax.set_xticklabels(names, rotation=20)
            dop_ax.set_title("三场景 DOP 对比")
            dop_ax.grid(True, alpha=0.3)
            dop_ax.legend(loc="best")
            self.trajectory_canvas.draw_idle()

        def closeEvent(self, event: Any) -> None:
            if self.worker is not None and self.worker.isRunning():
                QMessageBox.warning(self, "正在解算", "三场景解算仍在运行，请等待完成后关闭。")
                event.ignore()
                return
            super().closeEvent(event)


def main() -> int:
    if QT_IMPORT_ERROR is not None:
        print("未检测到可用 Qt 绑定，请安装 PyQt6、PySide6 或 PyQt5。")
        return 1

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    if hasattr(app, "exec"):
        return int(app.exec())
    return int(app.exec_())


if __name__ == "__main__":
    sys.exit(main())
