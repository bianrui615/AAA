"""
rinex_gui.py

北斗 RINEX NAV 连续定位可视化交互界面。

功能：
- 导入 RINEX NAV 导航文件；
- 设置连续定位起止时间、采样间隔、最大迭代次数和收敛阈值；
- 后台线程逐历元解算并实时刷新表格、统计信息和图表；
- 支持定位轨迹回放和误差曲线查看。
"""

from __future__ import annotations

import csv
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path("output") / "matplotlib_cache"))

try:
    from PySide6.QtCore import QDate, QDateTime, QThread, QTime, QTimer, Qt, Signal
    from PySide6.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QDateTimeEdit,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QSlider,
        QSpinBox,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    QT_BINDING = "PySide6"
    QT_IMPORT_ERROR: Optional[Exception] = None
except ModuleNotFoundError as pyside_error:
    try:
        from PyQt6.QtCore import QDate, QDateTime, QThread, QTime, QTimer, Qt, pyqtSignal as Signal
        from PyQt6.QtWidgets import (
            QApplication,
            QAbstractItemView,
            QDateTimeEdit,
            QDoubleSpinBox,
            QFileDialog,
            QFrame,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QProgressBar,
            QSlider,
            QSpinBox,
            QSplitter,
            QTableWidget,
            QTableWidgetItem,
            QTabWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
        QT_BINDING = "PyQt6"
        QT_IMPORT_ERROR = None
    except ModuleNotFoundError as pyqt6_error:
        try:
            from PyQt5.QtCore import QDate, QDateTime, QThread, QTime, QTimer, Qt, pyqtSignal as Signal
            from PyQt5.QtWidgets import (
                QApplication,
                QAbstractItemView,
                QDateTimeEdit,
                QDoubleSpinBox,
                QFileDialog,
                QFrame,
                QGridLayout,
                QGroupBox,
                QHBoxLayout,
                QLabel,
                QLineEdit,
                QMainWindow,
                QMessageBox,
                QPushButton,
                QProgressBar,
                QSlider,
                QSpinBox,
                QSplitter,
                QTableWidget,
                QTableWidgetItem,
                QTabWidget,
                QTextEdit,
                QVBoxLayout,
                QWidget,
            )
            QT_BINDING = "PyQt5"
            QT_IMPORT_ERROR = None
        except ModuleNotFoundError as pyqt5_error:
            QT_BINDING = ""
            QT_IMPORT_ERROR = pyqt5_error or pyqt6_error or pyside_error

if QT_IMPORT_ERROR is None:
    QT_HORIZONTAL = getattr(Qt, "Horizontal", getattr(getattr(Qt, "Orientation", Qt), "Horizontal", None))
    QT_ALIGN_RIGHT = getattr(Qt, "AlignRight", getattr(getattr(Qt, "AlignmentFlag", Qt), "AlignRight", None))
    QT_ALIGN_VCENTER = getattr(Qt, "AlignVCenter", getattr(getattr(Qt, "AlignmentFlag", Qt), "AlignVCenter", None))

    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    from module1_nav_parser import parse_rinex_nav_with_info
    from module3_spp_solver import ECEF, ecef_to_blh
    from module4_continuous_analysis import (
        AnalysisSummary,
        calculate_summary,
        plot_results,
        run_continuous_positioning,
    )
    from module5_main_system_test import (
        CONVERGENCE_THRESHOLD,
        ELEVATION_MASK_DEG,
        MAX_ITERATIONS,
        NAV_FILE_PATH,
        OUTPUT_DIR,
        RANDOM_SEED,
        RECEIVER_TRUE_POSITION,
        SAMPLING_INTERVAL_SECONDS,
        SIMULATION_END_TIME,
        SIMULATION_START_TIME,
    )


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


def _format_number(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NaN"
    if not math.isfinite(numeric):
        return "NaN"
    return f"{numeric:.{digits}f}"


def _blh_to_ecef(lat_deg: float, lon_deg: float, height: float) -> ECEF:
    """将 WGS84/CGCS2000 经纬高转换为 ECEF 坐标，单位为 m。"""

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

    x = (prime_vertical_radius + height) * cos_lat * math.cos(lon)
    y = (prime_vertical_radius + height) * cos_lat * math.sin(lon)
    z = (prime_vertical_radius * (1.0 - eccentricity_squared) + height) * sin_lat
    return x, y, z


if QT_IMPORT_ERROR is None:

    class PositioningWorker(QThread):
        progress_row = Signal(dict, int, int)
        finished_ok = Signal(list, object)
        failed = Signal(str)
        log_message = Signal(str)

        def __init__(
            self,
            nav_data: Dict[str, List[Any]],
            start_time: datetime,
            end_time: datetime,
            interval_seconds: int,
            receiver_true_position: ECEF,
            output_dir: Path,
            random_seed: int,
            max_iter: int,
            convergence_threshold: float,
            elevation_mask_deg: float = 0.0,
        ) -> None:
            super().__init__()
            self.nav_data = nav_data
            self.start_time = start_time
            self.end_time = end_time
            self.interval_seconds = interval_seconds
            self.receiver_true_position = receiver_true_position
            self.output_dir = output_dir
            self.random_seed = random_seed
            self.max_iter = max_iter
            self.convergence_threshold = convergence_threshold
            self.elevation_mask_deg = elevation_mask_deg

        def run(self) -> None:
            try:
                self.log_message.emit("开始连续定位解算...")

                def on_progress(row: dict, index: int, total: int) -> None:
                    self.progress_row.emit(row, index, total)

                results, summary = run_continuous_positioning(
                    nav_data=self.nav_data,
                    start_time=self.start_time,
                    end_time=self.end_time,
                    interval_seconds=self.interval_seconds,
                    receiver_true_position=self.receiver_true_position,
                    output_dir=self.output_dir,
                    random_seed=self.random_seed,
                    max_iter=self.max_iter,
                    convergence_threshold=self.convergence_threshold,
                    elevation_mask_deg=self.elevation_mask_deg,
                    progress_callback=on_progress,
                )
                self.finished_ok.emit(results, summary)
            except Exception as exc:
                self.failed.emit(str(exc))


    class MplCanvas(FigureCanvas):
        def __init__(self, width: float = 5.0, height: float = 4.0) -> None:
            self.figure = Figure(figsize=(width, height), tight_layout=True)
            super().__init__(self.figure)


    class MainWindow(QMainWindow):
        table_headers = [
            "历元时间",
            "状态",
            "卫星数",
            "X(m)",
            "Y(m)",
            "Z(m)",
            "纬度(deg)",
            "经度(deg)",
            "误差(m)",
            "迭代",
        ]

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(f"北斗 RINEX 连续定位可视化 - {QT_BINDING}")
            self.resize(1280, 820)

            self.nav_data: Optional[Dict[str, List[Any]]] = None
            self.results: List[dict] = []
            self.summary: Optional[AnalysisSummary] = None
            self.worker: Optional[PositioningWorker] = None

            self.play_timer = QTimer(self)
            self.play_timer.timeout.connect(self._advance_playback)

            self._build_ui()
            self._apply_default_values()
            self._set_running(False)

        def _build_ui(self) -> None:
            root = QWidget()
            root_layout = QHBoxLayout(root)
            root_layout.setContentsMargins(10, 10, 10, 10)

            splitter = QSplitter(QT_HORIZONTAL)
            splitter.addWidget(self._build_control_panel())
            splitter.addWidget(self._build_workspace())
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            root_layout.addWidget(splitter)
            self.setCentralWidget(root)

        def _build_control_panel(self) -> QWidget:
            panel = QWidget()
            panel.setMinimumWidth(330)
            panel.setMaximumWidth(410)
            layout = QVBoxLayout(panel)

            file_group = QGroupBox("RINEX 数据导入")
            file_layout = QVBoxLayout(file_group)
            self.nav_path_edit = QLineEdit()
            self.nav_path_edit.setReadOnly(True)
            self.nav_path_edit.setPlaceholderText("请选择 RINEX NAV 文件")
            self.import_button = QPushButton("导入 RINEX NAV")
            self.import_button.clicked.connect(self.import_nav_file)
            file_layout.addWidget(self.nav_path_edit)
            file_layout.addWidget(self.import_button)
            layout.addWidget(file_group)

            param_group = QGroupBox("解算参数")
            param_layout = QGridLayout(param_group)
            self.start_edit = QDateTimeEdit()
            self.start_edit.setCalendarPopup(True)
            self.start_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
            self.end_edit = QDateTimeEdit()
            self.end_edit.setCalendarPopup(True)
            self.end_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
            self.interval_spin = QSpinBox()
            self.interval_spin.setRange(1, 86400)
            self.interval_spin.setSuffix(" s")
            self.max_iter_spin = QSpinBox()
            self.max_iter_spin.setRange(1, 100)
            self.threshold_spin = QDoubleSpinBox()
            self.threshold_spin.setRange(1e-9, 1000.0)
            self.threshold_spin.setDecimals(9)
            self.threshold_spin.setSingleStep(0.0001)
            self.seed_spin = QSpinBox()
            self.seed_spin.setRange(0, 999999999)
            self.elevation_mask_spin = QDoubleSpinBox()
            self.elevation_mask_spin.setRange(0.0, 90.0)
            self.elevation_mask_spin.setDecimals(1)
            self.elevation_mask_spin.setSingleStep(1.0)
            self.elevation_mask_spin.setSuffix("°")

            param_layout.addWidget(QLabel("起始时间"), 0, 0)
            param_layout.addWidget(self.start_edit, 0, 1)
            param_layout.addWidget(QLabel("结束时间"), 1, 0)
            param_layout.addWidget(self.end_edit, 1, 1)
            param_layout.addWidget(QLabel("采样间隔"), 2, 0)
            param_layout.addWidget(self.interval_spin, 2, 1)
            param_layout.addWidget(QLabel("最大迭代"), 3, 0)
            param_layout.addWidget(self.max_iter_spin, 3, 1)
            param_layout.addWidget(QLabel("误差阈值(m)"), 4, 0)
            param_layout.addWidget(self.threshold_spin, 4, 1)
            param_layout.addWidget(QLabel("随机种子"), 5, 0)
            param_layout.addWidget(self.seed_spin, 5, 1)
            param_layout.addWidget(QLabel("高度角截止"), 6, 0)
            param_layout.addWidget(self.elevation_mask_spin, 6, 1)
            layout.addWidget(param_group)

            receiver_group = QGroupBox("接收机真实坐标")
            receiver_layout = QVBoxLayout(receiver_group)
            self.receiver_tabs = QTabWidget()

            ecef_page = QWidget()
            ecef_layout = QGridLayout(ecef_page)
            self.receiver_x_spin = self._ecef_spin()
            self.receiver_y_spin = self._ecef_spin()
            self.receiver_z_spin = self._ecef_spin()
            ecef_layout.addWidget(QLabel("X(m)"), 0, 0)
            ecef_layout.addWidget(self.receiver_x_spin, 0, 1)
            ecef_layout.addWidget(QLabel("Y(m)"), 1, 0)
            ecef_layout.addWidget(self.receiver_y_spin, 1, 1)
            ecef_layout.addWidget(QLabel("Z(m)"), 2, 0)
            ecef_layout.addWidget(self.receiver_z_spin, 2, 1)
            self.receiver_tabs.addTab(ecef_page, "方法一：ECEF")

            blh_page = QWidget()
            blh_layout = QGridLayout(blh_page)
            self.receiver_lat_spin = self._angle_spin(-90.0, 90.0)
            self.receiver_lon_spin = self._angle_spin(-180.0, 180.0)
            self.receiver_height_spin = self._height_spin()
            self.receiver_lat_spin.valueChanged.connect(self._update_blh_preview)
            self.receiver_lon_spin.valueChanged.connect(self._update_blh_preview)
            self.receiver_height_spin.valueChanged.connect(self._update_blh_preview)
            self.convert_blh_button = QPushButton("转换并填入 ECEF")
            self.convert_blh_button.clicked.connect(self._sync_blh_to_ecef)
            self.converted_ecef_label = QLabel("-")
            self.converted_ecef_label.setWordWrap(True)

            blh_layout.addWidget(QLabel("纬度(deg)"), 0, 0)
            blh_layout.addWidget(self.receiver_lat_spin, 0, 1)
            blh_layout.addWidget(QLabel("经度(deg)"), 1, 0)
            blh_layout.addWidget(self.receiver_lon_spin, 1, 1)
            blh_layout.addWidget(QLabel("高度(m)"), 2, 0)
            blh_layout.addWidget(self.receiver_height_spin, 2, 1)
            blh_layout.addWidget(self.convert_blh_button, 3, 0, 1, 2)
            blh_layout.addWidget(self.converted_ecef_label, 4, 0, 1, 2)
            self.receiver_tabs.addTab(blh_page, "方法二：经纬高")

            receiver_layout.addWidget(self.receiver_tabs)
            layout.addWidget(receiver_group)

            action_layout = QHBoxLayout()
            self.run_button = QPushButton("开始解算")
            self.run_button.clicked.connect(self.start_positioning)
            self.load_csv_button = QPushButton("载入结果")
            self.load_csv_button.clicked.connect(self.load_existing_csv)
            self.export_button = QPushButton("导出结果")
            self.export_button.clicked.connect(self.export_outputs)
            action_layout.addWidget(self.run_button)
            action_layout.addWidget(self.load_csv_button)
            action_layout.addWidget(self.export_button)
            layout.addLayout(action_layout)

            self.progress_bar = QProgressBar()
            layout.addWidget(self.progress_bar)

            self.log_box = QTextEdit()
            self.log_box.setReadOnly(True)
            self.log_box.setMinimumHeight(160)
            layout.addWidget(self.log_box)
            layout.addStretch(1)
            return panel

        def _build_workspace(self) -> QWidget:
            workspace = QWidget()
            layout = QVBoxLayout(workspace)

            self.stats_frame = QFrame()
            stats_layout = QGridLayout(self.stats_frame)
            self.stat_total = QLabel("-")
            self.stat_success = QLabel("-")
            self.stat_rate = QLabel("-")
            self.stat_mean_error = QLabel("-")
            self.stat_rms = QLabel("-")
            self.stat_pdop = QLabel("-")
            stats = [
                ("总历元", self.stat_total),
                ("成功历元", self.stat_success),
                ("成功率", self.stat_rate),
                ("平均误差", self.stat_mean_error),
                ("RMS 误差", self.stat_rms),
                ("平均 PDOP", self.stat_pdop),
            ]
            for index, (title, label) in enumerate(stats):
                title_label = QLabel(title)
                title_label.setStyleSheet("color: #666;")
                label.setStyleSheet("font-size: 18px; font-weight: 600;")
                stats_layout.addWidget(title_label, index // 3 * 2, index % 3)
                stats_layout.addWidget(label, index // 3 * 2 + 1, index % 3)
            layout.addWidget(self.stats_frame)

            self.tabs = QTabWidget()
            self.tabs.addTab(self._build_result_tab(), "实时结果")
            self.tabs.addTab(self._build_trajectory_tab(), "轨迹回放")
            self.tabs.addTab(self._build_error_tab(), "误差曲线")
            layout.addWidget(self.tabs, 1)
            return workspace

        def _build_result_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            self.result_table = QTableWidget(0, len(self.table_headers))
            self.result_table.setHorizontalHeaderLabels(self.table_headers)
            self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.result_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.result_table.verticalHeader().setVisible(False)
            self.result_table.horizontalHeader().setStretchLastSection(True)
            layout.addWidget(self.result_table)
            return tab

        def _build_trajectory_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            self.trajectory_canvas = MplCanvas(width=6, height=5)
            layout.addWidget(self.trajectory_canvas, 1)

            controls = QHBoxLayout()
            self.play_button = QPushButton("播放")
            self.play_button.clicked.connect(self.toggle_playback)
            self.play_slider = QSlider(QT_HORIZONTAL)
            self.play_slider.setRange(0, 0)
            self.play_slider.valueChanged.connect(self.update_playback_plot)
            self.play_speed_spin = QSpinBox()
            self.play_speed_spin.setRange(100, 3000)
            self.play_speed_spin.setSingleStep(100)
            self.play_speed_spin.setValue(500)
            self.play_speed_spin.setSuffix(" ms")
            controls.addWidget(self.play_button)
            controls.addWidget(self.play_slider, 1)
            controls.addWidget(QLabel("间隔"))
            controls.addWidget(self.play_speed_spin)
            layout.addLayout(controls)
            return tab

        def _build_error_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            self.error_canvas = MplCanvas(width=7, height=4)
            layout.addWidget(self.error_canvas, 1)
            return tab

        def _ecef_spin(self) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setRange(-50000000.0, 50000000.0)
            spin.setDecimals(4)
            spin.setSingleStep(10.0)
            return spin

        def _angle_spin(self, minimum: float, maximum: float) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setDecimals(10)
            spin.setSingleStep(0.000001)
            return spin

        def _height_spin(self) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setRange(-10000.0, 100000.0)
            spin.setDecimals(4)
            spin.setSingleStep(1.0)
            return spin

        def _apply_default_values(self) -> None:
            nav_path = Path(NAV_FILE_PATH)
            self.nav_path_edit.setText(str(nav_path))
            self.start_edit.setDateTime(_datetime_to_qdatetime(SIMULATION_START_TIME))
            self.end_edit.setDateTime(_datetime_to_qdatetime(SIMULATION_END_TIME))
            self.interval_spin.setValue(SAMPLING_INTERVAL_SECONDS)
            self.max_iter_spin.setValue(MAX_ITERATIONS)
            self.threshold_spin.setValue(CONVERGENCE_THRESHOLD)
            self.seed_spin.setValue(RANDOM_SEED)
            self.elevation_mask_spin.setValue(ELEVATION_MASK_DEG)
            self.receiver_x_spin.setValue(RECEIVER_TRUE_POSITION[0])
            self.receiver_y_spin.setValue(RECEIVER_TRUE_POSITION[1])
            self.receiver_z_spin.setValue(RECEIVER_TRUE_POSITION[2])
            lat, lon, height = ecef_to_blh(*RECEIVER_TRUE_POSITION)
            self.receiver_lat_spin.setValue(lat)
            self.receiver_lon_spin.setValue(lon)
            self.receiver_height_spin.setValue(height)
            self._update_blh_preview()
            self.log(f"界面已启动。默认文件：{nav_path}")
            if nav_path.exists():
                self.import_nav_file(nav_path)

        def _receiver_blh(self) -> tuple[float, float, float]:
            return (
                self.receiver_lat_spin.value(),
                self.receiver_lon_spin.value(),
                self.receiver_height_spin.value(),
            )

        def _receiver_position_from_inputs(self) -> ECEF:
            if self.receiver_tabs.currentIndex() == 1:
                lat, lon, height = self._receiver_blh()
                position = _blh_to_ecef(lat, lon, height)
                self.receiver_x_spin.setValue(position[0])
                self.receiver_y_spin.setValue(position[1])
                self.receiver_z_spin.setValue(position[2])
                self._update_blh_preview()
                self.log(
                    "经纬高已自动转换为 ECEF："
                    f"X={position[0]:.4f} m，"
                    f"Y={position[1]:.4f} m，"
                    f"Z={position[2]:.4f} m"
                )
                return position
            return (
                self.receiver_x_spin.value(),
                self.receiver_y_spin.value(),
                self.receiver_z_spin.value(),
            )

        def _sync_blh_to_ecef(self) -> None:
            lat, lon, height = self._receiver_blh()
            x, y, z = _blh_to_ecef(lat, lon, height)
            self.receiver_x_spin.setValue(x)
            self.receiver_y_spin.setValue(y)
            self.receiver_z_spin.setValue(z)
            self._update_blh_preview()
            self.log(f"已转换并填入 ECEF：X={x:.4f} m，Y={y:.4f} m，Z={z:.4f} m")

        def _update_blh_preview(self, *_: object) -> None:
            if not hasattr(self, "converted_ecef_label"):
                return
            lat, lon, height = self._receiver_blh()
            x, y, z = _blh_to_ecef(lat, lon, height)
            self.converted_ecef_label.setText(
                f"自动换算：X={x:.4f} m，Y={y:.4f} m，Z={z:.4f} m"
            )

        def log(self, message: str) -> None:
            self.log_box.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

        def import_nav_file(self, selected_path: Optional[Path] = None) -> None:
            if selected_path is None or isinstance(selected_path, bool):
                file_name, _ = QFileDialog.getOpenFileName(
                    self,
                    "选择 RINEX NAV 文件",
                    str(Path.cwd()),
                    "RINEX NAV (*.rnx *.nav *.26b *.??n *.*)",
                )
                if not file_name:
                    return
                selected_path = Path(file_name)

            try:
                self.log(f"正在解析：{selected_path}")
                nav_data, parse_info = parse_rinex_nav_with_info(selected_path)
                sat_count = len(nav_data)
                eph_count = sum(len(records) for records in nav_data.values())
                if sat_count < 4:
                    raise ValueError("北斗卫星数量少于 4 颗，无法进行定位解算。")
                self.nav_data = nav_data
                self.nav_path_edit.setText(str(selected_path))
                self.log(
                    "导入完成："
                    f"RINEX {parse_info.rinex_version}，"
                    f"北斗卫星 {sat_count} 颗，星历记录 {eph_count} 条。"
                )
            except Exception as exc:
                self.nav_data = None
                QMessageBox.critical(self, "导入失败", str(exc))
                self.log(f"导入失败：{exc}")

        def start_positioning(self) -> None:
            if self.nav_data is None:
                path = Path(self.nav_path_edit.text())
                if path.exists():
                    self.import_nav_file(path)
                if self.nav_data is None:
                    QMessageBox.warning(self, "缺少数据", "请先导入有效的 RINEX NAV 文件。")
                    return

            start_time = _qdatetime_to_datetime(self.start_edit.dateTime())
            end_time = _qdatetime_to_datetime(self.end_edit.dateTime())
            if end_time < start_time:
                QMessageBox.warning(self, "参数错误", "结束时间不能早于起始时间。")
                return

            receiver_position = self._receiver_position_from_inputs()
            self._reset_results()
            self._set_running(True)
            self.worker = PositioningWorker(
                nav_data=self.nav_data,
                start_time=start_time,
                end_time=end_time,
                interval_seconds=self.interval_spin.value(),
                receiver_true_position=receiver_position,
                output_dir=Path(OUTPUT_DIR),
                random_seed=self.seed_spin.value(),
                max_iter=self.max_iter_spin.value(),
                convergence_threshold=self.threshold_spin.value(),
                elevation_mask_deg=self.elevation_mask_spin.value(),
            )
            self.worker.progress_row.connect(self.add_result_row)
            self.worker.finished_ok.connect(self.positioning_finished)
            self.worker.failed.connect(self.positioning_failed)
            self.worker.log_message.connect(self.log)
            self.worker.start()

        def add_result_row(self, row: dict, index: int, total: int) -> None:
            self.results.append(row)
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(index)

            values = [
                row.get("epoch_time", ""),
                row.get("status", ""),
                str(row.get("satellite_count", "")),
                _format_number(row.get("X")),
                _format_number(row.get("Y")),
                _format_number(row.get("Z")),
                _format_number(row.get("lat"), 8),
                _format_number(row.get("lon"), 8),
                _format_number(row.get("error_3d")),
                str(row.get("iteration_count", "")),
            ]
            table_row = self.result_table.rowCount()
            self.result_table.insertRow(table_row)
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if col in {2, 3, 4, 5, 6, 7, 8, 9}:
                    item.setTextAlignment(QT_ALIGN_RIGHT | QT_ALIGN_VCENTER)
                self.result_table.setItem(table_row, col, item)
            self.result_table.scrollToBottom()

            live_summary = calculate_summary(self.results)
            self._update_summary(live_summary)
            self._refresh_error_plot()
            self._refresh_playback_controls()

        def positioning_finished(self, results: list, summary: object) -> None:
            self.results = list(results)
            self.summary = summary
            self._set_running(False)
            self._update_summary(summary)
            self._refresh_error_plot()
            self._refresh_playback_controls()
            self.update_playback_plot(self.play_slider.value())
            self.log("解算完成，CSV、统计文件和结果图已写入 output 目录。")
            QMessageBox.information(self, "完成", "连续定位解算完成。")

        def positioning_failed(self, message: str) -> None:
            self._set_running(False)
            QMessageBox.critical(self, "解算失败", message)
            self.log(f"解算失败：{message}")

        def load_existing_csv(self) -> None:
            default_path = Path(OUTPUT_DIR) / "module4_continuous_position_results.csv"
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "载入定位结果 CSV",
                str(default_path if default_path.exists() else Path.cwd()),
                "CSV 文件 (*.csv)",
            )
            if not file_name:
                return

            try:
                with Path(file_name).open("r", encoding="utf-8-sig", newline="") as file:
                    rows = list(csv.DictReader(file))
                self._reset_results()
                for row in rows:
                    self.add_result_row(row, len(self.results) + 1, len(rows))
                self.summary = calculate_summary(self.results)
                self._update_summary(self.summary)
                self.log(f"已载入结果：{file_name}")
            except Exception as exc:
                QMessageBox.critical(self, "载入失败", str(exc))
                self.log(f"载入失败：{exc}")

        def export_outputs(self) -> None:
            source_dir = Path(OUTPUT_DIR)
            if not source_dir.exists():
                QMessageBox.warning(self, "缺少输出", "当前还没有可导出的 output 目录。")
                return
            target_name = QFileDialog.getExistingDirectory(
                self,
                "选择导出目录",
                str(Path.cwd()),
            )
            if not target_name:
                return
            target_dir = Path(target_name)
            exported = 0
            wanted = [
                "positioning_results.csv",
                "module4_continuous_position_results.csv",
                "accuracy_report.md",
                "module4_error_statistics.txt",
                "trajectory.png",
                "position_error.png",
                "dop_and_sat_count.png",
                "module4_trajectory.png",
                "module4_error_curve.png",
                "module4_satellite_dop_curve.png",
                "test_report.md",
                "module5_system_test_report.txt",
            ]
            for name in wanted:
                src = source_dir / name
                if src.exists():
                    shutil.copy2(src, target_dir / name)
                    exported += 1
            self.log(f"已导出 {exported} 个结果/报告文件到：{target_dir}")
            QMessageBox.information(self, "导出完成", f"已导出 {exported} 个文件。")

        def toggle_playback(self) -> None:
            if not self.results:
                return
            if self.play_timer.isActive():
                self.play_timer.stop()
                self.play_button.setText("播放")
            else:
                self.play_timer.start(self.play_speed_spin.value())
                self.play_button.setText("暂停")

        def _advance_playback(self) -> None:
            value = self.play_slider.value()
            if value >= self.play_slider.maximum():
                self.play_timer.stop()
                self.play_button.setText("播放")
                return
            self.play_slider.setValue(value + 1)

        def update_playback_plot(self, value: int) -> None:
            self._draw_trajectory(value)

        def _refresh_playback_controls(self) -> None:
            maximum = max(len(self.results) - 1, 0)
            self.play_slider.blockSignals(True)
            self.play_slider.setRange(0, maximum)
            if self.play_slider.value() > maximum:
                self.play_slider.setValue(maximum)
            self.play_slider.blockSignals(False)
            self._draw_trajectory(self.play_slider.value())

        def _draw_trajectory(self, current_index: int) -> None:
            ax = self.trajectory_canvas.figure.clear()
            ax = self.trajectory_canvas.figure.add_subplot(111)
            success_rows = [
                row
                for row in self.results
                if self._is_finite(row.get("lat")) and self._is_finite(row.get("lon"))
            ]
            if not success_rows:
                ax.set_title("暂无可显示轨迹")
                ax.set_xlabel("经度 (deg)")
                ax.set_ylabel("纬度 (deg)")
                ax.grid(True, alpha=0.3)
                self.trajectory_canvas.draw_idle()
                return

            lons = [float(row["lon"]) for row in success_rows]
            lats = [float(row["lat"]) for row in success_rows]
            current_index = min(max(current_index, 0), len(success_rows) - 1)
            ax.plot(lons[: current_index + 1], lats[: current_index + 1], color="#2563eb", linewidth=1.5)
            ax.scatter(lons, lats, s=18, color="#94a3b8", label="全部历元")
            ax.scatter(
                [lons[current_index]],
                [lats[current_index]],
                s=70,
                color="#dc2626",
                label="当前历元",
                zorder=3,
            )
            ax.set_title("定位轨迹回放")
            ax.set_xlabel("经度 (deg)")
            ax.set_ylabel("纬度 (deg)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            self.trajectory_canvas.draw_idle()

        def _refresh_error_plot(self) -> None:
            ax = self.error_canvas.figure.clear()
            ax = self.error_canvas.figure.add_subplot(111)
            x_axis = list(range(1, len(self.results) + 1))
            errors = [
                float(row["error_3d"]) if self._is_finite(row.get("error_3d")) else math.nan
                for row in self.results
            ]
            if x_axis:
                ax.plot(x_axis, errors, marker="o", color="#0f766e", linewidth=1.4, markersize=4)
            ax.set_title("三维定位误差曲线")
            ax.set_xlabel("历元序号")
            ax.set_ylabel("误差 (m)")
            ax.grid(True, alpha=0.3)
            self.error_canvas.draw_idle()

        def _update_summary(self, summary: AnalysisSummary) -> None:
            self.stat_total.setText(str(summary.total_epochs))
            self.stat_success.setText(str(summary.success_epochs))
            self.stat_rate.setText(f"{summary.success_rate * 100:.2f}%")
            self.stat_mean_error.setText(f"{_format_number(summary.mean_error_3d)} m")
            self.stat_rms.setText(f"{_format_number(summary.rms_error_3d)} m")
            self.stat_pdop.setText(_format_number(summary.average_pdop))

        def _reset_results(self) -> None:
            self.results = []
            self.summary = None
            self.result_table.setRowCount(0)
            self.progress_bar.setValue(0)
            self._update_summary(
                AnalysisSummary(0, 0, 0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, 0.0, "")
            )
            self._refresh_error_plot()
            self._refresh_playback_controls()

        def _set_running(self, running: bool) -> None:
            self.run_button.setEnabled(not running)
            self.import_button.setEnabled(not running)
            self.load_csv_button.setEnabled(not running)
            self.progress_bar.setFormat("解算中 %p%" if running else "%p%")

        @staticmethod
        def _is_finite(value: Any) -> bool:
            try:
                return math.isfinite(float(value))
            except (TypeError, ValueError):
                return False

        def closeEvent(self, event: Any) -> None:
            if self.worker is not None and self.worker.isRunning():
                QMessageBox.warning(self, "正在解算", "当前解算仍在运行，请等待完成后关闭窗口。")
                event.ignore()
                return
            super().closeEvent(event)


def main() -> int:
    if QT_IMPORT_ERROR is not None:
        print("未检测到 Qt 绑定，请先安装 PyQt5 或 PySide6。")
        print("推荐命令：python -m pip install PyQt5")
        return 1

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    if hasattr(app, "exec"):
        return int(app.exec())
    return int(app.exec_())


if __name__ == "__main__":
    sys.exit(main())
