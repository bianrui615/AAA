"""
rinex_gui.py

北斗 RINEX NAV 连续定位可视化交互界面。

功能：
- 导入 RINEX NAV 导航文件；
- 设置连续定位起止时间、采样间隔、最大迭代次数和收敛阈值；
- 支持静态/动态接收机轨迹输入（匀速直线、表格折线、CSV 文件）；
- 后台线程逐历元解算并实时刷新表格、统计信息和图表；
- 支持定位轨迹回放、轨迹预览和误差曲线查看。
"""

from __future__ import annotations

import sys
from pathlib import Path
# 确保项目根目录在 sys.path 中，支持直接运行和作为模块导入
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import csv
import math
import os
import shutil
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/basic") / "matplotlib_cache"))

try:
    from PyQt5.QtCore import QDate, QDateTime, QThread, QTime, QTimer, Qt, pyqtSignal as Signal
    from PyQt5.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QComboBox,
        QDateTimeEdit,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
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
        QSlider,
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
    QT_IMPORT_ERROR: Optional[Exception] = None
except ModuleNotFoundError as pyqt5_error:
    QT_BINDING = ""
    QT_IMPORT_ERROR = pyqt5_error

if QT_IMPORT_ERROR is None:
    QT_HORIZONTAL = getattr(Qt, "Horizontal", getattr(getattr(Qt, "Orientation", Qt), "Horizontal", None))
    QT_ALIGN_RIGHT = getattr(Qt, "AlignRight", getattr(getattr(Qt, "AlignmentFlag", Qt), "AlignRight", None))
    QT_ALIGN_VCENTER = getattr(Qt, "AlignVCenter", getattr(getattr(Qt, "AlignmentFlag", Qt), "AlignVCenter", None))

    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    from basic.module1 import ecef_to_blh, parse_nav_file
    from basic.module3 import ECEF
    from basic.module4 import (
        AnalysisSummary,
        build_linear_receiver_trajectory,
        calculate_summary,
        plot_results,
        run_continuous_positioning,
    )
    from basic.module5 import (
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


def _interpolate_trajectory_point(
    points: List[Tuple[float, float, float, float]], dt: float
) -> Tuple[float, float, float]:
    """对轨迹点列表按 time_offset_s 做线性插值。

    points: [(time_offset_s, dx, dy, dz), ...]，已按 time_offset_s 递增排序。
    返回 (dx, dy, dz)。
    """
    if not points:
        return 0.0, 0.0, 0.0
    if dt <= points[0][0]:
        return points[0][1], points[0][2], points[0][3]
    if dt >= points[-1][0]:
        return points[-1][1], points[-1][2], points[-1][3]
    for i in range(len(points) - 1):
        t0, x0, y0, z0 = points[i]
        t1, x1, y1, z1 = points[i + 1]
        if t0 <= dt <= t1:
            if abs(t1 - t0) < 1e-12:
                return x0, y0, z0
            ratio = (dt - t0) / (t1 - t0)
            return (
                x0 + ratio * (x1 - x0),
                y0 + ratio * (y1 - y0),
                z0 + ratio * (z1 - z0),
            )
    return points[-1][1], points[-1][2], points[-1][3]


def _load_trajectory_csv(
    csv_path: str,
    initial_position: ECEF,
    start_time: datetime,
) -> Tuple[Optional[Callable[[datetime], ECEF]], str]:
    """加载轨迹 CSV 文件，返回 (trajectory_func, message)。

    支持格式 A：time_offset_s,dx,dy,dz
    支持格式 B：epoch,true_X,true_Y,true_Z
    """
    try:
        with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return None, "CSV 文件为空"

        headers = [h.strip().lower() for h in rows[0].keys()]

        # 检测格式 B
        if "true_x" in headers and "true_y" in headers and "true_z" in headers:
            # 格式 B：epoch,true_X,true_Y,true_Z
            # 尝试解析 epoch 列
            epochs_and_positions: List[Tuple[datetime, ECEF]] = []
            for row in rows:
                epoch_val = row.get("epoch", "").strip()
                if not epoch_val:
                    continue
                try:
                    # 尝试 ISO 格式
                    epoch_dt = datetime.fromisoformat(epoch_val.replace(" ", "T").replace("Z", "+00:00"))
                except ValueError:
                    # 尝试自定义格式
                    try:
                        epoch_dt = datetime.strptime(epoch_val, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        continue
                tx = float(row["true_X"])
                ty = float(row["true_Y"])
                tz = float(row["true_Z"])
                epochs_and_positions.append((epoch_dt, (tx, ty, tz)))
            if not epochs_and_positions:
                return None, "未能解析格式 B 中的任何有效历元"
            epochs_and_positions.sort(key=lambda x: x[0])

            def traj_b(epoch_time: datetime) -> ECEF:
                if epoch_time <= epochs_and_positions[0][0]:
                    return epochs_and_positions[0][1]
                if epoch_time >= epochs_and_positions[-1][0]:
                    return epochs_and_positions[-1][1]
                for i in range(len(epochs_and_positions) - 1):
                    t0, p0 = epochs_and_positions[i]
                    t1, p1 = epochs_and_positions[i + 1]
                    if t0 <= epoch_time <= t1:
                        dt_total = (t1 - t0).total_seconds()
                        if dt_total <= 0:
                            return p0
                        dt = (epoch_time - t0).total_seconds()
                        ratio = dt / dt_total
                        return (
                            p0[0] + ratio * (p1[0] - p0[0]),
                            p0[1] + ratio * (p1[1] - p0[1]),
                            p0[2] + ratio * (p1[2] - p0[2]),
                        )
                return epochs_and_positions[-1][1]

            return traj_b, f"格式 B：已加载 {len(epochs_and_positions)} 个历元点"

        # 检测格式 A
        if "time_offset_s" in headers:
            points: List[Tuple[float, float, float, float]] = []
            for row in rows:
                t = float(row["time_offset_s"])
                dx = float(row.get("dx", 0.0))
                dy = float(row.get("dy", 0.0))
                dz = float(row.get("dz", 0.0))
                points.append((t, dx, dy, dz))
            points.sort(key=lambda x: x[0])

            def traj_a(epoch_time: datetime) -> ECEF:
                dt = (epoch_time - start_time).total_seconds()
                odx, ody, odz = _interpolate_trajectory_point(points, dt)
                return (
                    initial_position[0] + odx,
                    initial_position[1] + ody,
                    initial_position[2] + odz,
                )

            return traj_a, f"格式 A：已加载 {len(points)} 个轨迹点"

        return None, "未能识别 CSV 格式（需要 time_offset_s,dx,dy,dz 或 epoch,true_X,true_Y,true_Z）"
    except Exception as exc:
        return None, f"加载 CSV 失败：{exc}"


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
            trajectory_mode: str = "静态接收机",
            velocity_mps: Optional[Tuple[float, float, float]] = None,
            trajectory_points: Optional[List[Tuple[float, float, float, float]]] = None,
            trajectory_csv_path: Optional[str] = None,
            receiver_initial_approx: Optional[ECEF] = None,
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
            self.trajectory_mode = trajectory_mode
            self.velocity_mps = velocity_mps
            self.trajectory_points = trajectory_points
            self.trajectory_csv_path = trajectory_csv_path
            self.receiver_initial_approx = receiver_initial_approx

        def run(self) -> None:
            try:
                self.log_message.emit("开始连续定位解算...")

                receiver_trajectory: Optional[Callable[[datetime], ECEF]] = None

                if self.trajectory_mode == "匀速直线运动" and self.velocity_mps is not None:
                    receiver_trajectory = build_linear_receiver_trajectory(
                        self.start_time,
                        self.receiver_true_position,
                        self.velocity_mps,
                    )
                    self.log_message.emit(
                        f"动态轨迹：匀速直线运动，V=({self.velocity_mps[0]}, {self.velocity_mps[1]}, {self.velocity_mps[2]}) m/s"
                    )
                elif self.trajectory_mode == "表格折线轨迹" and self.trajectory_points:
                    x0, y0, z0 = self.receiver_true_position
                    points = list(self.trajectory_points)

                    def table_traj(epoch_time: datetime) -> ECEF:
                        dt = (epoch_time - self.start_time).total_seconds()
                        dx, dy, dz = _interpolate_trajectory_point(points, dt)
                        return (x0 + dx, y0 + dy, z0 + dz)

                    receiver_trajectory = table_traj
                    self.log_message.emit(f"动态轨迹：表格折线轨迹，共 {len(points)} 个控制点")
                elif self.trajectory_mode == "CSV轨迹文件" and self.trajectory_csv_path:
                    traj_func, msg = _load_trajectory_csv(
                        self.trajectory_csv_path,
                        self.receiver_true_position,
                        self.start_time,
                    )
                    if traj_func is None:
                        self.failed.emit(msg)
                        return
                    receiver_trajectory = traj_func
                    self.log_message.emit(f"动态轨迹：{msg}")
                else:
                    self.log_message.emit("静态接收机模式")

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
                    receiver_trajectory=receiver_trajectory,
                    receiver_initial_approx=self.receiver_initial_approx,
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
            "真值X",
            "真值Y",
            "真值Z",
            "解算X",
            "解算Y",
            "解算Z",
            "纬度",
            "经度",
            "误差(m)",
            "迭代",
        ]

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(f"北斗 RINEX 连续定位可视化 - {QT_BINDING}")
            self.resize(1400, 900)

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
            panel.setMinimumWidth(340)
            panel.setMaximumWidth(440)
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

            # 接收机轨迹设置
            trajectory_group = self._build_trajectory_group()
            layout.addWidget(trajectory_group)

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

        def _build_trajectory_group(self) -> QGroupBox:
            group = QGroupBox("接收机轨迹设置")
            layout = QVBoxLayout(group)

            self.trajectory_mode_combo = QComboBox()
            self.trajectory_mode_combo.addItems([
                "静态接收机",
                "匀速直线运动",
                "表格折线轨迹",
                "CSV轨迹文件",
            ])
            self.trajectory_mode_combo.currentTextChanged.connect(self._on_trajectory_mode_changed)
            layout.addWidget(QLabel("轨迹模式"))
            layout.addWidget(self.trajectory_mode_combo)

            self.trajectory_stack = QStackedWidget()

            # Page 0: 静态（空占位）
            static_page = QWidget()
            static_layout = QVBoxLayout(static_page)
            static_layout.addWidget(QLabel("静态模式：接收机真实位置不随时间变化。"))
            static_layout.addStretch(1)
            self.trajectory_stack.addWidget(static_page)

            # Page 1: 匀速直线运动
            linear_page = QWidget()
            linear_layout = QGridLayout(linear_page)
            self.velocity_x_spin = self._velocity_spin()
            self.velocity_y_spin = self._velocity_spin()
            self.velocity_z_spin = self._velocity_spin()
            linear_layout.addWidget(QLabel("Vx(m/s)"), 0, 0)
            linear_layout.addWidget(self.velocity_x_spin, 0, 1)
            linear_layout.addWidget(QLabel("Vy(m/s)"), 1, 0)
            linear_layout.addWidget(self.velocity_y_spin, 1, 1)
            linear_layout.addWidget(QLabel("Vz(m/s)"), 2, 0)
            linear_layout.addWidget(self.velocity_z_spin, 2, 1)
            self.trajectory_stack.addWidget(linear_page)

            # Page 2: 表格折线轨迹
            table_page = QWidget()
            table_layout = QVBoxLayout(table_page)
            self.trajectory_table = QTableWidget(2, 4)
            self.trajectory_table.setHorizontalHeaderLabels(["time_offset_s", "dx", "dy", "dz"])
            self.trajectory_table.horizontalHeader().setStretchLastSection(True)
            self.trajectory_table.setItem(0, 0, QTableWidgetItem("0"))
            self.trajectory_table.setItem(0, 1, QTableWidgetItem("0"))
            self.trajectory_table.setItem(0, 2, QTableWidgetItem("0"))
            self.trajectory_table.setItem(0, 3, QTableWidgetItem("0"))
            self.trajectory_table.setItem(1, 0, QTableWidgetItem("300"))
            self.trajectory_table.setItem(1, 1, QTableWidgetItem("100"))
            self.trajectory_table.setItem(1, 2, QTableWidgetItem("50"))
            self.trajectory_table.setItem(1, 3, QTableWidgetItem("20"))
            table_layout.addWidget(QLabel("轨迹控制点（time_offset_s 必须递增）："))
            table_layout.addWidget(self.trajectory_table)
            add_row_btn = QPushButton("添加一行")
            add_row_btn.clicked.connect(self._add_trajectory_table_row)
            table_layout.addWidget(add_row_btn)
            self.trajectory_stack.addWidget(table_page)

            # Page 3: CSV轨迹文件
            csv_page = QWidget()
            csv_layout = QVBoxLayout(csv_page)
            self.csv_path_edit = QLineEdit()
            self.csv_path_edit.setReadOnly(True)
            self.csv_path_edit.setPlaceholderText("未选择 CSV 文件")
            csv_btn = QPushButton("导入轨迹CSV")
            csv_btn.clicked.connect(self.import_trajectory_csv)
            csv_layout.addWidget(QLabel("支持格式 A：time_offset_s,dx,dy,dz\n支持格式 B：epoch,true_X,true_Y,true_Z"))
            csv_layout.addWidget(self.csv_path_edit)
            csv_layout.addWidget(csv_btn)
            csv_layout.addStretch(1)
            self.trajectory_stack.addWidget(csv_page)

            layout.addWidget(self.trajectory_stack)

            preview_btn = QPushButton("预览轨迹")
            preview_btn.clicked.connect(self.preview_trajectory)
            layout.addWidget(preview_btn)
            return group

        def _add_trajectory_table_row(self) -> None:
            row = self.trajectory_table.rowCount()
            self.trajectory_table.insertRow(row)
            for col in range(4):
                self.trajectory_table.setItem(row, col, QTableWidgetItem("0"))

        def _velocity_spin(self) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setRange(-10000.0, 10000.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.1)
            return spin

        def _on_trajectory_mode_changed(self, mode: str) -> None:
            index_map = {
                "静态接收机": 0,
                "匀速直线运动": 1,
                "表格折线轨迹": 2,
                "CSV轨迹文件": 3,
            }
            self.trajectory_stack.setCurrentIndex(index_map.get(mode, 0))

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
            self.tabs.addTab(self._build_preview_tab(), "轨迹预览")
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
            self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
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

        def _build_preview_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            self.preview_canvas = MplCanvas(width=6, height=5)
            layout.addWidget(self.preview_canvas, 1)
            hint = QLabel("点击左侧‘预览轨迹’按钮，可在此查看真实轨迹预览。")
            hint.setStyleSheet("color: #888;")
            layout.addWidget(hint)
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
            # 默认速度
            self.velocity_x_spin.setValue(0.5)
            self.velocity_y_spin.setValue(0.2)
            self.velocity_z_spin.setValue(0.1)
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
                nav_dir = Path("nav")
                default_dir = str(nav_dir) if nav_dir.exists() else str(Path.cwd())
                file_name, _ = QFileDialog.getOpenFileName(
                    self,
                    "选择 RINEX NAV 文件",
                    default_dir,
                    "RINEX NAV (*.26b_cnav *.cnav *.rnx *.nav *.26b *.??n *.*)",
                )
                if not file_name:
                    return
                selected_path = Path(file_name)

            try:
                self.log(f"正在解析：{selected_path}")
                nav_data, parse_info = parse_nav_file(selected_path)
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

        def import_trajectory_csv(self) -> None:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "导入轨迹 CSV",
                str(Path.cwd()),
                "CSV 文件 (*.csv)",
            )
            if not file_name:
                return
            self.csv_path_edit.setText(file_name)
            self.log(f"已选择轨迹 CSV：{file_name}")

        def _get_trajectory_table_points(self) -> List[Tuple[float, float, float, float]]:
            points: List[Tuple[float, float, float, float]] = []
            for row in range(self.trajectory_table.rowCount()):
                try:
                    t = float(self.trajectory_table.item(row, 0).text())
                    dx = float(self.trajectory_table.item(row, 1).text())
                    dy = float(self.trajectory_table.item(row, 2).text())
                    dz = float(self.trajectory_table.item(row, 3).text())
                    points.append((t, dx, dy, dz))
                except (AttributeError, ValueError):
                    continue
            points.sort(key=lambda x: x[0])
            return points

        def _validate_trajectory_params(self, mode: str) -> Tuple[bool, str]:
            if mode == "静态接收机":
                return True, ""
            if mode == "匀速直线运动":
                return True, ""
            if mode == "表格折线轨迹":
                points = self._get_trajectory_table_points()
                if len(points) < 1:
                    return False, "表格折线轨迹至少需要 1 个控制点。"
                # 检查 time_offset_s 是否递增
                for i in range(1, len(points)):
                    if points[i][0] <= points[i - 1][0]:
                        return False, f"表格第 {i + 1} 行的 time_offset_s 必须大于上一行。"
                return True, ""
            if mode == "CSV轨迹文件":
                path = self.csv_path_edit.text().strip()
                if not path:
                    return False, "请先导入轨迹 CSV 文件。"
                if not Path(path).exists():
                    return False, f"CSV 文件不存在：{path}"
                return True, ""
            return False, f"未知的轨迹模式：{mode}"

        def _build_receiver_trajectory(self, mode: str, initial_position: ECEF, start_time: datetime):
            """根据 GUI 参数构造 receiver_trajectory 和初始概略坐标。"""
            receiver_trajectory: Optional[Callable[[datetime], ECEF]] = None
            receiver_initial_approx: Optional[ECEF] = None

            if mode == "匀速直线运动":
                velocity = (
                    self.velocity_x_spin.value(),
                    self.velocity_y_spin.value(),
                    self.velocity_z_spin.value(),
                )
                receiver_trajectory = build_linear_receiver_trajectory(
                    start_time, initial_position, velocity
                )
                receiver_initial_approx = (
                    initial_position[0] + 50.0,
                    initial_position[1] - 50.0,
                    initial_position[2] + 30.0,
                )
            elif mode == "表格折线轨迹":
                points = self._get_trajectory_table_points()
                x0, y0, z0 = initial_position

                def table_traj(epoch_time: datetime) -> ECEF:
                    dt = (epoch_time - start_time).total_seconds()
                    dx, dy, dz = _interpolate_trajectory_point(points, dt)
                    return (x0 + dx, y0 + dy, z0 + dz)

                receiver_trajectory = table_traj
                receiver_initial_approx = (
                    initial_position[0] + 50.0,
                    initial_position[1] - 50.0,
                    initial_position[2] + 30.0,
                )
            elif mode == "CSV轨迹文件":
                csv_path = self.csv_path_edit.text().strip()
                traj_func, msg = _load_trajectory_csv(csv_path, initial_position, start_time)
                if traj_func is None:
                    raise ValueError(msg)
                receiver_trajectory = traj_func
                receiver_initial_approx = (
                    initial_position[0] + 50.0,
                    initial_position[1] - 50.0,
                    initial_position[2] + 30.0,
                )

            return receiver_trajectory, receiver_initial_approx

        def preview_trajectory(self) -> None:
            try:
                start_time = _qdatetime_to_datetime(self.start_edit.dateTime())
                end_time = _qdatetime_to_datetime(self.end_edit.dateTime())
                interval = self.interval_spin.value()
                mode = self.trajectory_mode_combo.currentText()
                initial_position = self._receiver_position_from_inputs()

                receiver_trajectory, _ = self._build_receiver_trajectory(
                    mode, initial_position, start_time
                )

                if receiver_trajectory is None:
                    # 静态模式：只有一个点
                    times = [start_time]
                    positions = [initial_position]
                else:
                    times = []
                    positions = []
                    current = start_time
                    while current <= end_time:
                        times.append(current)
                        positions.append(receiver_trajectory(current))
                        current += timedelta(seconds=interval)

                lats = []
                lons = []
                for pos in positions:
                    lat, lon, _ = ecef_to_blh(*pos)
                    lats.append(lat)
                    lons.append(lon)

                ax = self.preview_canvas.figure.clear()
                ax = self.preview_canvas.figure.add_subplot(111)
                ax.plot(lons, lats, marker="s", linewidth=1.5, linestyle="--", label="真实轨迹")
                ax.scatter([lons[0]], [lats[0]], marker="*", s=130, color="green", label="起点", zorder=5)
                ax.scatter([lons[-1]], [lats[-1]], marker="X", s=130, color="red", label="终点", zorder=5)
                ax.set_title("轨迹预览：真实接收机轨迹")
                ax.set_xlabel("经度 (deg)")
                ax.set_ylabel("纬度 (deg)")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best")
                self.preview_canvas.draw_idle()
                self.tabs.setCurrentIndex(2)  # 切换到轨迹预览页
                self.log(f"轨迹预览已生成：{len(positions)} 个点")
            except Exception as exc:
                QMessageBox.critical(self, "预览失败", str(exc))
                self.log(f"轨迹预览失败：{exc}")

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
            mode = self.trajectory_mode_combo.currentText()

            ok, msg = self._validate_trajectory_params(mode)
            if not ok:
                QMessageBox.warning(self, "轨迹参数错误", msg)
                return

            velocity_mps: Optional[Tuple[float, float, float]] = None
            trajectory_points: Optional[List[Tuple[float, float, float, float]]] = None
            trajectory_csv_path: Optional[str] = None
            receiver_initial_approx: Optional[ECEF] = None
            receiver_trajectory: Optional[Callable[[datetime], ECEF]] = None

            try:
                receiver_trajectory, receiver_initial_approx = self._build_receiver_trajectory(
                    mode, receiver_position, start_time
                )
            except ValueError as exc:
                QMessageBox.warning(self, "轨迹构造失败", str(exc))
                return

            if mode == "匀速直线运动":
                velocity_mps = (
                    self.velocity_x_spin.value(),
                    self.velocity_y_spin.value(),
                    self.velocity_z_spin.value(),
                )
            elif mode == "表格折线轨迹":
                trajectory_points = self._get_trajectory_table_points()
            elif mode == "CSV轨迹文件":
                trajectory_csv_path = self.csv_path_edit.text().strip()

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
                trajectory_mode=mode,
                velocity_mps=velocity_mps,
                trajectory_points=trajectory_points,
                trajectory_csv_path=trajectory_csv_path,
                receiver_initial_approx=receiver_initial_approx,
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
                _format_number(row.get("true_X")),
                _format_number(row.get("true_Y")),
                _format_number(row.get("true_Z")),
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
                if col >= 3:  # 数字列右对齐
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
            self.log("解算完成，CSV、统计文件和结果图已写入 outputs/basic 目录。")
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
                QMessageBox.warning(self, "缺少输出", "当前还没有可导出的 outputs/basic 目录。")
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
                "module4_continuous_position_results.csv",
                "module4_error_statistics.txt",
                "module4_error_curve.png",
                "module4_trajectory.png",
                "module4_true_vs_estimated_trajectory.png",
                "module4_satellite_dop_curve.png",
                "module5_system_test_report.txt",
                "module5_multi_scenario_summary.csv",
                "module5_multi_scenario_test_report.txt",
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

            # 真实轨迹
            has_true = all(
                self._is_finite(row.get("true_lat")) and self._is_finite(row.get("true_lon"))
                for row in success_rows
            )
            if has_true:
                true_lons = [float(row["true_lon"]) for row in success_rows]
                true_lats = [float(row["true_lat"]) for row in success_rows]
                ax.plot(true_lons, true_lats, marker="s", linewidth=1.5, linestyle="--", color="#16a34a", label="真实轨迹")
                ax.scatter([true_lons[0]], [true_lats[0]], marker="*", s=130, color="green", label="真实起点", zorder=5)
                ax.scatter([true_lons[-1]], [true_lats[-1]], marker="X", s=130, color="red", label="真实终点", zorder=5)

            # 解算轨迹（已播放部分）
            ax.plot(lons[: current_index + 1], lats[: current_index + 1], color="#2563eb", linewidth=1.5, label="解算轨迹")
            # 全部解算历元（淡色）
            ax.scatter(lons, lats, s=18, color="#94a3b8", label="全部解算历元")
            # 当前历元
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
        print("未检测到 Qt 绑定，请先安装 PyQt5。")
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
