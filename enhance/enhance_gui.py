"""
enhance_gui.py

提高部分（ML 误差补偿）图形化操作界面。

功能：
- 数据集构建、模型训练、误差补偿、评估可视化
- 数据集预览、训练指标、误差对比图、散点图、报告阅读
- 所有操作均在后台线程中执行，日志实时同步到 GUI 日志框

启动方式：
    python enhance/enhance_gui.py
"""

from __future__ import annotations

import io
import math
import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/enhance") / "matplotlib_cache"))

# ============================================================================
# PyQt5 兼容导入
# ============================================================================
try:
    from PyQt5.QtCore import QThread, Qt, pyqtSignal as Signal
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QDoubleSpinBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QSpinBox,
        QSplitter,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        QComboBox,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QAbstractItemView,
    )
    QT_BINDING = "PyQt5"
    QT_IMPORT_ERROR = None
except ModuleNotFoundError as pyqt5_err:
    QT_BINDING = ""
    QT_IMPORT_ERROR = pyqt5_err

if QT_IMPORT_ERROR is None:
    QT_ALIGN_LEFT = getattr(Qt, "AlignLeft", None)

    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    from enhance.dataset_builder import build_dataset
    from enhance.train_models import train_models
    from enhance.compensate import run_compensation
    from enhance.evaluate_models import evaluate_and_visualize
    from enhance.enhance_config import BASE_OUTPUT_DIR, FEATURE_COLUMNS


# ============================================================================
# Matplotlib 画布封装
# ============================================================================

class MplCanvas(FigureCanvas if QT_IMPORT_ERROR is None else object):
    """Matplotlib 图嵌入 Qt 画布。"""

    def __init__(self, width: int = 8, height: int = 5, dpi: int = 100) -> None:
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def clear(self) -> None:
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.draw()


# ============================================================================
# 日志重定向
# ============================================================================

class _LogRedirect(io.TextIOBase):
    """将 stdout 重定向到 Qt 信号。"""

    def __init__(self, signal) -> None:
        super().__init__()
        self._signal = signal

    def write(self, text: str) -> int:
        if text.strip():
            self._signal.emit(text.rstrip())
        return len(text)

    def flush(self) -> None:
        pass


# ============================================================================
# 后台工作线程
# ============================================================================

class EnhanceWorker(QThread):
    """在后台线程中执行 ML 全流程或单步骤。"""

    log_signal: Signal = Signal(str)
    progress_signal: Signal = Signal(int)
    finished_signal: Signal = Signal(dict)
    error_signal: Signal = Signal(str)

    def __init__(
        self,
        task: str,
        dataset_path: str = "",
        test_size: float = 0.3,
        random_state: int = 2026,
        enable_grid_search: bool = False,
        train_result: dict = None,
    ) -> None:
        super().__init__()
        self.task = task
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.random_state = random_state
        self.enable_grid_search = enable_grid_search
        self.train_result = train_result or {}

    def run(self) -> None:
        redirect = _LogRedirect(self.log_signal)
        old_stdout = sys.stdout
        sys.stdout = redirect
        result: dict = {}
        try:
            if self.task in ("build", "all"):
                self.log_signal.emit(">>> 开始构建数据集...")
                self.progress_signal.emit(10)
                dataset_path = build_dataset()
                result["dataset_path"] = str(dataset_path)
                self.log_signal.emit(f">>> 数据集已保存：{dataset_path}")
                self.progress_signal.emit(30)

            if self.task in ("train", "all"):
                if self.task == "train":
                    dataset_path = Path(self.dataset_path)
                else:
                    dataset_path = Path(result.get("dataset_path", ""))
                self.log_signal.emit(">>> 开始训练模型...")
                self.progress_signal.emit(40)
                train_result = train_models(
                    dataset_path,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    enable_grid_search=self.enable_grid_search,
                )
                result["train_result"] = train_result
                self.log_signal.emit(
                    f">>> 训练完成：训练集 {train_result['n_train']} 样本，"
                    f"测试集 {train_result['n_test']} 样本"
                )
                self.progress_signal.emit(60)

            if self.task in ("compensate", "all"):
                tr = result.get("train_result") or self.train_result
                self.log_signal.emit(">>> 开始误差补偿...")
                self.progress_signal.emit(70)
                prediction_paths = run_compensation(tr)
                result["prediction_paths"] = prediction_paths
                self.log_signal.emit(">>> 误差补偿完成")
                self.progress_signal.emit(80)

                self.log_signal.emit(">>> 开始评估与可视化...")
                eval_paths = evaluate_and_visualize(
                    prediction_paths,
                    n_train=tr["n_train"],
                    n_test=tr["n_test"],
                    feature_columns=FEATURE_COLUMNS,
                )
                result["eval_paths"] = eval_paths
                self.log_signal.emit(">>> 评估完成")
                self.progress_signal.emit(100)

            self.finished_signal.emit(result)
        except Exception as exc:
            self.error_signal.emit(str(exc))
        finally:
            sys.stdout = old_stdout


# ============================================================================
# 主窗口
# ============================================================================

class EnhanceMainWindow(QMainWindow):
    """ML 提高部分主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("北斗 SPP — ML 误差补偿（提高部分）")
        self.resize(1280, 800)

        self._train_result: dict = {}
        self._worker: EnhanceWorker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        # ── 左侧控制面板 ──────────────────────────────────────────
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        # [1] 数据集设置
        ds_group = QGroupBox("[1] 数据集设置")
        ds_layout = QVBoxLayout(ds_group)

        csv_row = QHBoxLayout()
        self._csv_edit = QLineEdit()
        self._csv_edit.setPlaceholderText("ml_dataset.csv 路径（留空则使用默认路径）")
        csv_btn = QPushButton("浏览")
        csv_btn.setFixedWidth(55)
        csv_btn.clicked.connect(self._browse_csv)
        csv_row.addWidget(self._csv_edit)
        csv_row.addWidget(csv_btn)
        ds_layout.addLayout(csv_row)

        ts_row = QHBoxLayout()
        ts_row.addWidget(QLabel("测试集比例:"))
        self._test_size_spin = QDoubleSpinBox()
        self._test_size_spin.setRange(0.1, 0.5)
        self._test_size_spin.setSingleStep(0.05)
        self._test_size_spin.setValue(0.3)
        ts_row.addWidget(self._test_size_spin)
        ts_row.addWidget(QLabel("随机种子:"))
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 99999)
        self._seed_spin.setValue(2026)
        ts_row.addWidget(self._seed_spin)
        ds_layout.addLayout(ts_row)

        left_layout.addWidget(ds_group)

        # [2] 模型设置
        model_group = QGroupBox("[2] 模型设置")
        model_layout = QVBoxLayout(model_group)

        model_sel_row = QHBoxLayout()
        model_sel_row.addWidget(QLabel("运行模型:"))
        self._model_combo = QComboBox()
        self._model_combo.addItems(["两个都跑", "线性回归", "随机森林"])
        model_sel_row.addWidget(self._model_combo)
        model_layout.addLayout(model_sel_row)

        rf_row = QHBoxLayout()
        rf_row.addWidget(QLabel("n_estimators:"))
        self._n_est_spin = QSpinBox()
        self._n_est_spin.setRange(50, 1000)
        self._n_est_spin.setSingleStep(50)
        self._n_est_spin.setValue(200)
        rf_row.addWidget(self._n_est_spin)
        rf_row.addWidget(QLabel("max_depth:"))
        self._max_depth_spin = QSpinBox()
        self._max_depth_spin.setRange(0, 50)
        self._max_depth_spin.setValue(10)
        self._max_depth_spin.setSpecialValueText("None")
        rf_row.addWidget(self._max_depth_spin)
        model_layout.addLayout(rf_row)

        self._grid_search_cb = QCheckBox("启用 GridSearchCV（较慢）")
        model_layout.addWidget(self._grid_search_cb)

        left_layout.addWidget(model_group)

        # [3] 操作按钮
        btn_group = QGroupBox("[3] 操作")
        btn_layout = QVBoxLayout(btn_group)

        self._build_btn = QPushButton("构建数据集")
        self._train_btn = QPushButton("训练模型")
        self._comp_btn = QPushButton("补偿与评估")
        self._all_btn = QPushButton("一键运行全部")
        for btn in (self._build_btn, self._train_btn, self._comp_btn, self._all_btn):
            btn.setMinimumHeight(30)
            btn_layout.addWidget(btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        btn_layout.addWidget(self._progress_bar)

        self._build_btn.clicked.connect(lambda: self._run_task("build"))
        self._train_btn.clicked.connect(lambda: self._run_task("train"))
        self._comp_btn.clicked.connect(lambda: self._run_task("compensate"))
        self._all_btn.clicked.connect(lambda: self._run_task("all"))

        left_layout.addWidget(btn_group)

        # [4] 日志
        log_group = QGroupBox("[4] 日志")
        log_layout = QVBoxLayout(log_group)
        self._log_edit = QTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setMinimumHeight(160)
        log_layout.addWidget(self._log_edit)
        left_layout.addWidget(log_group)

        left_layout.addStretch()
        root_layout.addWidget(left_panel)

        # ── 右侧标签页 ────────────────────────────────────────────
        self._tabs = QTabWidget()

        # Tab1: 数据集预览
        tab_dataset = QWidget()
        tab_ds_layout = QVBoxLayout(tab_dataset)
        self._ds_table = QTableWidget()
        self._ds_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._ds_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        tab_ds_layout.addWidget(QLabel("机器学习数据集预览（最多显示 100 行）"))
        tab_ds_layout.addWidget(self._ds_table)
        self._ds_stats_label = QLabel()
        tab_ds_layout.addWidget(self._ds_stats_label)
        self._tabs.addTab(tab_dataset, "数据集预览")

        # Tab2: 训练指标
        tab_metrics = QWidget()
        tab_metrics_layout = QVBoxLayout(tab_metrics)
        self._metrics_text = QTextEdit()
        self._metrics_text.setReadOnly(True)
        tab_metrics_layout.addWidget(self._metrics_text)
        self._tabs.addTab(tab_metrics, "训练指标")

        # Tab3: 误差对比图
        tab_curve = QWidget()
        tab_curve_layout = QVBoxLayout(tab_curve)
        self._curve_canvas = MplCanvas(width=9, height=5)
        tab_curve_layout.addWidget(self._curve_canvas)
        self._tabs.addTab(tab_curve, "误差对比图")

        # Tab4: 散点图
        tab_scatter = QWidget()
        tab_scatter_layout = QVBoxLayout(tab_scatter)
        self._scatter_canvas = MplCanvas(width=9, height=5)
        tab_scatter_layout.addWidget(self._scatter_canvas)
        self._tabs.addTab(tab_scatter, "散点图")

        # Tab5: 报告
        tab_report = QWidget()
        tab_report_layout = QVBoxLayout(tab_report)
        self._report_text = QTextEdit()
        self._report_text.setReadOnly(True)
        tab_report_layout.addWidget(self._report_text)
        self._tabs.addTab(tab_report, "报告")

        root_layout.addWidget(self._tabs, 1)

    # ── 槽函数 ────────────────────────────────────────────────────

    def _browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择数据集 CSV", "", "CSV 文件 (*.csv)")
        if path:
            self._csv_edit.setText(path)

    def _log(self, msg: str) -> None:
        self._log_edit.append(msg)
        self._log_edit.verticalScrollBar().setValue(self._log_edit.verticalScrollBar().maximum())

    def _set_buttons_enabled(self, enabled: bool) -> None:
        for btn in (self._build_btn, self._train_btn, self._comp_btn, self._all_btn):
            btn.setEnabled(enabled)

    def _run_task(self, task: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "提示", "当前有任务正在运行，请等待完成后再操作。")
            return

        csv_path = self._csv_edit.text().strip() or str(BASE_OUTPUT_DIR / "ml_dataset.csv")
        self._progress_bar.setValue(0)
        self._set_buttons_enabled(False)
        self._log(f"\n[{task}] 任务开始...")

        self._worker = EnhanceWorker(
            task=task,
            dataset_path=csv_path,
            test_size=self._test_size_spin.value(),
            random_state=self._seed_spin.value(),
            enable_grid_search=self._grid_search_cb.isChecked(),
            train_result=self._train_result,
        )
        self._worker.log_signal.connect(self._log)
        self._worker.progress_signal.connect(self._progress_bar.setValue)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.error_signal.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, result: dict) -> None:
        self._set_buttons_enabled(True)
        self._log("[完成] 任务执行完毕。")

        if "train_result" in result:
            self._train_result = result["train_result"]
            self._show_metrics(result["train_result"])

        if "dataset_path" in result:
            self._load_dataset_preview(Path(result["dataset_path"]))
        elif (BASE_OUTPUT_DIR / "ml_dataset.csv").exists():
            self._load_dataset_preview(BASE_OUTPUT_DIR / "ml_dataset.csv")

        if "eval_paths" in result:
            self._load_charts(result["eval_paths"])
            self._load_report()

    def _on_error(self, msg: str) -> None:
        self._set_buttons_enabled(True)
        self._log(f"[错误] {msg}")
        QMessageBox.critical(self, "运行错误", msg)

    def _load_dataset_preview(self, csv_path: Path) -> None:
        """在 Tab1 显示数据集前 100 行。"""
        if not csv_path.exists():
            return
        import csv as csv_mod
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv_mod.DictReader(f)
            rows = list(reader)
        if not rows:
            return
        headers = list(rows[0].keys())
        display_rows = rows[:100]
        self._ds_table.setColumnCount(len(headers))
        self._ds_table.setRowCount(len(display_rows))
        self._ds_table.setHorizontalHeaderLabels(headers)
        for r_idx, row in enumerate(display_rows):
            for c_idx, key in enumerate(headers):
                self._ds_table.setItem(r_idx, c_idx, QTableWidgetItem(str(row.get(key, ""))))
        self._ds_stats_label.setText(
            f"共 {len(rows)} 行，已显示前 {len(display_rows)} 行，{len(headers)} 列"
        )

    def _show_metrics(self, train_result: dict) -> None:
        """在 Tab2 显示训练指标。"""
        from sklearn.metrics import mean_squared_error, r2_score
        import numpy as np

        lines = ["模型训练指标\n" + "=" * 50]
        lines.append(f"样本总数：{train_result['n_train'] + train_result['n_test']}")
        lines.append(f"训练集：{train_result['n_train']} | 测试集：{train_result['n_test']}")

        X_test = train_result.get("X_test")
        y_test = train_result.get("y_test")
        if X_test is not None and y_test is not None:
            for name, key in [("线性回归", "lr_model"), ("随机森林", "rf_model")]:
                model = train_result.get(key)
                if model is not None:
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    rms = math.sqrt(float(np.mean(np.sum((y_pred - y_test) ** 2, axis=1))))
                    lines.append(f"\n{name}:")
                    lines.append(f"  MSE：{mse:.6f}")
                    lines.append(f"  R²：{r2:.6f}")
                    lines.append(f"  三维 RMS：{rms:.6f} m")
        self._metrics_text.setPlainText("\n".join(lines))

    def _load_charts(self, eval_paths: dict) -> None:
        """将 matplotlib 图加载到 Tab3 误差对比和 Tab4 散点图。"""
        import numpy as np
        import csv as csv_mod

        # Tab3: 误差对比折线
        self._curve_canvas.fig.clf()
        axes = self._curve_canvas.fig.subplots(1, 2, sharey=False)
        for ax, key, title in [
            (axes[0], "linear_regression", "线性回归误差对比"),
            (axes[1], "random_forest", "随机森林误差对比"),
        ]:
            csv_p = eval_paths.get(key) or eval_paths.get("summary_csv")
            if csv_p and Path(csv_p).exists():
                before, after = [], []
                with open(csv_p, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv_mod.DictReader(f)
                    for row in reader:
                        b = row.get("error_before", "")
                        a = row.get("error_after", "")
                        if b:
                            before.append(float(b))
                        if a:
                            after.append(float(a))
                if before:
                    ax.plot(before, label="补偿前", alpha=0.7, linewidth=1.2)
                if after:
                    ax.plot(after, label="补偿后", alpha=0.7, linewidth=1.2)
                ax.set_xlabel("样本索引")
                ax.set_ylabel("三维误差 (m)")
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
        self._curve_canvas.fig.tight_layout()
        self._curve_canvas.draw()

        # Tab4: 散点图（预测 vs 真实误差，取 error_x 轴）
        self._scatter_canvas.fig.clf()
        axes2 = self._scatter_canvas.fig.subplots(1, 2)
        for ax, key, title in [
            (axes2[0], "linear_regression", "线性回归"),
            (axes2[1], "random_forest", "随机森林"),
        ]:
            summary_p = eval_paths.get("summary_csv")
            if summary_p and Path(summary_p).exists():
                pred_err_x, true_err_x = [], []
                with open(summary_p, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv_mod.DictReader(f)
                    for row in reader:
                        if row.get("model") == key or key in str(row.get("model", "")):
                            pe = row.get("pred_error_x", "")
                            te = row.get("true_error_x", "")
                            if pe and te:
                                pred_err_x.append(float(pe))
                                true_err_x.append(float(te))
            if pred_err_x:
                ax.scatter(true_err_x, pred_err_x, alpha=0.5, s=20)
                lim = max(abs(min(true_err_x + pred_err_x)), abs(max(true_err_x + pred_err_x)), 1.0)
                ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1.2)
                ax.set_xlabel("真实误差 X (m)")
                ax.set_ylabel("预测误差 X (m)")
            else:
                ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{title}：预测 vs 真实 (error_x)")
            ax.grid(True, alpha=0.3)
        self._scatter_canvas.fig.tight_layout()
        self._scatter_canvas.draw()

    def _load_report(self) -> None:
        """在 Tab5 读取技术报告文本。"""
        report_path = BASE_OUTPUT_DIR / "ml_technical_report.txt"
        if report_path.exists():
            self._report_text.setPlainText(report_path.read_text(encoding="utf-8-sig"))
        else:
            stats_path = BASE_OUTPUT_DIR / "ml_compensation_statistics.txt"
            if stats_path.exists():
                self._report_text.setPlainText(stats_path.read_text(encoding="utf-8-sig"))


# ============================================================================
# 入口
# ============================================================================

def main() -> int:
    if QT_IMPORT_ERROR is not None:
        print(f"PyQt5 导入失败：{QT_IMPORT_ERROR}")
        print("请安装 PyQt5：pip install PyQt5")
        return 1
    app = QApplication(sys.argv)
    window = EnhanceMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
