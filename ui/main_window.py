"""
=============================================================================
主窗口 - 三个 Tab 页面
=============================================================================

三个 Tab:
  1. 单图复原   —— 打开/拖入图片、参数、进度、对比、指标、另存
  2. 批量评测   —— 目录选择、后台评测、表格、导出 CSV
  3. 关于/环境  —— 显示依赖与指标可用性，排查环境问题

视觉风格集中在 ui/theme.py，本文件只负责布局与 objectName。

后续可扩展:
  - 批量页加 pyqtgraph 柱状图对比不同方法
  - 记住上次使用的目录与参数（QSettings）
=============================================================================
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFontMetrics, QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT, ensure_dir
from core.image_utils import IMAGE_EXTS, list_images, load_image, save_image
from core.metrics import METRIC_LABELS, available_metrics, compute_metrics, format_value
from engine.evaluate import EvalRow, write_csv
from engine.inference import ModelCache
from ui.compare_view import CompareView
from ui.panels import ControlPanel, MetricPanel
from ui.theme import palette
from ui.worker import EvalWorker, RestoreWorker

APP_TITLE = "恶劣天气图像复原系统"
APP_SUBTITLE = "基于扩散模型的雨 / 雾 / 雪图像复原与定量评测"
SAMPLES_DIR = DATA_DIR / "samples"

#: 右侧参数栏宽度（固定）与其内部文本可用宽度
SIDEBAR_WIDTH = 336
SIDEBAR_TEXT_WIDTH = SIDEBAR_WIDTH - 36


# --------------------------------------------------------------------------- #
# 小工具：统一的卡片容器 / 分区标题
# --------------------------------------------------------------------------- #
def card(*, margins: int = 12, spacing: int = 10) -> tuple:
    """返回 (卡片Widget, 垂直布局)，统一圆角边框样式。"""
    box = QWidget()
    box.setObjectName("Card")
    layout = QVBoxLayout(box)
    layout.setContentsMargins(margins, margins, margins, margins)
    layout.setSpacing(spacing)
    return box, layout


def section_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("SectionTitle")
    return label


def hint_label(text: str, max_width: int = SIDEBAR_TEXT_WIDTH) -> QLabel:
    """自动换行的灰字提示。必须限宽，否则 wordWrap 的 sizeHint 会撑破固定宽侧栏。"""
    label = QLabel(text)
    label.setObjectName("Hint")
    label.setWordWrap(True)
    label.setMaximumWidth(max_width)
    label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    return label


def ghost_button(text: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setObjectName("GhostButton")
    btn.setCursor(Qt.PointingHandCursor)
    return btn


class SingleImagePage(QWidget):
    """Tab 1：单图复原。"""

    def __init__(self, cache: ModelCache, parent=None):
        super().__init__(parent)
        self.cache = cache
        self.worker: Optional[RestoreWorker] = None
        self.input_image: Optional[np.ndarray] = None
        self.output_image: Optional[np.ndarray] = None
        self.gt_image: Optional[np.ndarray] = None
        self.input_path: Optional[Path] = None

        self.compare = CompareView()
        self.control = ControlPanel()
        self.metric_panel = MetricPanel()
        self.progress = QProgressBar()
        self.status = QLabel("就绪 —— 打开一张退化图片，或点击『载入示例』")
        self.file_badge = QLabel("未载入图片")
        self.gt_badge = QLabel("无 GT")

        self.open_button = QPushButton("打开图片")
        self.sample_button = QPushButton("载入示例")
        self.gt_button = ghost_button("选择 GT")
        self.save_button = ghost_button("保存结果")
        self.mode_button = ghost_button("擦除 / 并排")
        self.export_button = ghost_button("导出对比图")
        self.weather_filter = QComboBox()
        self.weather_filter.addItems(["雨 (rain)", "雾 (haze)", "雪 (snow)", "真实 (real)"])
        self.weather_filter.setToolTip("选择要载入的天气类型示例")
        self.weather_filter.setMinimumHeight(28)
        self._sample_index = 0
        self._sample_files: List[Path] = []

        self._build()
        self._connect()
        self.setAcceptDrops(True)

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        # ---- 工具条 ----
        self.open_button.setObjectName("PrimaryButton")
        self.open_button.setCursor(Qt.PointingHandCursor)
        self.sample_button.setCursor(Qt.PointingHandCursor)
        self.sample_button.setToolTip("按当前选择的天气类型，循环载入示例图片")
        self.save_button.setEnabled(False)
        self.export_button.setToolTip("把当前对比视图存成图片，写论文/答辩配图直接用")

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        for b in (self.open_button, self.sample_button, self.gt_button):
            toolbar.addWidget(b)
        toolbar.addWidget(self._vline())
        toolbar.addWidget(QLabel("示例类型:"))
        toolbar.addWidget(self.weather_filter)
        toolbar.addWidget(self._vline())
        for b in (self.mode_button, self.export_button, self.save_button):
            toolbar.addWidget(b)
        toolbar.addStretch(1)
        for badge in (self.file_badge, self.gt_badge):
            badge.setObjectName("Badge")
            badge.setMaximumWidth(240)
            toolbar.addWidget(badge)

        toolbar_card, toolbar_layout = card(margins=10, spacing=0)
        toolbar_layout.addLayout(toolbar)

        # ---- 画布 + 进度 ----
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        self.status.setObjectName("StatusText")

        canvas_card, canvas_layout = card(margins=10, spacing=8)
        canvas_layout.addWidget(self.compare, stretch=1)
        canvas_layout.addWidget(self.progress)
        canvas_layout.addWidget(self.status)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(10)
        left.addWidget(toolbar_card)
        left.addWidget(canvas_card, stretch=1)
        left_widget = QWidget()
        left_widget.setLayout(left)

        # ---- 右侧参数栏 ----
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(10)
        right.addWidget(self.control)
        right.addWidget(self.metric_panel)
        right.addWidget(
            hint_label(
                "小贴士\n"
                "• 双击画面切换 擦除 / 并排 对比\n"
                "• 拖动中缝可逐像素比对雨纹与雪点\n"
                "• 样本目录下的 gt/ 会自动关联\n"
                "• 导出对比图可直接用于论文配图"
            )
        )
        right.addStretch(1)
        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setFixedWidth(SIDEBAR_WIDTH)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setHandleWidth(8)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.addWidget(splitter)

    @staticmethod
    def _vline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFixedWidth(1)
        line.setStyleSheet(f"background-color: {palette()['border']}; border: none;")
        return line

    def _connect(self) -> None:
        self.open_button.clicked.connect(self.open_image)
        self.sample_button.clicked.connect(self.load_next_sample)
        self.gt_button.clicked.connect(self.open_gt)
        self.save_button.clicked.connect(self.save_result)
        self.mode_button.clicked.connect(self._toggle_mode)
        self.export_button.clicked.connect(self.export_compare)
        self.control.start_requested.connect(self.start_restore)
        self.control.cancel_requested.connect(self.cancel_restore)
        self.weather_filter.currentIndexChanged.connect(self._on_weather_filter_changed)

    def _toggle_mode(self) -> None:
        self.compare.toggle_mode()
        self.status.setText(f"对比模式：{self.compare.mode_text()}")

    # ------------------------------------------------------------------ #
    # 拖拽支持
    # ------------------------------------------------------------------ #
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in IMAGE_EXTS:
                self.load_input(path)
                break

    # ------------------------------------------------------------------ #
    def open_image(self) -> None:
        filter_str = "图片 (" + " ".join(f"*{e}" for e in IMAGE_EXTS) + ")"
        start_dir = SAMPLES_DIR if SAMPLES_DIR.is_dir() else PROJECT_ROOT
        path, _ = QFileDialog.getOpenFileName(self, "选择退化图片", str(start_dir), filter_str)
        if path:
            self.load_input(Path(path))

    def _get_sample_files(self) -> List[Path]:
        """根据天气类型筛选获取示例文件列表。"""
        weather_map = {
            0: ("rain",),
            1: ("haze",),
            2: ("snow",),
            3: ("real",),
        }
        weathers = weather_map.get(self.weather_filter.currentIndex(), ("rain",))
        files = []
        for weather in weathers:
            files.extend(list_images(SAMPLES_DIR / weather / "input"))
        return files

    def load_next_sample(self) -> None:
        """循环载入 data/samples 里的示例图（按天气类型筛选），演示时最方便。"""
        files = self._get_sample_files()
        if not files:
            QMessageBox.information(
                self,
                "没有示例数据",
                "请先在项目目录执行:\n\n    python scripts/make_samples.py\n\n"
                "会自动生成雨/雾/雪 的合成样本与对应 GT。",
            )
            return
        self.load_input(files[self._sample_index % len(files)])
        self._sample_index += 1

    def _on_weather_filter_changed(self) -> None:
        """天气类型切换时重置索引，从该类型第一张开始。"""
        self._sample_index = 0
        files = self._get_sample_files()
        if files:
            self.status.setText(f"已切换到: {self.weather_filter.currentText()}，共 {len(files)} 张示例")

    def open_gt(self) -> None:
        filter_str = "图片 (" + " ".join(f"*{e}" for e in IMAGE_EXTS) + ")"
        path, _ = QFileDialog.getOpenFileName(self, "选择对应的清晰图 GT", str(PROJECT_ROOT), filter_str)
        if not path:
            return
        try:
            self.gt_image = load_image(path)
            self._set_gt_badge(Path(path).name)
            self.status.setText(f"已载入 GT: {Path(path).name}，可计算 PSNR/SSIM")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "读取失败", str(exc))

    def load_input(self, path: Path) -> None:
        try:
            self.input_image = load_image(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "读取失败", str(exc))
            return
        self.input_path = path
        self.output_image = None
        self.gt_image = None
        self.compare.set_images(self.input_image, None)
        self.metric_panel.clear()
        self.save_button.setEnabled(False)
        self.progress.setValue(0)
        h, w = self.input_image.shape[:2]

        self._set_badge(self.file_badge, f"{path.name}  {w}×{h}", ok=True)

        # 自动寻找同名 GT（目录结构 .../input/x.png <-> .../gt/x.png）
        gt_hint = ""
        gt_path = self._guess_gt_path(path)
        if gt_path is not None:
            try:
                self.gt_image = load_image(gt_path)
                self._set_gt_badge(f"{gt_path.parent.name}/{gt_path.name}")
                gt_hint = f"  已自动关联 GT: {gt_path.parent.name}/{gt_path.name}"
            except Exception:  # noqa: BLE001
                self.gt_image = None
        if self.gt_image is None:
            self._set_gt_badge(None)
        self.status.setText(f"已载入 {path.name}  ({w}×{h}){gt_hint}")

    def _set_gt_badge(self, name: Optional[str]) -> None:
        if name:
            self._set_badge(self.gt_badge, f"GT: {name}", ok=True)
        else:
            self._set_badge(self.gt_badge, "无 GT（PSNR/SSIM 不可算）", ok=False)

    @staticmethod
    def _set_badge(label: QLabel, text: str, ok: bool) -> None:
        """设徽章文字，过长自动中间省略，避免长文件名撑破工具条。"""
        metrics = QFontMetrics(label.font())
        label.setText(metrics.elidedText(text, Qt.ElideMiddle, label.maximumWidth() - 26))
        label.setToolTip(text)
        label.setObjectName("BadgeOk" if ok else "Badge")
        label.style().unpolish(label)
        label.style().polish(label)

    @staticmethod
    def _guess_gt_path(path: Path) -> Optional[Path]:
        """把 .../input/xxx.png 映射到 .../gt/xxx.*，找不到返回 None。"""
        for gt_name in ("gt", "target", "clean", "norain"):
            gt_dir = path.parent.parent / gt_name
            if not gt_dir.is_dir():
                continue
            for ext in IMAGE_EXTS:
                candidate = gt_dir / f"{path.stem}{ext}"
                if candidate.is_file():
                    return candidate
        return None

    def save_result(self) -> None:
        if self.output_image is None:
            return
        default = str(ensure_dir(OUTPUT_DIR) / f"restored_{(self.input_path or Path('out')).name}")
        path, _ = QFileDialog.getSaveFileName(self, "保存复原结果", default, "PNG (*.png);;JPEG (*.jpg)")
        if path:
            save_image(path, self.output_image)
            self.status.setText(f"已保存: {path}")

    def export_compare(self) -> None:
        """把当前对比视图导出成图片，答辩配图直接用。"""
        if self.input_image is None:
            QMessageBox.information(self, "提示", "请先打开一张图片")
            return
        stem = (self.input_path or Path("compare")).stem
        default = str(ensure_dir(OUTPUT_DIR) / f"compare_{stem}.png")
        path, _ = QFileDialog.getSaveFileName(self, "导出对比视图", default, "PNG (*.png)")
        if path:
            self.compare.export_pixmap().save(path)
            self.status.setText(f"对比图已导出: {path}")

    # ------------------------------------------------------------------ #
    def start_restore(self) -> None:
        if self.input_image is None:
            QMessageBox.information(self, "提示", "请先打开一张退化图片")
            return
        weather = self.control.current_weather()
        if not weather:
            QMessageBox.warning(self, "提示", "没有可用的配置，请检查 configs/ 目录")
            return

        self.control.set_running(True)
        self.progress.setValue(0)
        self.status.setText("正在加载模型...")

        self.worker = RestoreWorker(
            cache=self.cache,
            weather=weather,
            image=self.input_image,
            overrides=self.control.overrides(),
            emit_preview=True,
        )
        self.worker.loading.connect(lambda: self.status.setText("正在加载模型/权重..."))
        self.worker.progress.connect(self._on_progress)
        self.worker.preview.connect(self.compare.set_after)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.start()

    def cancel_restore(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.status.setText("正在取消...")

    # ------------------------------------------------------------------ #
    def _on_progress(self, step: int, total: int) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(step)
        self.status.setText(f"采样中  {step}/{total}")

    def _on_finished(self, result) -> None:
        self.output_image = result.image
        self.compare.set_after(result.image)
        self.metric_panel.update_runtime(result.elapsed, result.device)
        metrics = compute_metrics(
            result.image,
            self.gt_image,
            metrics=["psnr", "ssim"] if self.gt_image is not None else [],
        )
        self.metric_panel.update_metrics(metrics)
        # 注: NIQE/BRISQUE 计算较慢，可考虑放到单独线程后再接进来
        self.control.set_running(False)
        self.save_button.setEnabled(True)
        self.status.setText(f"复原完成，耗时 {result.elapsed:.2f}s（{result.device}）")

    def _on_failed(self, message: str) -> None:
        self.control.set_running(False)
        self.status.setText("复原失败")
        QMessageBox.critical(self, "复原失败", message)

    def _on_cancelled(self) -> None:
        self.control.set_running(False)
        self.progress.setValue(0)
        self.status.setText("已取消")


class BatchEvalPage(QWidget):
    """Tab 2：批量评测。"""

    METRIC_KEYS = ["psnr", "ssim", "niqe", "brisque"]

    def __init__(self, cache: ModelCache, parent=None):
        super().__init__(parent)
        self.cache = cache
        self.worker: Optional[EvalWorker] = None
        self.rows: List[EvalRow] = []

        self.control = ControlPanel()
        self.input_edit = QLineEdit()
        self.gt_edit = QLineEdit()
        self.out_edit = QLineEdit(str(OUTPUT_DIR / "batch"))
        # 有示例数据时预填路径，演示时直接点"开始复原"就能跑
        if (SAMPLES_DIR / "rain" / "input").is_dir():
            self.input_edit.setText(str(SAMPLES_DIR / "rain" / "input"))
            self.gt_edit.setText(str(SAMPLES_DIR / "rain" / "gt"))
        self.limit_spin = QSpinBox()
        self.metric_boxes: Dict[str, QCheckBox] = {}
        self.table = QTableWidget()
        self.progress = QProgressBar()
        self.status = QLabel("就绪")
        self.summary_label = QLabel("均值：-")
        self.export_button = QPushButton("导出 CSV")

        self._build()
        self._connect()

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        for edit, tip in (
            (self.input_edit, "退化图所在目录"),
            (self.gt_edit, "清晰图目录，留空则只算无参考指标"),
            (self.out_edit, "复原结果保存目录"),
        ):
            edit.setMinimumHeight(30)
            edit.setPlaceholderText(tip)
            edit.setToolTip(edit.text() or tip)
            edit.setCursorPosition(0)  # 长路径默认显示开头而不是末尾

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(9)
        grid.setColumnStretch(1, 1)
        for row, (text, edit) in enumerate(
            (("退化图目录", self.input_edit), ("GT 目录（可选）", self.gt_edit), ("结果输出目录", self.out_edit))
        ):
            key = QLabel(text)
            key.setObjectName("MetricKey")
            grid.addWidget(key, row, 0)
            grid.addWidget(edit, row, 1)
            grid.addWidget(self._browse_button(edit), row, 2)

        path_card, path_layout = card(margins=12, spacing=10)
        path_layout.addWidget(section_title("数据目录"))
        path_layout.addLayout(grid)

        # ---- 指标勾选 ----
        self.limit_spin.setRange(0, 100000)
        self.limit_spin.setValue(20)
        self.limit_spin.setMinimumHeight(28)
        self.limit_spin.setToolTip("只跑前 N 张，0 表示全部")

        metric_row = QHBoxLayout()
        metric_row.setSpacing(12)
        avail = available_metrics()
        for key in self.METRIC_KEYS:
            box = QCheckBox(METRIC_LABELS.get(key, key))
            usable = avail.get(key, True)
            box.setChecked(key in ("psnr", "ssim") and usable)
            box.setEnabled(usable)
            if not usable:
                box.setToolTip("缺少依赖：pip install pyiqa")
            self.metric_boxes[key] = box
            metric_row.addWidget(box)
        metric_row.addStretch(1)
        limit_key = QLabel("限制张数")
        limit_key.setObjectName("MetricKey")
        metric_row.addWidget(limit_key)
        metric_row.addWidget(self.limit_spin)

        metric_card, metric_layout = card(margins=12, spacing=10)
        metric_layout.addWidget(section_title("评测指标"))
        metric_layout.addLayout(metric_row)

        # ---- 结果表 ----
        self.table.setColumnCount(4 + len(self.METRIC_KEYS))
        self.table.setHorizontalHeaderLabels(
            ["文件名", "天气", "方法", "耗时(s)"] + [METRIC_LABELS.get(k, k) for k in self.METRIC_KEYS]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._show_empty_hint()

        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        self.status.setObjectName("StatusText")
        self.summary_label.setObjectName("SummaryText")
        self.export_button.setObjectName("GhostButton")
        self.export_button.setCursor(Qt.PointingHandCursor)

        footer = QHBoxLayout()
        footer.setSpacing(10)
        footer.addWidget(self.status, stretch=1)
        footer.addWidget(self.export_button)

        table_card, table_layout = card(margins=12, spacing=10)
        table_layout.addWidget(section_title("评测结果"))
        table_layout.addWidget(self.table, stretch=1)
        table_layout.addWidget(self.progress)
        table_layout.addWidget(self.summary_label)
        table_layout.addLayout(footer)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(10)
        left.addWidget(path_card)
        left.addWidget(metric_card)
        left.addWidget(table_card, stretch=1)
        left_widget = QWidget()
        left_widget.setLayout(left)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        right_layout.addWidget(self.control)
        right_layout.addWidget(
            hint_label(
                "批量流程：\n"
                "1. 指定退化图目录（可选 GT 目录）\n"
                "2. 勾选指标、设置张数上限\n"
                "3. 点击『开始复原』，结果实时入表\n"
                "4. 完成后导出 CSV 用于论文表格"
            )
        )
        right_layout.addStretch(1)
        right_widget.setFixedWidth(SIDEBAR_WIDTH)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setHandleWidth(8)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.addWidget(splitter)

    def _show_empty_hint(self) -> None:
        """空表时占一行居中提示，避免一大片空白。"""
        self.table.setRowCount(1)
        self.table.setSpan(0, 0, 1, self.table.columnCount())
        item = QTableWidgetItem("尚未评测 —— 确认上方目录后，点击右侧『开始复原』")
        item.setTextAlignment(Qt.AlignCenter)
        item.setForeground(QColor(palette()["text_faint"]))
        self.table.setItem(0, 0, item)
        self.table.setRowHeight(0, 64)

    def _clear_table(self) -> None:
        self.table.clearSpans()
        self.table.setRowCount(0)

    def _browse_button(self, target: QLineEdit) -> QPushButton:
        btn = ghost_button("浏览")

        def choose() -> None:
            start = target.text().strip() or str(PROJECT_ROOT)
            path = QFileDialog.getExistingDirectory(self, "选择目录", start)
            if path:
                target.setText(path)

        btn.clicked.connect(choose)
        return btn

    def _connect(self) -> None:
        self.control.start_requested.connect(self.start_eval)
        self.control.cancel_requested.connect(self.cancel_eval)
        self.export_button.clicked.connect(self.export_csv)

    # ------------------------------------------------------------------ #
    def start_eval(self) -> None:
        input_dir = self.input_edit.text().strip()
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.information(self, "提示", "请先选择退化图目录")
            return
        weather = self.control.current_weather()
        metrics = [k for k, box in self.metric_boxes.items() if box.isChecked()]

        self.rows = []
        self._clear_table()
        self.progress.setValue(0)
        self.control.set_running(True)
        self.status.setText("评测中...")

        self.worker = EvalWorker(
            cache=self.cache,
            weather=weather,
            input_dir=input_dir,
            gt_dir=self.gt_edit.text().strip() or None,
            output_dir=self.out_edit.text().strip() or None,
            metrics=metrics,
            limit=self.limit_spin.value() or None,
            overrides=self.control.overrides(),
        )
        self.worker.item_done.connect(self._on_item)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.start()

    def cancel_eval(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.status.setText("正在取消...")

    def export_csv(self) -> None:
        if not self.rows:
            QMessageBox.information(self, "提示", "还没有评测结果")
            return
        default = str(ensure_dir(OUTPUT_DIR) / "eval_result.csv")
        path, _ = QFileDialog.getSaveFileName(self, "导出 CSV", default, "CSV (*.csv)")
        if path:
            write_csv(self.rows, path)
            self.status.setText(f"已导出: {path}")

    # ------------------------------------------------------------------ #
    def _on_item(self, done: int, total: int, row: EvalRow) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.rows.append(row)
        r = self.table.rowCount()
        self.table.insertRow(r)
        values = [row.filename, row.weather, row.restorer, f"{row.elapsed:.2f}"]
        for key in self.METRIC_KEYS:
            values.append(format_value(row.metrics.get(key), 2 if key == "psnr" else 4))
        for c, v in enumerate(values):
            item = QTableWidgetItem(str(v))
            if c >= 3:
                item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, c, item)
        if row.error:
            self.table.item(r, 0).setToolTip(row.error)
        self.table.scrollToBottom()
        self.status.setText(f"{done}/{total}   {row.filename}")

    def _on_finished(self, rows, summary) -> None:
        self.control.set_running(False)
        parts = [
            f"{METRIC_LABELS.get(k, k)} = {format_value(v, 2 if k == 'psnr' else 4)}"
            for k, v in (summary or {}).items()
            if k in METRIC_LABELS
        ]
        avg_time = (summary or {}).get("elapsed_s")
        if avg_time:
            parts.append(f"平均耗时 = {avg_time:.2f}s")
        self.summary_label.setText("均值：" + ("     ".join(parts) if parts else "-"))
        self.status.setText(f"评测完成，共 {len(rows)} 张")
        # 后续可接 pyqtgraph 柱状图，把不同方法的均值指标画出来

    def _on_failed(self, message: str) -> None:
        self.control.set_running(False)
        self.status.setText("评测失败")
        QMessageBox.critical(self, "评测失败", message)

    def _on_cancelled(self) -> None:
        self.control.set_running(False)
        self.status.setText("已取消")


class AboutPage(QWidget):
    """Tab 3：环境自检，排查"指标算不出来/权重找不到"这类问题。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        text = QTextEdit()
        text.setObjectName("MonoText")
        text.setReadOnly(True)
        text.setHtml(self._collect_html())

        wrapper, wrapper_layout = card(margins=14, spacing=10)
        wrapper_layout.addWidget(section_title("环境与依赖自检"))
        wrapper_layout.addWidget(
            hint_label("此页用于排查『指标算不出来 / 权重找不到 / GPU 没用上』这类问题。")
        )
        wrapper_layout.addWidget(text, stretch=1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.addWidget(wrapper)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _row(name: str, value: str, ok: Optional[bool] = None) -> str:
        c = palette()
        tone = {True: c["success"], False: c["danger"], None: c["text_dim"]}[ok]
        return (
            f"<tr><td style='padding:2px 18px 2px 0;color:{c['text_dim']};'>{name}</td>"
            f"<td style='padding:2px 0;color:{tone};'>{value}</td></tr>"
        )

    @classmethod
    def _collect_html(cls) -> str:
        c = palette()
        blocks: List[str] = [
            f"<div style='color:{c['text']};font-size:14px;font-weight:600;'>{APP_TITLE}</div>",
            f"<div style='color:{c['text_faint']};'>{APP_SUBTITLE}</div><br>",
            "<table>",
            cls._row("项目根目录", str(PROJECT_ROOT)),
            cls._row("强制设备 AWR_DEVICE", os.environ.get("AWR_DEVICE", "未设置（自动选择）")),
            "</table><br>",
            f"<div style='color:{c['primary']};font-weight:600;'>依赖状态</div><table>",
        ]
        for module in ("numpy", "PIL", "yaml", "PyQt5", "torch", "skimage", "pyiqa", "lpips", "pytorch_fid"):
            try:
                m = __import__(module)
                blocks.append(cls._row(module, str(getattr(m, "__version__", "已安装")), True))
            except ImportError:
                blocks.append(cls._row(module, "未安装", False))
        blocks.append("</table><br>")

        blocks.append(f"<div style='color:{c['primary']};font-weight:600;'>指标可用性</div><table>")
        for name, ok in available_metrics().items():
            blocks.append(cls._row(name.upper(), "可用" if ok else "不可用（缺依赖）", bool(ok)))
        blocks.append("</table><br>")

        blocks.append(f"<div style='color:{c['primary']};font-weight:600;'>计算设备</div><table>")
        try:
            import torch  # noqa: PLC0415

            cuda = torch.cuda.is_available()
            blocks.append(cls._row("CUDA", "可用" if cuda else "不可用（将使用 CPU）", cuda))
            if cuda:
                blocks.append(cls._row("GPU", torch.cuda.get_device_name(0), True))
        except ImportError:
            blocks.append(cls._row("torch", "未安装", False))
        blocks.append("</table><br>")

        blocks.append(f"<div style='color:{c['primary']};font-weight:600;'>权重目录</div><table>")
        wdir = PROJECT_ROOT / "weights"
        if wdir.is_dir():
            files = [p.name for p in wdir.iterdir() if p.is_file() and p.suffix != ".md"]
            if files:
                blocks.append(cls._row(str(wdir.name) + "/", ", ".join(files), True))
            else:
                blocks.append(cls._row("weights/", "空 —— 可先运行 scripts/download_weights.py", False))
        else:
            blocks.append(cls._row("weights/", "目录不存在", False))
        blocks.append("</table><br>")

        blocks.append(
            f"<div style='color:{c['primary']};font-weight:600;'>常用命令</div>"
            f"<div style='color:{c['text_dim']};line-height:170%;'>"
            f"python scripts/make_samples.py&nbsp;&nbsp;<span style='color:{c['text_faint']};'>"
            "# 生成雨/雾/雪合成样本</span><br>"
            f"python scripts/demo.py&nbsp;&nbsp;<span style='color:{c['text_faint']};'>"
            "# 命令行演示 + 指标对比表</span><br>"
            f"python scripts/smoke_test.py&nbsp;&nbsp;<span style='color:{c['text_faint']};'>"
            "# 环境自检</span><br>"
            f"python scripts/ui_snapshot.py&nbsp;&nbsp;<span style='color:{c['text_faint']};'>"
            "# 把三个页面渲染成图片</span>"
            "</div>"
        )
        return "".join(blocks)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_TITLE} — {APP_SUBTITLE}")
        self.resize(1360, 860)
        self.setMinimumSize(1080, 700)
        self.cache = ModelCache()

        self.single_page = SingleImagePage(self.cache)
        self.batch_page = BatchEvalPage(self.cache)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self.single_page, "单图复原")
        self.tabs.addTab(self.batch_page, "批量评测")
        self.tabs.addTab(AboutPage(), "环境自检")

        header = self._build_header()
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 12, 14, 0)
        root_layout.setSpacing(10)
        root_layout.addWidget(header)
        root_layout.addWidget(self.tabs, stretch=1)
        self.setCentralWidget(root)

        self.statusBar().showMessage("就绪    Ctrl+O 打开图片    Ctrl+E 载入示例    Ctrl+S 保存结果    Esc 取消")
        self._build_shortcuts()

    # ------------------------------------------------------------------ #
    def _build_header(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("HeaderBar")
        bar.setFixedHeight(62)

        # 左侧色块 logo
        logo = QLabel("AWR")
        logo.setFixedSize(42, 42)
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(
            f"background-color: {palette()['primary']}; color: #FFFFFF; border-radius: 10px;"
            "font-weight: 700; font-size: 13px;"
        )

        title = QLabel(APP_TITLE)
        title.setObjectName("AppTitle")
        subtitle = QLabel(APP_SUBTITLE)
        subtitle.setObjectName("AppSubtitle")
        text_box = QVBoxLayout()
        text_box.setContentsMargins(0, 0, 0, 0)
        text_box.setSpacing(1)
        text_box.addWidget(title)
        text_box.addWidget(subtitle)

        env_badge = QLabel(self._env_summary())
        env_badge.setObjectName("Badge")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(12)
        layout.addWidget(logo)
        layout.addLayout(text_box)
        layout.addStretch(1)
        layout.addWidget(env_badge)
        return bar

    @staticmethod
    def _env_summary() -> str:
        device = "CPU"
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                device = f"GPU · {torch.cuda.get_device_name(0)}"
        except ImportError:
            device = "CPU（未装 torch）"
        forced = os.environ.get("AWR_DEVICE")
        suffix = f"  ·  已强制 {forced}" if forced else ""
        return f"运行设备：{device}{suffix}"

    def _build_shortcuts(self) -> None:
        def add(seq: str, slot) -> None:
            act = QAction(self)
            act.setShortcut(QKeySequence(seq))
            act.triggered.connect(slot)
            self.addAction(act)

        add("Ctrl+O", self.single_page.open_image)
        add("Ctrl+E", self.single_page.load_next_sample)
        add("Ctrl+S", self.single_page.save_result)
        add("Ctrl+D", self.single_page._toggle_mode)
        add("Esc", self.single_page.cancel_restore)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.cache.clear()
        super().closeEvent(event)


__all__ = ["MainWindow"]
