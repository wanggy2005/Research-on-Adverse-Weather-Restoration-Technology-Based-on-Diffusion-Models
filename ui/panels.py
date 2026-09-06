"""
=============================================================================
【界面组 U1 任务】参数面板与指标面板
=============================================================================

ControlPanel: 天气类型、质量预设（快速/标准/高质量）、采样步数、grid_r、开始/取消
MetricPanel:  单图指标展示（PSNR/SSIM 需要 GT，NIQE/BRISQUE 不需要）

已实现基础版本，U1 可继续加：预设记忆、参数联动提示、GPU 显存显示等。
=============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.config import list_weather_configs, load_config, weight_path
from core.metrics import METRIC_LABELS, format_value
from ui.theme import palette

#: 三档质量预设 -> (采样步数, grid_r)
QUALITY_PRESETS: Dict[str, Dict[str, int]] = {
    "快速 (演示用)": {"timesteps": 10, "grid_r": 32},
    "标准 (推荐)": {"timesteps": 25, "grid_r": 16},
    "高质量 (慢)": {"timesteps": 25, "grid_r": 8},
}

#: 缺权重标记，界面上用它判断要不要显示警示徽章
MISSING_WEIGHT_TAG = "缺权重"

#: 参数栏内自动换行文本的最大宽度（与 main_window.SIDEBAR_WIDTH 匹配）
TEXT_WIDTH = 286


class ControlPanel(QWidget):
    """参数面板。对外只发两个信号，主窗口不需要关心内部控件。"""

    start_requested = pyqtSignal()
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.weather_combo = QComboBox()
        self.preset_combo = QComboBox()
        self.steps_slider = QSlider(Qt.Horizontal)
        self.steps_label = QLabel("25")
        self.grid_spin = QSpinBox()
        self.start_button = QPushButton("开始复原")
        self.cancel_button = QPushButton("取消")
        self.desc_label = QLabel("")
        self.status_badge = QLabel("")
        self._build()
        self._connect()
        self.reload_configs()

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        self.preset_combo.addItems(list(QUALITY_PRESETS.keys()))
        self.preset_combo.setCurrentText("标准 (推荐)")

        self.steps_slider.setRange(2, 50)
        self.steps_slider.setValue(25)
        self.steps_slider.setToolTip("扩散采样步数：越大越精细，耗时线性增长")
        self.steps_label.setObjectName("MetricValue")
        self.steps_label.setMinimumWidth(34)
        self.steps_label.setAlignment(Qt.AlignCenter)

        self.grid_spin.setRange(2, 64)
        self.grid_spin.setValue(16)
        self.grid_spin.setSingleStep(2)
        self.grid_spin.setToolTip("滑窗步长：越小越平滑，但速度越慢")

        self.weather_combo.setMinimumHeight(30)
        self.preset_combo.setMinimumHeight(30)
        self.grid_spin.setMinimumHeight(28)

        self.start_button.setObjectName("PrimaryButton")
        self.start_button.setMinimumHeight(36)
        self.start_button.setCursor(Qt.PointingHandCursor)
        self.cancel_button.setObjectName("DangerButton")
        self.cancel_button.setMinimumHeight(36)
        self.cancel_button.setCursor(Qt.PointingHandCursor)
        self.cancel_button.setEnabled(False)

        self.desc_label.setObjectName("Hint")
        self.desc_label.setWordWrap(True)
        self.desc_label.setMaximumWidth(TEXT_WIDTH)
        self.desc_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.status_badge.setObjectName("Badge")
        self.status_badge.setVisible(False)

        steps_row = QHBoxLayout()
        steps_row.setContentsMargins(0, 0, 0, 0)
        steps_row.setSpacing(8)
        steps_row.addWidget(self.steps_slider, stretch=1)
        steps_row.addWidget(self.steps_label)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setVerticalSpacing(11)
        form.setHorizontalSpacing(10)
        form.addRow(self._key("复原方法"), self.weather_combo)
        form.addRow(self._key("质量预设"), self.preset_combo)
        form.addRow(self._key("采样步数"), self._wrap(steps_row))
        form.addRow(self._key("滑窗步长"), self.grid_spin)

        group = QGroupBox("复原参数")
        inner = QVBoxLayout(group)
        inner.setContentsMargins(4, 6, 4, 4)
        inner.setSpacing(10)
        inner.addLayout(form)
        inner.addWidget(self.status_badge, alignment=Qt.AlignLeft)
        inner.addWidget(self._separator())
        inner.addWidget(self.desc_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addWidget(self.start_button, stretch=2)
        buttons.addWidget(self.cancel_button, stretch=1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(group)
        layout.addLayout(buttons)

    @staticmethod
    def _key(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("MetricKey")
        return label

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFixedHeight(1)
        line.setStyleSheet(f"background-color: {palette()['border']}; border: none;")
        return line

    @staticmethod
    def _wrap(layout) -> QWidget:
        w = QWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)
        return w

    def _connect(self) -> None:
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        self.steps_slider.valueChanged.connect(lambda v: self.steps_label.setText(str(v)))
        self.weather_combo.currentIndexChanged.connect(self._update_desc)
        self.start_button.clicked.connect(self.start_requested.emit)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)

    # ------------------------------------------------------------------ #
    def reload_configs(self) -> None:
        """
        扫描 configs/*.yaml 填充下拉框，新增天气类型不用改界面代码。

        同时检查权重是否就位：缺权重的项会标注出来，
        并自动默认选中第一个"现在就能跑"的方法，避免一打开就报错。
        """
        self.weather_combo.clear()
        first_ready = -1
        for name in list_weather_configs():
            ready = True
            try:
                cfg = load_config(name)
                label = str(cfg.get("display_name") or name)
                wp = weight_path(cfg)
                if wp and not Path(wp).is_file():
                    ready = False
                    label = f"{label}  [{MISSING_WEIGHT_TAG}]"
            except Exception:  # noqa: BLE001
                label = name
                ready = False
            self.weather_combo.addItem(label, userData=name)
            if ready and first_ready < 0:
                first_ready = self.weather_combo.count() - 1
        if first_ready >= 0:
            self.weather_combo.setCurrentIndex(first_ready)
        self._update_desc()

    def _apply_preset(self, text: str) -> None:
        preset = QUALITY_PRESETS.get(text)
        if not preset:
            return
        self.steps_slider.setValue(int(preset["timesteps"]))
        self.grid_spin.setValue(int(preset["grid_r"]))

    def _update_desc(self) -> None:
        name = self.current_weather()
        if not name:
            self.status_badge.setVisible(False)
            return

        # 徽章：就绪 / 缺权重，一眼看出这个方法现在能不能跑
        missing = MISSING_WEIGHT_TAG in self.weather_combo.currentText()
        self.status_badge.setObjectName("BadgeWarn" if missing else "BadgeOk")
        self.status_badge.setText("缺少权重文件，暂不可运行" if missing else "已就绪，可直接运行")
        self.status_badge.setVisible(True)
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)

        try:
            cfg = load_config(name)
            self.desc_label.setText(str(cfg.get("description") or ""))
        except Exception as exc:  # noqa: BLE001
            self.desc_label.setText(f"配置读取失败: {exc}")

    # ------------------------------------------------------------------ #
    def current_weather(self) -> Optional[str]:
        return self.weather_combo.currentData()

    def overrides(self) -> Dict[str, Any]:
        """把界面参数打包成 config 覆盖项，直接传给 ModelCache.get()。"""
        return {
            "sampling": {
                "timesteps": int(self.steps_slider.value()),
                "grid_r": int(self.grid_spin.value()),
            }
        }

    def set_running(self, running: bool) -> None:
        """推理中禁用参数，避免中途改参数造成状态混乱。"""
        self.start_button.setEnabled(not running)
        self.start_button.setText("复原中..." if running else "开始复原")
        self.cancel_button.setEnabled(running)
        for w in (self.weather_combo, self.preset_combo, self.steps_slider, self.grid_spin):
            w.setEnabled(not running)


class MetricPanel(QGroupBox):
    """单图指标展示。用大号等宽数字，投屏答辩时看得清。"""

    KEYS = ("psnr", "ssim", "niqe", "brisque")

    def __init__(self, parent=None):
        super().__init__("质量指标", parent)
        self._labels: Dict[str, QLabel] = {}

        grid = QGridLayout()
        grid.setContentsMargins(4, 6, 4, 4)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        for i, key in enumerate(self.KEYS):
            row, col = divmod(i, 2)
            grid.addWidget(self._tile(key), row, col)

        self.time_label = QLabel("-")
        self.device_label = QLabel("-")
        runtime = QHBoxLayout()
        runtime.setSpacing(6)
        for text, label in (("耗时", self.time_label), ("设备", self.device_label)):
            key = QLabel(text)
            key.setObjectName("MetricKey")
            label.setObjectName("Badge")
            label.setAlignment(Qt.AlignCenter)
            runtime.addWidget(key)
            runtime.addWidget(label, stretch=1)
        runtime_box = QWidget()
        runtime_box.setLayout(runtime)

        self.hint = QLabel("PSNR / SSIM 需 GT 清晰图；NIQE / BRISQUE 无需 GT")
        self.hint.setObjectName("Hint")
        self.hint.setWordWrap(True)
        self.hint.setMaximumWidth(TEXT_WIDTH)
        self.hint.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.setSpacing(10)
        layout.addLayout(grid)
        layout.addWidget(runtime_box)
        layout.addWidget(self.hint)

    def _tile(self, key: str) -> QWidget:
        """一个指标一个小卡片：上面是名字，下面是大数字。"""
        card = QWidget()
        card.setObjectName("Card")
        box = QVBoxLayout(card)
        box.setContentsMargins(10, 8, 10, 8)
        box.setSpacing(2)

        name = QLabel(METRIC_LABELS.get(key, key))
        name.setObjectName("MetricKey")
        value = QLabel("-")
        value.setObjectName("MetricValueDim")
        self._labels[key] = value

        box.addWidget(name)
        box.addWidget(value)
        return card

    def clear(self) -> None:
        for label in self._labels.values():
            label.setText("-")
            self._set_tone(label, active=False)
        self.time_label.setText("-")
        self.device_label.setText("-")

    def update_metrics(self, metrics: Dict[str, Optional[float]]) -> None:
        for key, label in self._labels.items():
            if key not in metrics:
                continue
            digits = 2 if key == "psnr" else 4
            text = format_value(metrics.get(key), digits)
            label.setText(text)
            self._set_tone(label, active=text not in ("-", "N/A", ""))

    def update_runtime(self, elapsed: Optional[float], device: str = "") -> None:
        if elapsed is not None:
            self.time_label.setText(f"{elapsed:.2f} s")
        if device:
            self.device_label.setText(device)

    @staticmethod
    def _set_tone(label: QLabel, active: bool) -> None:
        label.setObjectName("MetricValue" if active else "MetricValueDim")
        label.style().unpolish(label)
        label.style().polish(label)


__all__ = ["ControlPanel", "MetricPanel", "QUALITY_PRESETS"]
