"""
=============================================================================
统一视觉主题
=============================================================================

集中管理配色、圆角、字体与全局 QSS，界面各处只引用这里的常量，
避免样式散落在各文件里。

用法（app.py 里调一次即可）:
    from ui.theme import apply_theme
    apply_theme(app)                 # 默认深色
    apply_theme(app, dark=False)     # 浅色

新增控件想套样式时，给控件设 objectName，然后在下面 QSS 里加一条规则:
    btn.setObjectName("PrimaryButton")
=============================================================================
"""

from __future__ import annotations

from typing import Dict, Optional

# --------------------------------------------------------------------------- #
# 调色板
# --------------------------------------------------------------------------- #
DARK: Dict[str, str] = {
    "bg": "#16181F",           # 窗口底色
    "surface": "#1E2129",      # 卡片/面板
    "surface_alt": "#252935",  # 次级面板、输入框
    "border": "#333846",
    "border_light": "#3E4453",
    "text": "#E8EAF0",
    "text_dim": "#98A0B3",
    "text_faint": "#6C7488",
    "primary": "#4C8DFF",
    "primary_hover": "#6BA1FF",
    "primary_press": "#3A78E0",
    "success": "#3DD68C",
    "warning": "#F5A623",
    "danger": "#FF6B6B",
    "accent": "#A78BFA",
    "canvas": "#101218",
}

LIGHT: Dict[str, str] = {
    "bg": "#F4F6FA",
    "surface": "#FFFFFF",
    "surface_alt": "#EEF1F7",
    "border": "#D8DEE9",
    "border_light": "#C6CEDC",
    "text": "#1F2430",
    "text_dim": "#5B6478",
    "text_faint": "#8B93A6",
    "primary": "#2F6FED",
    "primary_hover": "#4A84F5",
    "primary_press": "#245BC7",
    "success": "#1FA971",
    "warning": "#D98416",
    "danger": "#E14C4C",
    "accent": "#7C5CD6",
    "canvas": "#E4E8F0",
}

RADIUS = 8
RADIUS_SM = 6
FONT_FAMILY = '"Microsoft YaHei UI", "Segoe UI", "PingFang SC", sans-serif'
MONO_FAMILY = '"Cascadia Mono", "Consolas", "JetBrains Mono", monospace'

#: apply_theme 会刷新它，自绘控件/富文本用 palette() 取当前配色
_CURRENT_DARK = True


# --------------------------------------------------------------------------- #
# QSS
# --------------------------------------------------------------------------- #
def build_stylesheet(dark: bool = True) -> str:
    """按调色板生成全局 QSS。"""
    c = DARK if dark else LIGHT
    return f"""
/* ============================ 全局 ============================ */
QWidget {{
    background-color: {c['bg']};
    color: {c['text']};
    font-family: {FONT_FAMILY};
    font-size: 13px;
}}
QMainWindow, QDialog {{
    background-color: {c['bg']};
}}
/* QLabel 默认不画底色，否则在卡片上会出现一块块深色矩形 */
QLabel, QCheckBox, QRadioButton {{
    background: transparent;
}}
QToolTip {{
    background-color: {c['surface_alt']};
    color: {c['text']};
    border: 1px solid {c['border_light']};
    border-radius: {RADIUS_SM}px;
    padding: 5px 8px;
}}

/* ============================ 卡片 ============================ */
QWidget#Card {{
    background-color: {c['surface']};
    border: 1px solid {c['border']};
    border-radius: {RADIUS}px;
}}
QWidget#HeaderBar {{
    background-color: {c['surface']};
    border: 1px solid {c['border']};
    border-radius: {RADIUS}px;
}}
QLabel#AppTitle {{
    font-size: 17px;
    font-weight: 600;
    color: {c['text']};
}}
QLabel#AppSubtitle {{
    font-size: 12px;
    color: {c['text_faint']};
}}
QLabel#SectionTitle {{
    font-size: 12px;
    font-weight: 600;
    color: {c['text_dim']};
    padding: 2px 0 2px 0;
}}
QLabel#Hint {{
    color: {c['text_faint']};
    font-size: 12px;
}}
QLabel#StatusText {{
    color: {c['text_dim']};
    padding: 2px 4px;
}}
QLabel#SummaryText {{
    color: {c['text']};
    background-color: {c['surface_alt']};
    border: 1px solid {c['border']};
    border-radius: {RADIUS_SM}px;
    padding: 7px 10px;
}}

/* 徽章 */
QLabel#Badge {{
    background-color: {c['surface_alt']};
    color: {c['text_dim']};
    border: 1px solid {c['border_light']};
    border-radius: 10px;
    padding: 2px 10px;
    font-size: 11px;
}}
QLabel#BadgeOk {{
    background-color: rgba(61, 214, 140, 0.15);
    color: {c['success']};
    border: 1px solid rgba(61, 214, 140, 0.35);
    border-radius: 10px;
    padding: 2px 10px;
    font-size: 11px;
}}
QLabel#BadgeWarn {{
    background-color: rgba(245, 166, 35, 0.15);
    color: {c['warning']};
    border: 1px solid rgba(245, 166, 35, 0.35);
    border-radius: 10px;
    padding: 2px 10px;
    font-size: 11px;
}}

/* ============================ 按钮 ============================ */
QPushButton {{
    background-color: {c['surface_alt']};
    color: {c['text']};
    border: 1px solid {c['border_light']};
    border-radius: {RADIUS_SM}px;
    padding: 7px 14px;
    min-height: 18px;
}}
QPushButton:hover {{
    background-color: {c['border']};
    border-color: {c['primary']};
}}
QPushButton:pressed {{
    background-color: {c['border_light']};
}}
QPushButton:disabled {{
    color: {c['text_faint']};
    background-color: {c['surface']};
    border-color: {c['border']};
}}
QPushButton#PrimaryButton {{
    background-color: {c['primary']};
    color: #FFFFFF;
    border: 1px solid {c['primary']};
    font-weight: 600;
    padding: 9px 16px;
}}
QPushButton#PrimaryButton:hover {{
    background-color: {c['primary_hover']};
    border-color: {c['primary_hover']};
}}
QPushButton#PrimaryButton:pressed {{
    background-color: {c['primary_press']};
}}
QPushButton#PrimaryButton:disabled {{
    background-color: {c['border']};
    border-color: {c['border']};
    color: {c['text_faint']};
}}
QPushButton#DangerButton {{
    background-color: transparent;
    color: {c['danger']};
    border: 1px solid rgba(255, 107, 107, 0.45);
    font-weight: 600;
    padding: 9px 16px;
}}
QPushButton#DangerButton:hover {{
    background-color: rgba(255, 107, 107, 0.12);
    border-color: {c['danger']};
}}
QPushButton#DangerButton:disabled {{
    color: {c['text_faint']};
    border-color: {c['border']};
    background-color: transparent;
}}
QPushButton#GhostButton {{
    background-color: transparent;
    border: 1px solid {c['border_light']};
    padding: 6px 12px;
}}
QPushButton#GhostButton:hover {{
    background-color: {c['surface_alt']};
    border-color: {c['primary']};
    color: {c['primary']};
}}

/* ============================ Tab ============================ */
QTabWidget::pane {{
    border: none;
    background: transparent;
    top: -1px;
}}
QTabBar {{
    qproperty-drawBase: 0;
}}
QTabBar::tab {{
    background: transparent;
    color: {c['text_faint']};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 9px 20px;
    margin-right: 4px;
    font-size: 13px;
}}
QTabBar::tab:hover {{
    color: {c['text']};
}}
QTabBar::tab:selected {{
    color: {c['primary']};
    border-bottom: 2px solid {c['primary']};
    font-weight: 600;
}}

/* ============================ 分组框 ============================ */
QGroupBox {{
    background-color: {c['surface']};
    border: 1px solid {c['border']};
    border-radius: {RADIUS}px;
    margin-top: 14px;
    padding: 14px 12px 12px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: 2px;
    padding: 0 6px;
    color: {c['text_dim']};
    background-color: {c['bg']};
    font-size: 12px;
}}

/* ============================ 输入控件 ============================ */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {{
    background-color: {c['surface_alt']};
    color: {c['text']};
    border: 1px solid {c['border_light']};
    border-radius: {RADIUS_SM}px;
    padding: 6px 9px;
    selection-background-color: {c['primary']};
    selection-color: #FFFFFF;
}}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus, QTextEdit:focus {{
    border-color: {c['primary']};
}}
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{
    color: {c['text_faint']};
    background-color: {c['surface']};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 20px;
    border: none;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {c['text_dim']};
    width: 0;
    height: 0;
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {c['surface_alt']};
    color: {c['text']};
    border: 1px solid {c['border_light']};
    border-radius: {RADIUS_SM}px;
    selection-background-color: {c['primary']};
    selection-color: #FFFFFF;
    outline: none;
    padding: 4px;
}}
QSpinBox::up-button, QSpinBox::down-button {{
    background-color: {c['border']};
    border: none;
    width: 16px;
}}
QSpinBox::up-button {{
    border-top-right-radius: {RADIUS_SM}px;
}}
QSpinBox::down-button {{
    border-bottom-right-radius: {RADIUS_SM}px;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
    background-color: {c['primary']};
}}
QSpinBox::up-arrow {{
    image: none;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-bottom: 4px solid {c['text']};
    width: 0; height: 0;
}}
QSpinBox::down-arrow {{
    image: none;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-top: 4px solid {c['text']};
    width: 0; height: 0;
}}

/* ============================ 复选框 ============================ */
QCheckBox {{
    spacing: 7px;
    color: {c['text_dim']};
}}
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border-radius: 4px;
    border: 1px solid {c['border_light']};
    background-color: {c['surface_alt']};
}}
QCheckBox::indicator:hover {{
    border-color: {c['primary']};
}}
QCheckBox::indicator:checked {{
    background-color: {c['primary']};
    border-color: {c['primary']};
}}

/* ============================ 滑块 ============================ */
QSlider::groove:horizontal {{
    height: 5px;
    background: {c['border']};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {c['primary']};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: #FFFFFF;
    border: 2px solid {c['primary']};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 9px;
}}
QSlider::handle:horizontal:hover {{
    border-color: {c['primary_hover']};
}}
QSlider:disabled::sub-page:horizontal {{
    background: {c['border_light']};
}}
QSlider:disabled::handle:horizontal {{
    background: {c['border_light']};
    border-color: {c['border_light']};
}}

/* ============================ 进度条 ============================ */
QProgressBar {{
    background-color: {c['surface_alt']};
    border: none;
    border-radius: 6px;
    height: 10px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {c['primary']};
    border-radius: 6px;
}}

/* ============================ 表格 ============================ */
QTableWidget, QTableView {{
    background-color: {c['surface']};
    alternate-background-color: {c['surface_alt']};
    border: 1px solid {c['border']};
    border-radius: {RADIUS}px;
    gridline-color: {c['border']};
    selection-background-color: rgba(76, 141, 255, 0.25);
    selection-color: {c['text']};
    outline: none;
}}
QTableWidget::item, QTableView::item {{
    padding: 5px 6px;
    border: none;
}}
QHeaderView::section {{
    background-color: {c['surface_alt']};
    color: {c['text_dim']};
    border: none;
    border-bottom: 1px solid {c['border_light']};
    border-right: 1px solid {c['border']};
    padding: 8px 6px;
    font-weight: 600;
    font-size: 12px;
}}
QTableCornerButton::section {{
    background-color: {c['surface_alt']};
    border: none;
}}

/* ============================ 滚动条 ============================ */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {c['border_light']};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {c['primary']};
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {c['border_light']};
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {c['primary']};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0; width: 0;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
}}

/* ============================ 分割器 ============================ */
QSplitter::handle {{
    background-color: transparent;
}}
QSplitter::handle:horizontal {{
    width: 8px;
}}
QSplitter::handle:hover {{
    background-color: {c['primary']};
}}

/* ============================ 状态栏 ============================ */
QStatusBar {{
    background-color: {c['surface']};
    color: {c['text_faint']};
    border-top: 1px solid {c['border']};
}}
QStatusBar::item {{
    border: none;
}}

/* ============================ 菜单 ============================ */
QMenuBar {{
    background-color: {c['surface']};
    color: {c['text_dim']};
    border-bottom: 1px solid {c['border']};
}}
QMenuBar::item {{
    padding: 6px 12px;
    background: transparent;
}}
QMenuBar::item:selected {{
    color: {c['primary']};
    background-color: {c['surface_alt']};
}}
QMenu {{
    background-color: {c['surface_alt']};
    border: 1px solid {c['border_light']};
    border-radius: {RADIUS_SM}px;
    padding: 5px;
}}
QMenu::item {{
    padding: 6px 22px 6px 14px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {c['primary']};
    color: #FFFFFF;
}}
QMenu::separator {{
    height: 1px;
    background: {c['border']};
    margin: 5px 8px;
}}

/* ============================ 指标数值 ============================ */
QLabel#MetricValue {{
    font-family: {MONO_FAMILY};
    font-size: 15px;
    font-weight: 600;
    color: {c['primary']};
}}
QLabel#MetricValueDim {{
    font-family: {MONO_FAMILY};
    font-size: 15px;
    color: {c['text_faint']};
}}
QLabel#MetricKey {{
    color: {c['text_dim']};
    font-size: 12px;
}}
QTextEdit#MonoText {{
    font-family: {MONO_FAMILY};
    font-size: 12px;
    background-color: {c['surface']};
    border: 1px solid {c['border']};
    border-radius: {RADIUS}px;
    padding: 12px;
    line-height: 150%;
}}
"""


def apply_theme(app, dark: bool = True) -> None:
    """给 QApplication 套主题：高分屏适配 + 全局 QSS。"""
    global _CURRENT_DARK  # noqa: PLW0603
    _CURRENT_DARK = dark

    from PyQt5.QtCore import Qt  # noqa: PLC0415
    from PyQt5.QtGui import QFont  # noqa: PLC0415

    try:
        app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:  # noqa: BLE001  某些平台不支持，忽略
        pass
    app.setFont(QFont("Microsoft YaHei UI", 9))
    app.setStyle("Fusion")
    app.setStyleSheet(build_stylesheet(dark))


def is_dark() -> bool:
    return _CURRENT_DARK


def palette() -> Dict[str, str]:
    """当前主题的调色盘，给自绘控件/HTML 文本用。"""
    return DARK if _CURRENT_DARK else LIGHT


def color(key: str, dark: Optional[bool] = None) -> str:
    """给自绘控件（如 CompareView）取色用，dark 缺省跟随当前主题。"""
    table = (DARK if _CURRENT_DARK else LIGHT) if dark is None else (DARK if dark else LIGHT)
    return table.get(key, "#000000")


__all__ = [
    "apply_theme",
    "build_stylesheet",
    "color",
    "is_dark",
    "palette",
    "DARK",
    "LIGHT",
    "RADIUS",
    "RADIUS_SM",
]
