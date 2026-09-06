"""
=============================================================================
【界面组 U2 任务】原图 / 复原图对比控件
=============================================================================

本文件已实现一个可用的基础版本：
  - 等比缩放居中显示，两图共用同一显示区域
  - 滑块擦除对比（拖动中缝或用鼠标左右拖动）
  - 并排模式切换
  - 圆角画布 + 棋盘底纹 + 徽章式标注（视觉美化）

留给 U2 继续做的（TODO 已标注在代码里）:
  1. 滚轮缩放 + 按住拖动平移，且两图缩放平移完全同步
  2. 局部放大镜（鼠标位置的 2~4 倍细节框），答辩演示雨纹/雪点细节必用
  3. 导出当前对比视图为图片（答辩配图）
=============================================================================
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtCore import QPoint, QRect, QRectF, Qt
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import QSizePolicy, QWidget

from .qt_utils import numpy_to_pixmap
from .theme import is_dark, palette

RADIUS = 10.0
LABEL_BEFORE = "退化图"
LABEL_AFTER = "复原图"
TONE_BEFORE = QColor(76, 141, 255)     # 蓝：退化图
TONE_AFTER = QColor(61, 214, 140)      # 绿：复原图


def _c(key: str) -> QColor:
    """从当前主题调色盘取颜色。"""
    return QColor(palette()[key])


def _canvas_colors() -> tuple:
    """返回 (渐变上, 渐变下, 网格色)。"""
    if is_dark():
        return QColor(20, 22, 29), QColor(13, 15, 20), QColor(255, 255, 255, 8)
    return QColor(232, 236, 243), QColor(216, 222, 233), QColor(0, 0, 0, 12)


class CompareView(QWidget):
    """双图对比控件。before = 退化图，after = 复原图。"""

    MODE_WIPE = "wipe"
    MODE_SIDE_BY_SIDE = "side"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._before: Optional[QPixmap] = None
        self._after: Optional[QPixmap] = None
        self._split = 0.5           # 擦除位置，0~1
        self._mode = self.MODE_WIPE
        self._dragging = False
        self._hover_handle = False
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------ #
    # 对外接口（主窗口只用这几个方法）
    # ------------------------------------------------------------------ #
    def set_before(self, image: Optional[np.ndarray]) -> None:
        self._before = numpy_to_pixmap(image) if image is not None else None
        self.update()

    def set_after(self, image: Optional[np.ndarray]) -> None:
        self._after = numpy_to_pixmap(image) if image is not None else None
        self.update()

    def set_images(self, before: Optional[np.ndarray], after: Optional[np.ndarray]) -> None:
        self.set_before(before)
        self.set_after(after)

    def clear(self) -> None:
        self._before = None
        self._after = None
        self._split = 0.5
        self.update()

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self.update()

    def toggle_mode(self) -> str:
        self.set_mode(self.MODE_SIDE_BY_SIDE if self._mode == self.MODE_WIPE else self.MODE_WIPE)
        return self._mode

    def mode_text(self) -> str:
        return "擦除对比" if self._mode == self.MODE_WIPE else "并排对比"

    def export_pixmap(self) -> QPixmap:
        """把当前对比视图渲染成一张图，方便答辩截图。"""
        pix = QPixmap(self.size())
        pix.fill(Qt.transparent)
        painter = QPainter(pix)
        self._render(painter)
        painter.end()
        return pix

    # ------------------------------------------------------------------ #
    # 绘制
    # ------------------------------------------------------------------ #
    def paintEvent(self, event) -> None:  # noqa: N802 Qt 命名
        painter = QPainter(self)
        self._render(painter)

    def _render(self, painter: QPainter) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        canvas = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(canvas, RADIUS, RADIUS)

        # 渐变底 + 细网格，空白时不至于太单调
        top, bottom, grid = _canvas_colors()
        gradient = QLinearGradient(canvas.topLeft(), canvas.bottomLeft())
        gradient.setColorAt(0.0, top)
        gradient.setColorAt(1.0, bottom)
        painter.fillPath(path, QBrush(gradient))

        painter.save()
        painter.setClipPath(path)
        self._draw_grid(painter, grid)

        if self._before is None and self._after is None:
            self._draw_placeholder(painter)
        elif self._mode == self.MODE_SIDE_BY_SIDE:
            self._paint_side_by_side(painter)
        else:
            self._paint_wipe(painter)
        painter.restore()

        painter.setPen(QPen(_c("border"), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

    def _draw_grid(self, painter: QPainter, grid_color: QColor, step: int = 28) -> None:
        painter.save()
        painter.setPen(QPen(grid_color, 1))
        for x in range(0, self.width(), step):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), step):
            painter.drawLine(0, y, self.width(), y)
        painter.restore()

    def _paint_side_by_side(self, painter: QPainter) -> None:
        w = self.width() // 2
        left_rect = QRect(0, 0, w, self.height())
        right_rect = QRect(w, 0, self.width() - w, self.height())
        self._draw_fit(painter, self._before, left_rect, LABEL_BEFORE)
        self._draw_fit(painter, self._after, right_rect, LABEL_AFTER)

        painter.setPen(QPen(QColor(128, 128, 128, 90), 1))
        painter.drawLine(w, 8, w, self.height() - 8)
        self._draw_badge(painter, left_rect.left() + 12, 12, LABEL_BEFORE, TONE_BEFORE)
        self._draw_badge(painter, right_rect.left() + 12, 12, LABEL_AFTER, TONE_AFTER)

    def _paint_wipe(self, painter: QPainter) -> None:
        target = self._fit_rect(self._before or self._after)
        self._draw_shadow(painter, target)

        clip_path = QPainterPath()
        clip_path.addRoundedRect(QRectF(target), 6, 6)
        painter.save()
        painter.setClipPath(clip_path)

        # 底层复原图，上层按 split 裁剪盖退化图
        self._draw_in_rect(painter, self._after or self._before, target)
        split_x = int(target.left() + target.width() * self._split)
        painter.save()
        painter.setClipRect(QRect(target.left(), target.top(), max(0, split_x - target.left()), target.height()))
        self._draw_in_rect(painter, self._before or self._after, target)
        painter.restore()
        painter.restore()

        # 中缝：发光竖线 + 圆形手柄
        self._draw_handle(painter, split_x, target)

        if split_x - target.left() > 76:
            self._draw_badge(painter, target.left() + 10, target.top() + 10, LABEL_BEFORE, TONE_BEFORE)
        if target.right() - split_x > 76:
            label_w = self._badge_width(LABEL_AFTER)
            self._draw_badge(
                painter,
                target.right() - 10 - label_w,
                target.top() + 10,
                LABEL_AFTER,
                TONE_AFTER,
            )
        if self._after is None:
            self._draw_footer(painter, target, "尚未复原 —— 右侧选择方法后点击『开始复原』")

    # ------------------------------------------------------------------ #
    # 交互
    # ------------------------------------------------------------------ #
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._mode == self.MODE_WIPE:
            self._dragging = True
            self._update_split(event.pos().x())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._dragging:
            self._update_split(event.pos().x())
            return
        if self._mode == self.MODE_WIPE and (self._before is not None or self._after is not None):
            target = self._fit_rect(self._before or self._after)
            split_x = target.left() + target.width() * self._split
            near = abs(event.pos().x() - split_x) < 14 and target.top() <= event.pos().y() <= target.bottom()
            self.setCursor(Qt.SplitHCursor if near else Qt.ArrowCursor)
            if near != self._hover_handle:
                self._hover_handle = near
                self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._dragging = False

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.toggle_mode()

    # TODO(U2): wheelEvent 实现同步缩放，mouseMoveEvent 加平移，再加放大镜绘制

    def _update_split(self, x: int) -> None:
        target = self._fit_rect(self._before or self._after)
        if target.width() <= 0:
            return
        ratio = (x - target.left()) / float(target.width())
        self._split = float(min(1.0, max(0.0, ratio)))
        self.update()

    # ------------------------------------------------------------------ #
    # 绘制工具
    # ------------------------------------------------------------------ #
    def _fit_rect(self, pixmap: Optional[QPixmap]) -> QRect:
        """计算等比缩放后居中显示的矩形（留出 16px 边距）。"""
        if pixmap is None or pixmap.isNull():
            return self.rect()
        area = self.rect().adjusted(16, 16, -16, -16)
        scale = min(area.width() / pixmap.width(), area.height() / pixmap.height())
        w = max(1, int(pixmap.width() * scale))
        h = max(1, int(pixmap.height() * scale))
        return QRect(area.left() + (area.width() - w) // 2, area.top() + (area.height() - h) // 2, w, h)

    def _draw_in_rect(self, painter: QPainter, pixmap: Optional[QPixmap], rect: QRect) -> None:
        if pixmap is None or pixmap.isNull():
            return
        painter.drawPixmap(rect, pixmap)

    def _draw_fit(self, painter: QPainter, pixmap: Optional[QPixmap], area: QRect, hint: str = "") -> None:
        if pixmap is None or pixmap.isNull():
            if hint:
                painter.setPen(QPen(_c("text_faint")))
                painter.drawText(area, Qt.AlignCenter, f"{hint}\n（暂无）")
            return
        inner = area.adjusted(14, 14, -14, -14)
        scale = min(inner.width() / pixmap.width(), inner.height() / pixmap.height())
        w = max(1, int(pixmap.width() * scale))
        h = max(1, int(pixmap.height() * scale))
        rect = QRect(inner.left() + (inner.width() - w) // 2, inner.top() + (inner.height() - h) // 2, w, h)
        self._draw_shadow(painter, rect)
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), 6, 6)
        painter.save()
        painter.setClipPath(path)
        painter.drawPixmap(rect, pixmap)
        painter.restore()

    @staticmethod
    def _draw_shadow(painter: QPainter, rect: QRect) -> None:
        """用几层半透明描边模拟投影，让图像从背景里“浮”起来。"""
        painter.save()
        painter.setBrush(Qt.NoBrush)
        for i in range(1, 7):
            painter.setPen(QPen(QColor(0, 0, 0, 16), 1))
            painter.drawRoundedRect(QRectF(rect).adjusted(-i, -i, i, i), 6 + i, 6 + i)
        painter.restore()

    def _draw_handle(self, painter: QPainter, split_x: int, target: QRect) -> None:
        """中缝：发光竖线 + 带左右箭头的圆形手柄。线用白色（覆在图像上而非背景）。"""
        painter.save()
        glow = QColor(255, 255, 255, 60 if self._hover_handle else 30)
        painter.setPen(QPen(glow, 5))
        painter.drawLine(split_x, target.top(), split_x, target.bottom())
        painter.setPen(QPen(QColor(255, 255, 255, 230), 2))
        painter.drawLine(split_x, target.top(), split_x, target.bottom())

        r = 11 if self._hover_handle else 9
        center = QPoint(split_x, target.center().y())
        painter.setPen(QPen(QColor(255, 255, 255, 235), 2))
        painter.setBrush(QColor(30, 33, 41, 230))
        painter.drawEllipse(center, r, r)
        # 手柄里的左右箭头
        painter.setPen(QPen(QColor(255, 255, 255, 235), 2))
        painter.drawLine(split_x - 4, center.y(), split_x - 1, center.y() - 3)
        painter.drawLine(split_x - 4, center.y(), split_x - 1, center.y() + 3)
        painter.drawLine(split_x + 4, center.y(), split_x + 1, center.y() - 3)
        painter.drawLine(split_x + 4, center.y(), split_x + 1, center.y() + 3)
        painter.restore()

    def _badge_font(self) -> QFont:
        font = QFont(self.font())
        font.setPointSize(9)
        font.setBold(True)
        return font

    def _badge_width(self, text: str) -> int:
        return QFontMetrics(self._badge_font()).width(text) + 22

    def _draw_badge(self, painter: QPainter, x: int, y: int, text: str, tone: QColor) -> None:
        """胶囊形标注，盖在图像上，固定用深底白字保证可读。"""
        font = self._badge_font()
        metrics = QFontMetrics(font)
        w = metrics.width(text) + 22
        h = metrics.height() + 8
        rect = QRectF(x, y, w, h)
        painter.save()
        painter.setFont(font)
        painter.setPen(QPen(QColor(tone.red(), tone.green(), tone.blue(), 170), 1))
        painter.setBrush(QColor(16, 18, 24, 205))
        painter.drawRoundedRect(rect, h / 2, h / 2)
        # 左侧小圆点
        dot = QRectF(x + 8, y + h / 2 - 3, 6, 6)
        painter.setPen(Qt.NoPen)
        painter.setBrush(tone)
        painter.drawEllipse(dot)
        painter.setPen(QPen(QColor(235, 238, 245)))
        painter.drawText(rect.adjusted(18, 0, -6, 0), Qt.AlignVCenter | Qt.AlignLeft, text)
        painter.restore()

    def _draw_footer(self, painter: QPainter, target: QRect, text: str) -> None:
        painter.save()
        font = QFont(self.font())
        font.setPointSize(9)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        w = metrics.width(text) + 24
        h = metrics.height() + 12
        rect = QRectF(target.center().x() - w / 2, target.bottom() - h - 12, w, h)
        painter.setPen(QPen(QColor(255, 255, 255, 30), 1))
        painter.setBrush(QColor(16, 18, 24, 195))
        painter.drawRoundedRect(rect, h / 2, h / 2)
        painter.setPen(QPen(QColor(200, 206, 218)))
        painter.drawText(rect, Qt.AlignCenter, text)
        painter.restore()

    def _draw_placeholder(self, painter: QPainter) -> None:
        """空状态：虚线框 + 图标 + 提示，比一行灰字体面得多。"""
        painter.save()
        box = QRectF(self.rect()).adjusted(48, 48, -48, -48)
        icon_tone = _c("text_faint")
        painter.setPen(QPen(_c("border_light"), 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(box, RADIUS, RADIUS)

        cx, cy = box.center().x(), box.center().y()
        # 简笔画“图片”图标
        icon = QRectF(cx - 34, cy - 52, 68, 52)
        painter.setPen(QPen(icon_tone, 2))
        painter.drawRoundedRect(icon, 6, 6)
        painter.setBrush(icon_tone)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QRectF(icon.left() + 11, icon.top() + 10, 11, 11))
        peak = QPainterPath()
        peak.moveTo(icon.left() + 8, icon.bottom() - 7)
        peak.lineTo(icon.left() + 27, icon.bottom() - 27)
        peak.lineTo(icon.left() + 41, icon.bottom() - 12)
        peak.lineTo(icon.left() + 52, icon.bottom() - 22)
        peak.lineTo(icon.right() - 8, icon.bottom() - 7)
        peak.closeSubpath()
        painter.drawPath(peak)

        font = QFont(self.font())
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(_c("text_dim")))
        painter.drawText(
            QRectF(box.left(), cy + 6, box.width(), 26),
            Qt.AlignHCenter | Qt.AlignTop,
            "把图片拖到这里",
        )
        font.setPointSize(9)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QPen(_c("text_faint")))
        painter.drawText(
            QRectF(box.left(), cy + 34, box.width(), 44),
            Qt.AlignHCenter | Qt.AlignTop,
            "或点击上方『打开图片』/『载入示例』\n双击画面可切换 擦除 / 并排 对比模式",
        )
        painter.restore()


__all__ = ["CompareView"]
