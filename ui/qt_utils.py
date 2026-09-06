"""numpy 图像与 Qt 对象之间的转换工具。"""

from __future__ import annotations

import numpy as np
from PyQt5.QtGui import QImage, QPixmap


def numpy_to_qimage(image: np.ndarray) -> QImage:
    """RGB uint8 (H, W, 3) -> QImage。注意必须 copy()，否则底层内存被回收会花屏。"""
    if image is None:
        return QImage()
    arr = np.ascontiguousarray(image)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        return QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
    return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def numpy_to_pixmap(image: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(numpy_to_qimage(image))


def qimage_to_numpy(qimage: QImage) -> np.ndarray:
    """QImage -> RGB uint8 (H, W, 3)，用于处理界面里粘贴/拖入的图。"""
    img = qimage.convertToFormat(QImage.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    ptr.setsize(h * img.bytesPerLine())
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, img.bytesPerLine())
    return np.ascontiguousarray(arr[:, : w * 3].reshape(h, w, 3))


__all__ = ["numpy_to_qimage", "numpy_to_pixmap", "qimage_to_numpy"]
