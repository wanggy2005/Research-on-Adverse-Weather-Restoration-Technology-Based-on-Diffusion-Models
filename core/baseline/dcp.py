"""
非扩散对照基线：暗通道先验 (Dark Channel Prior, He et al. CVPR 2009) 去雾。

特点：纯 numpy 实现，不需要权重、不需要 GPU、不需要 torch，
所以它是全项目第一天就能跑出真实复原效果的方法，
既是指标表格里的"传统方法"对照列，也是界面演示的兜底方案。

评测组后续可在本目录再加 Restormer / TransWeather 等深度学习对照，
只要同样继承 BaseRestorer 并 @register_restorer("xxx") 即可。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import BaseRestorer, CancelCallback, ProgressCallback
from ..image_utils import box_filter, postprocess_to_origin, preprocess_for_model, to_float01
from ..registry import register_restorer


@register_restorer("dcp")
class DarkChannelPriorRestorer(BaseRestorer):
    """暗通道先验去雾。对雨雾叠加场景也有一定效果，对雨滴/雪几乎无效（这点正好作为对照结论）。"""

    display_name = "对照基线 (DCP 暗通道先验)"

    @property
    def default_steps(self) -> int:
        return 5  # 五个处理阶段，用于进度条

    def load_model(self) -> None:
        self.device = "cpu"
        self.loaded = True

    def restore(
        self,
        image: np.ndarray,
        progress_cb: Optional[ProgressCallback] = None,
        cancel_cb: Optional[CancelCallback] = None,
    ) -> np.ndarray:
        if not self.loaded:
            self.load_model()

        cfg = self.config.get("dcp") or {}
        data_cfg = self.config.get("data") or {}
        patch = int(cfg.get("patch_size", 15))
        omega = float(cfg.get("omega", 0.95))
        t0 = float(cfg.get("t0", 0.1))
        guide_radius = int(cfg.get("guided_radius", 40))
        guide_eps = float(cfg.get("guided_eps", 1e-3))

        total = self.default_steps
        # 大图先降到 max_side，最后还原到原尺寸
        proc, orig_hw = preprocess_for_model(
            image, max_side=data_cfg.get("max_side", 1024), size_multiple=1
        )
        img = to_float01(proc)

        # 1. 暗通道
        self.check_cancel(cancel_cb)
        dark = self._dark_channel(img, patch)
        self.report(progress_cb, 1, total, None)

        # 2. 大气光 A
        self.check_cancel(cancel_cb)
        atmos = self._atmospheric_light(img, dark)
        self.report(progress_cb, 2, total, None)

        # 3. 粗透射率
        self.check_cancel(cancel_cb)
        norm = img / np.maximum(atmos.reshape(1, 1, 3), 1e-6)
        trans = 1.0 - omega * self._dark_channel(norm, patch)
        self.report(progress_cb, 3, total, None)

        # 4. 引导滤波细化透射率（用灰度图作引导图）
        self.check_cancel(cancel_cb)
        gray = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        trans = self._guided_filter(gray, trans, guide_radius, guide_eps)
        trans = np.clip(trans, t0, 1.0)
        self.report(progress_cb, 4, total, None)

        # 5. 复原成像方程 J = (I - A) / t + A
        self.check_cancel(cancel_cb)
        out = (img - atmos.reshape(1, 1, 3)) / trans[:, :, None] + atmos.reshape(1, 1, 3)
        out = np.clip(out, 0.0, 1.0)
        result = (out * 255.0).astype(np.uint8)
        result = postprocess_to_origin(result, orig_hw)
        self.report(progress_cb, 5, total, result)
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def _dark_channel(img: np.ndarray, patch: int) -> np.ndarray:
        """通道最小 + 局部最小值滤波。用滑窗视图实现，避免依赖 opencv。"""
        min_c = img.min(axis=2)
        r = max(1, patch // 2)
        pad = np.pad(min_c, ((r, r), (r, r)), mode="edge")
        h, w = min_c.shape
        # 先行方向最小，再列方向最小（可分离，速度快）
        row_min = np.min(
            np.stack([pad[:, i:i + w] for i in range(2 * r + 1)], axis=0), axis=0
        )
        dark = np.min(
            np.stack([row_min[i:i + h, :] for i in range(2 * r + 1)], axis=0), axis=0
        )
        return dark.astype(np.float32)

    @staticmethod
    def _atmospheric_light(img: np.ndarray, dark: np.ndarray) -> np.ndarray:
        """取暗通道最亮 0.1% 像素中，原图亮度最高者作为大气光。"""
        h, w = dark.shape
        n = max(1, int(h * w * 0.001))
        idx = np.argpartition(dark.ravel(), -n)[-n:]
        flat = img.reshape(-1, 3)[idx]
        brightest = flat[np.argmax(flat.sum(axis=1))]
        return np.clip(brightest.astype(np.float32), 0.1, 1.0)

    @staticmethod
    def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
        """标准引导滤波（He et al. 2010），用于细化透射率边缘。"""
        mean_i = box_filter(guide, radius)
        mean_p = box_filter(src, radius)
        corr_i = box_filter(guide * guide, radius)
        corr_ip = box_filter(guide * src, radius)
        var_i = corr_i - mean_i * mean_i
        cov_ip = corr_ip - mean_i * mean_p
        a = cov_ip / (var_i + eps)
        b = mean_p - a * mean_i
        return box_filter(a, radius) * guide + box_filter(b, radius)
