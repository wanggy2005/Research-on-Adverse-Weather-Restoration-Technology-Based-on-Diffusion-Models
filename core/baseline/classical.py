"""
非扩散对照基线（二）：导向滤波分层 + 形态学/中值抑制 的传统复原方法。

它同时兼顾三类天气，且**不需要任何权重、不需要 GPU、不需要 torch**，
因此是"系统第一天就能演示、并且指标真的会变好"的方法：

  雨 / 雪  -> 图像分解成 base(结构) + detail(纹理)，雨纹与雪花是 detail 层里
             异常偏亮的尖峰，用中值参考值替换掉这些尖峰即可去除，同时保留正常纹理
  雾       -> 先按大气散射模型做暗通道去雾，再走上面的分层流程

评测报告里可以把它作为"传统方法"列，与扩散模型对比，突出扩散模型的优势。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import BaseRestorer, CancelCallback, ProgressCallback
from ..image_utils import (
    box_filter,
    directional_median,
    guided_filter,
    median_filter,
    min_filter,
    postprocess_to_origin,
    preprocess_for_model,
    to_float01,
    to_gray,
)
from ..registry import register_restorer
from .dcp import DarkChannelPriorRestorer


@register_restorer("classical")
class ClassicalRestorer(BaseRestorer):
    """传统复原：导向滤波分层 + detail 层尖峰抑制（+ 可选去雾）。"""

    display_name = "传统方法 (无需权重)"

    #: 最近一次推理自动判别出的退化类型，界面可以显示出来
    last_profile: str = ""

    @property
    def default_steps(self) -> int:
        return 6

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

        cfg = dict(self.config.get("classical") or {})
        data_cfg = self.config.get("data") or {}
        total = self.default_steps

        # 大图先降到 max_side，控制中值滤波的耗时；最后会还原到原尺寸
        proc, orig_hw = preprocess_for_model(
            image, max_side=data_cfg.get("max_side", 1024), size_multiple=1
        )
        img = to_float01(proc)

        # 1) 判别退化类型（雨/雾/雪的最优参数不同），并套用对应 profile
        self.check_cancel(cancel_cb)
        profile = str(cfg.get("profile", "auto")).lower()
        if profile == "auto":
            profile = self.detect_profile(img)
        overrides = (cfg.get("profiles") or {}).get(profile) or {}
        cfg.update(overrides)
        self.last_profile = profile

        dehaze_mode = str(cfg.get("dehaze", "auto")).lower()
        radius = int(cfg.get("guided_radius", 8))
        eps = float(cfg.get("guided_eps", 0.01))
        med_size = int(cfg.get("detail_median", 3))
        tau = float(cfg.get("spike_tau", 0.015))
        sharpen = float(cfg.get("sharpen", 0.0))
        second_pass = int(cfg.get("second_median", 5))
        bright_median = int(cfg.get("bright_median", 0))
        bright_tau = float(cfg.get("bright_tau", 0.08))
        bright_kernel = str(cfg.get("bright_kernel", "square")).lower()
        self.report(progress_cb, 1, total, None)

        # 2) 去雾（大气散射模型 + 暗通道先验）
        self.check_cancel(cancel_cb)
        need_dehaze = dehaze_mode == "true" or (
            dehaze_mode == "auto" and profile == "haze"
        )
        if need_dehaze:
            img = self._dehaze(img, cfg)
        # 亮斑抑制：比大窗口中值高出很多的像素（大颗雪花/粗雨纹）直接用中值替换，
        # 而窗户等“本来就大片偏亮”的结构其局部中值也高，不会被误删。
        # bright_kernel=horizontal 时用 1xK 水平窗口，专门对付近垂直的雨纹。
        if bright_median >= 3:
            if bright_kernel.startswith("h"):
                img_med = directional_median(img, 1, bright_median)
            else:
                img_med = median_filter(img, bright_median)
            img = np.where(img > img_med + bright_tau, img_med, img)
        self.report(progress_cb, 2, total, (img * 255).astype(np.uint8))

        # 3) 保边分层：base 保留结构，detail 包含纹理与雨雪
        self.check_cancel(cancel_cb)
        gray = to_gray(img)
        base = guided_filter(gray, img, radius=radius, eps=eps)
        detail = img - base
        self.report(progress_cb, 3, total, None)

        # 4) detail 层去尖峰：偏亮异常值用中值替换（雨纹/雪花都是亮尖峰）
        self.check_cancel(cancel_cb)
        detail_med = median_filter(detail, med_size)
        detail = np.where(detail > detail_med + tau, detail_med, detail)
        self.report(progress_cb, 4, total, None)

        # 5) 第二轮：更大窗口专门清粗一点的雪花/宽雨纹
        self.check_cancel(cancel_cb)
        if second_pass >= 3:
            detail_med2 = median_filter(detail, second_pass)
            detail = np.where(detail > detail_med2 + tau * 1.6, detail_med2, detail)
        restored = np.clip(base + detail, 0.0, 1.0)
        self.report(progress_cb, 5, total, (restored * 255).astype(np.uint8))

        # 6) 轻微锐化，把去噪损失的清晰度找回来
        self.check_cancel(cancel_cb)
        if sharpen > 0:
            blur = box_filter(restored, 1)
            restored = np.clip(restored + sharpen * (restored - blur), 0.0, 1.0)
        result = (restored * 255.0).astype(np.uint8)
        result = postprocess_to_origin(result, orig_hw)
        self.report(progress_cb, 6, total, result)
        return result

    # ------------------------------------------------------------------ #
    @classmethod
    def detect_profile(cls, img01: np.ndarray) -> str:
        """
        自动判别退化类型，返回 "haze" / "rain" / "snow"。

        依据:
          1. 暗通道均值高 -> 整体发白 -> 雾
          2. 否则看 detail 层亮尖峰的形状:
             雨纹是近垂直的长条，垂直连通性远大于水平连通性；
             雪花是各向同性的小圆斑，两者接近。
        """
        if cls._haze_score(img01) > 0.42:
            return "haze"

        gray = to_gray(img01)
        base = guided_filter(gray, gray, radius=8, eps=0.01)
        detail = gray - base
        mask = detail > 0.04
        if mask.mean() < 0.0015:        # 几乎没有尖峰，按最温和的雨处理
            return "rain"

        vert = np.logical_and(mask[:-2, :], np.logical_and(mask[1:-1, :], mask[2:, :]))
        horiz = np.logical_and(mask[:, :-2], np.logical_and(mask[:, 1:-1], mask[:, 2:]))
        v = float(vert.sum())
        hz = float(horiz.sum())
        ratio = v / max(hz, 1.0)
        return "rain" if ratio > 1.6 else "snow"

    @staticmethod
    def _haze_score(img01: np.ndarray) -> float:
        """暗通道均值：值越大说明整体越"发白"，越可能有雾。"""
        dark = min_filter(img01.min(axis=2), 15)
        return float(np.mean(dark))

    @staticmethod
    def _dehaze(img01: np.ndarray, cfg: dict) -> np.ndarray:
        """复用 DCP 的暗通道 / 大气光 / 引导滤波实现做去雾。"""
        patch = int(cfg.get("dcp_patch", 15))
        omega = float(cfg.get("dcp_omega", 0.92))
        t0 = float(cfg.get("dcp_t0", 0.12))

        dark = DarkChannelPriorRestorer._dark_channel(img01, patch)
        atmos = DarkChannelPriorRestorer._atmospheric_light(img01, dark)
        norm = img01 / np.maximum(atmos.reshape(1, 1, 3), 1e-6)
        trans = 1.0 - omega * DarkChannelPriorRestorer._dark_channel(norm, patch)
        trans = DarkChannelPriorRestorer._guided_filter(to_gray(img01), trans, 40, 1e-3)
        trans = np.clip(trans, t0, 1.0)
        out = (img01 - atmos.reshape(1, 1, 3)) / trans[:, :, None] + atmos.reshape(1, 1, 3)
        return np.clip(out, 0.0, 1.0)
