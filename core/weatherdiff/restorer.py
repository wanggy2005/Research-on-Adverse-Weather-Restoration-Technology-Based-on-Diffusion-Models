"""
=============================================================================
WeatherDiffusion 复原器（本文件已实现，模型组一般不需要改）
=============================================================================

它负责所有"工程活"：
  - 设备选择、权重加载（兼容 DataParallel 的 module. 前缀、EMA 权重）
  - 推理前预处理（长边限制、尺寸对齐 16、小图反射填充）
  - 调用 patch-based DDIM 采样
  - 显存不足时自动把 patch_batch_size 减半重试
  - 推理后还原到原始尺寸

模型组只需要把 unet.py 和 sampling.py 两个 TODO 填完，本文件即可直接工作。
=============================================================================
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

from ..base import BaseRestorer, CancelCallback, ProgressCallback, WeightNotFoundError
from ..config import weight_path
from ..image_utils import postprocess_to_origin, preprocess_for_model, to_float01
from ..registry import register_restorer
from . import sampling as S
from .unet import build_unet


@register_restorer("weatherdiff")
class WeatherDiffRestorer(BaseRestorer):
    """基于 patch 的条件扩散模型复原器（WeatherDiffusion, TPAMI 2023）。"""

    display_name = "扩散模型 (WeatherDiffusion)"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = None
        self.betas = None
        self._torch = None

    # ------------------------------------------------------------------ #
    # 权重加载
    # ------------------------------------------------------------------ #
    def load_model(self) -> None:
        torch = self._import_torch()

        ckpt_path = weight_path(self.config)
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise WeightNotFoundError(
                f"找不到权重文件: {ckpt_path}\n"
                f"请执行 python scripts/download_weights.py 下载，"
                f"或手动把权重放到 weights/ 目录后检查 configs 里的 weights.path"
            )

        device = self._pick_device(torch)
        model = build_unet(self.config).to(device)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = self._extract_state_dict(ckpt)
        state = self._strip_prefix(state, "module.")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[weatherdiff] 警告: 有 {len(missing)} 个参数未从权重中加载, 例如 {missing[:3]}")
        if unexpected:
            print(f"[weatherdiff] 警告: 权重里有 {len(unexpected)} 个多余参数, 例如 {unexpected[:3]}")

        # 官方推理会把 EMA 权重覆盖到模型上，效果明显更好
        if (self.config.get("model") or {}).get("ema", True) and isinstance(ckpt, dict):
            ema_state = ckpt.get("ema_helper")
            if ema_state:
                self._apply_ema(model, ema_state)
                print("[weatherdiff] 已应用 EMA 权重")

        model.eval()
        self.model = model
        self.device = device

        diff_cfg = self.config.get("diffusion") or {}
        betas = S.get_beta_schedule(
            beta_schedule=diff_cfg.get("beta_schedule", "linear"),
            beta_start=float(diff_cfg.get("beta_start", 1e-4)),
            beta_end=float(diff_cfg.get("beta_end", 2e-2)),
            num_diffusion_timesteps=int(diff_cfg.get("num_diffusion_timesteps", 1000)),
        )
        self.betas = torch.from_numpy(betas).float().to(device)
        self.loaded = True
        print(f"[weatherdiff] 权重加载完成, device={device}")

    def unload(self) -> None:
        self.model = None
        self.betas = None
        self.loaded = False
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 推理
    # ------------------------------------------------------------------ #
    def restore(
        self,
        image: np.ndarray,
        progress_cb: Optional[ProgressCallback] = None,
        cancel_cb: Optional[CancelCallback] = None,
    ) -> np.ndarray:
        if not self.loaded:
            self.load_model()
        torch = self._import_torch()

        data_cfg = self.config.get("data") or {}
        sampling_cfg = self.config.get("sampling") or {}
        patch_size = int(data_cfg.get("image_size", 64))

        proc, orig_hw = preprocess_for_model(
            image,
            max_side=data_cfg.get("max_side", 1024),
            size_multiple=int(data_cfg.get("size_multiple", 16)),
            min_size=patch_size,
        )

        x_cond = (
            torch.from_numpy(to_float01(proc)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )
        h, w = proc.shape[:2]
        corners = S.corner_list(h, w, patch_size, int(sampling_cfg.get("grid_r", 16)))
        seq = S.make_timestep_seq(
            int((self.config.get("diffusion") or {}).get("num_diffusion_timesteps", 1000)),
            int(sampling_cfg.get("timesteps", 25)),
        )

        seed = sampling_cfg.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))

        batch = int(sampling_cfg.get("patch_batch_size", 32))
        while True:
            try:
                with torch.no_grad():
                    x = torch.randn_like(x_cond)
                    out = S.generalized_steps_overlapping(
                        x=x,
                        x_cond=x_cond,
                        seq=seq,
                        model=self.model,
                        betas=self.betas,
                        eta=float(sampling_cfg.get("eta", 0.0)),
                        corners=corners,
                        patch_size=patch_size,
                        patch_batch_size=batch,
                        progress_cb=self._wrap_progress(progress_cb, len(seq), orig_hw),
                        cancel_cb=cancel_cb,
                    )
                break
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or batch <= 1:
                    raise
                torch.cuda.empty_cache()
                batch = max(1, batch // 2)
                print(f"[weatherdiff] 显存不足, patch_batch_size 降为 {batch} 后重试")

        if isinstance(out, (list, tuple)):
            out = out[0][-1] if isinstance(out[0], (list, tuple)) else out[-1]
        result = S.inverse_data_transform(out)
        arr = (result.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        return postprocess_to_origin(arr, orig_hw)

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _wrap_progress(self, progress_cb, total: int, orig_hw):
        """把采样过程中的 tensor 预览转成 UI 能直接显示的 RGB uint8。"""
        if progress_cb is None:
            return None

        def _cb(step: int, _total: int, preview=None) -> None:
            img = None
            if preview is not None:
                try:
                    t = S.inverse_data_transform(preview)
                    img = (
                        t.squeeze(0).permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy() * 255.0
                    ).astype(np.uint8)
                    img = postprocess_to_origin(img, orig_hw)
                except Exception:  # noqa: BLE001 预览失败不影响推理
                    img = None
            self.report(progress_cb, step, total, img)

        return _cb

    def _import_torch(self):
        if self._torch is None:
            try:
                import torch  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "扩散模型需要 PyTorch。请先安装 torch，参考 README 或执行："
                    "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
                ) from exc
            self._torch = torch
        return self._torch

    @staticmethod
    def _pick_device(torch) -> str:
        forced = os.environ.get("AWR_DEVICE")
        if forced:
            return forced
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _extract_state_dict(ckpt: Any) -> Dict[str, Any]:
        """官方 checkpoint 是 dict，也兼容直接存 state_dict 的情况。"""
        if not isinstance(ckpt, dict):
            raise ValueError("无法解析的权重文件格式")
        for key in ("state_dict", "model", "net", "params"):
            value = ckpt.get(key)
            if isinstance(value, dict) and value:
                return value
        # 官方旧版本用 list 存 [state_dict, optimizer, epoch, step, ema]
        return ckpt

    @staticmethod
    def _strip_prefix(state: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        if not any(k.startswith(prefix) for k in state):
            return state
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}

    @staticmethod
    def _apply_ema(model, ema_state: Dict[str, Any]) -> None:
        """把 EMA 影子参数覆盖到模型（官方 EMAHelper.ema 的等价实现）。"""
        shadow = ema_state.get("shadow", ema_state)
        if not isinstance(shadow, dict):
            return
        clean = {k[len("module."):] if k.startswith("module.") else k: v for k, v in shadow.items()}
        named = dict(model.named_parameters())
        applied = 0
        for name, param in named.items():
            if name in clean:
                param.data.copy_(clean[name].data.to(param.device))
                applied += 1
        if applied == 0:
            print("[weatherdiff] 警告: EMA 参数名与模型不匹配，已跳过 EMA")
