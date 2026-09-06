"""
图像 IO 与几何处理工具。只依赖 numpy + Pillow，保证无 torch/opencv 也能用。

统一约定：内存中的图像一律是 RGB uint8 (H, W, 3) 的 numpy 数组。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

IMAGE_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# --------------------------------------------------------------------------- #
# 读写
# --------------------------------------------------------------------------- #
def load_image(path: str | os.PathLike) -> np.ndarray:
    """读取图片为 RGB uint8。支持中文路径。"""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def save_image(path: str | os.PathLike, image: np.ndarray) -> str:
    """保存 RGB uint8 图片，自动创建父目录，返回实际路径。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(image)).save(p)
    return str(p)


def list_images(directory: str | os.PathLike, recursive: bool = False) -> List[Path]:
    """列出目录下所有图片，按文件名排序。"""
    d = Path(directory)
    if not d.is_dir():
        return []
    it = d.rglob("*") if recursive else d.glob("*")
    return sorted(p for p in it if p.suffix.lower() in IMAGE_EXTS)


def match_pairs(
    input_dir: str | os.PathLike,
    gt_dir: Optional[str | os.PathLike] = None,
    recursive: bool = False,
) -> List[Tuple[Path, Optional[Path]]]:
    """
    把退化图与 GT 按"文件名主干"配对，用于批量评测。

    GT 目录允许扩展名不同（jpg / png 混用）；配不上的 GT 记为 None，
    这类样本只能算无参考指标（NIQE / BRISQUE）。
    """
    inputs = list_images(input_dir, recursive=recursive)
    if gt_dir is None:
        return [(p, None) for p in inputs]

    gt_map: Dict[str, Path] = {}
    for g in list_images(gt_dir, recursive=recursive):
        gt_map.setdefault(g.stem, g)
        # RainDrop 等数据集常见命名: xxx_rain.png <-> xxx_clean.png
        for suffix in ("_clean", "_gt", "_target", "_norain", "_free"):
            if g.stem.endswith(suffix):
                gt_map.setdefault(g.stem[: -len(suffix)], g)

    pairs: List[Tuple[Path, Optional[Path]]] = []
    for p in inputs:
        gt = gt_map.get(p.stem)
        if gt is None:
            for suffix in ("_rain", "_hazy", "_snow", "_input", "_degraded"):
                if p.stem.endswith(suffix):
                    gt = gt_map.get(p.stem[: -len(suffix)])
                    if gt is not None:
                        break
        pairs.append((p, gt))
    return pairs


# --------------------------------------------------------------------------- #
# 类型与尺寸
# --------------------------------------------------------------------------- #
def to_uint8(image: np.ndarray) -> np.ndarray:
    """把 float(0~1 或 0~255) / 其他 dtype 安全转成 uint8。"""
    if image.dtype == np.uint8:
        return image
    arr = np.asarray(image, dtype=np.float32)
    if arr.max() <= 1.0 + 1e-6:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def to_float01(image: np.ndarray) -> np.ndarray:
    """uint8 -> float32 [0, 1]。"""
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)


def resize(image: np.ndarray, size: Tuple[int, int], resample: int = Image.BICUBIC) -> np.ndarray:
    """size = (width, height)。"""
    w, h = size
    if (image.shape[1], image.shape[0]) == (w, h):
        return image
    return np.asarray(Image.fromarray(to_uint8(image)).resize((w, h), resample), dtype=np.uint8)


def limit_long_side(image: np.ndarray, max_side: Optional[int]) -> np.ndarray:
    """长边超过 max_side 时等比缩小（控制显存与耗时）。max_side<=0 或 None 表示不限制。"""
    if not max_side or max_side <= 0:
        return image
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / float(max(h, w))
    return resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))))


def resize_to_multiple(image: np.ndarray, multiple: int) -> np.ndarray:
    """把宽高向上取整到 multiple 的倍数（WeatherDiffusion 官方预处理为 16）。"""
    if multiple is None or multiple <= 1:
        return image
    h, w = image.shape[:2]
    nh = int(np.ceil(h / multiple) * multiple)
    nw = int(np.ceil(w / multiple) * multiple)
    if (nh, nw) == (h, w):
        return image
    return resize(image, (nw, nh))


def pad_to_min_size(image: np.ndarray, min_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    图像小于 min_size（如 patch=64）时做反射填充，返回 (填充后图, 原始 (h, w))。
    """
    h, w = image.shape[:2]
    ph = max(0, min_size - h)
    pw = max(0, min_size - w)
    if ph == 0 and pw == 0:
        return image, (h, w)
    padded = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    return padded, (h, w)


def preprocess_for_model(
    image: np.ndarray,
    max_side: Optional[int] = 1024,
    size_multiple: int = 16,
    min_size: int = 0,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    模型推理前的统一预处理，返回 (处理后图像, 原始 (h, w))。
    推理完请用 postprocess_to_origin 还原尺寸，保证输出与输入等大。
    """
    orig_h, orig_w = image.shape[:2]
    out = limit_long_side(image, max_side)
    out = resize_to_multiple(out, size_multiple)
    if min_size:
        out, _ = pad_to_min_size(out, min_size)
    return out, (orig_h, orig_w)


def postprocess_to_origin(image: np.ndarray, orig_hw: Tuple[int, int]) -> np.ndarray:
    """把模型输出还原到原始尺寸。"""
    h, w = orig_hw
    out = image[:h, :w] if image.shape[0] >= h and image.shape[1] >= w else image
    return resize(out, (w, h))


# --------------------------------------------------------------------------- #
# 简单空域滤波（纯 numpy 实现，无需 opencv）
# --------------------------------------------------------------------------- #
def box_filter(image: np.ndarray, radius: int) -> np.ndarray:
    """积分图实现的均值滤波，支持 (H,W) 与 (H,W,C) 的 float 输入。"""
    arr = np.asarray(image, dtype=np.float32)
    single = arr.ndim == 2
    if single:
        arr = arr[:, :, None]
    h, w, c = arr.shape
    r = int(max(1, radius))
    pad = np.pad(arr, ((r, r), (r, r), (0, 0)), mode="reflect")
    cum = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    cum = np.pad(cum, ((1, 0), (1, 0), (0, 0)), mode="constant")
    k = 2 * r + 1
    total = cum[k:k + h, k:k + w] - cum[:h, k:k + w] - cum[k:k + h, :w] + cum[:h, :w]
    out = total / float(k * k)
    return out[:, :, 0] if single else out


def unsharp_mask(image: np.ndarray, radius: int = 2, amount: float = 0.6) -> np.ndarray:
    """USM 锐化，返回 uint8。"""
    arr = np.asarray(image, dtype=np.float32)
    blurred = box_filter(arr, radius)
    return np.clip(arr + amount * (arr - blurred), 0, 255).astype(np.uint8)


def _window_stack(plane: np.ndarray, size: int, mode: str = "reflect") -> np.ndarray:
    """把单通道图的 size x size 邻域堆叠成 (size*size, H, W)，用于中值/最小/最大滤波。"""
    arr = np.asarray(plane, dtype=np.float32)
    r = size // 2
    pad = np.pad(arr, ((r, r), (r, r)), mode=mode)
    h, w = arr.shape[:2]
    views = [pad[dy:dy + h, dx:dx + w] for dy in range(size) for dx in range(size)]
    return np.stack(views, axis=0)


def _apply_per_channel(image: np.ndarray, func) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return func(arr)
    return np.stack([func(arr[:, :, c]) for c in range(arr.shape[2])], axis=-1)


def median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """中值滤波（纯 numpy，逐通道处理以控制内存）。size 会自动取奇数。"""
    size = max(3, int(size) | 1)
    return _apply_per_channel(image, lambda p: np.median(_window_stack(p, size), axis=0))


def _window_stack_rect(plane: np.ndarray, kh: int, kw: int, mode: str = "reflect") -> np.ndarray:
    """矩形邻域堆叠，用于方向性滤波（如 1xK 的水平窗口）。"""
    arr = np.asarray(plane, dtype=np.float32)
    rh, rw = kh // 2, kw // 2
    pad = np.pad(arr, ((rh, rh), (rw, rw)), mode=mode)
    h, w = arr.shape[:2]
    views = [pad[dy:dy + h, dx:dx + w] for dy in range(kh) for dx in range(kw)]
    return np.stack(views, axis=0)


def directional_median(image: np.ndarray, kh: int, kw: int) -> np.ndarray:
    """
    方向性中值滤波。

    kh=1, kw=7 即水平长条窗口：对“近垂直的雨纹”特别有效，
    因为雨纹在水平方向上只占少数像素，会被中值直接抹除，
    而水平方向的结构（屋顶线、地平线）不受影响。
    """
    kh = max(1, int(kh) | 1)
    kw = max(1, int(kw) | 1)
    return _apply_per_channel(
        image, lambda p: np.median(_window_stack_rect(p, kh, kw), axis=0)
    )


def min_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """最小值滤波（形态学腐蚀），用于压制亮的雪点/雨纹。"""
    size = max(3, int(size) | 1)
    return _apply_per_channel(image, lambda p: np.min(_window_stack(p, size), axis=0))


def max_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """最大值滤波（形态学膨胀）。"""
    size = max(3, int(size) | 1)
    return _apply_per_channel(image, lambda p: np.max(_window_stack(p, size), axis=0))


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 1e-2) -> np.ndarray:
    """
    引导滤波（He et al. 2010）。guide 为单通道引导图，src 可为单通道或三通道。
    保边平滑，用于把图像分解成 base(结构) + detail(纹理/雨雪) 两层。
    """
    g = np.asarray(guide, dtype=np.float32)
    mean_i = box_filter(g, radius)
    var_i = box_filter(g * g, radius) - mean_i * mean_i

    def one(plane: np.ndarray) -> np.ndarray:
        mean_p = box_filter(plane, radius)
        cov_ip = box_filter(g * plane, radius) - mean_i * mean_p
        a = cov_ip / (var_i + eps)
        b = mean_p - a * mean_i
        return box_filter(a, radius) * g + box_filter(b, radius)

    return _apply_per_channel(src, one)


def to_gray(image: np.ndarray) -> np.ndarray:
    """RGB -> 单通道灰度（保持与输入相同的数值范围）。"""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    return arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 8) -> np.ndarray:
    """把两张图横向拼接（生成答辩用对比图）。"""
    h = max(left.shape[0], right.shape[0])
    l = resize(left, (int(left.shape[1] * h / left.shape[0]), h))
    r = resize(right, (int(right.shape[1] * h / right.shape[0]), h))
    sep = np.full((h, gap, 3), 255, dtype=np.uint8)
    return np.concatenate([l, sep, r], axis=1)


def make_demo_image(size: Tuple[int, int] = (256, 256), seed: int = 0) -> np.ndarray:
    """
    生成一张带"雨纹噪声"的合成图，供无数据集时跑通流程 / 单元测试使用。
    """
    rng = np.random.default_rng(seed)
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack(
        [
            128 + 100 * np.sin(xx / 24.0),
            128 + 100 * np.sin(yy / 32.0),
            128 + 100 * np.sin((xx + yy) / 40.0),
        ],
        axis=-1,
    )
    streaks = np.zeros((h, w), dtype=np.float32)
    for _ in range(60):
        x0 = rng.integers(0, w)
        y0 = rng.integers(0, h)
        length = int(rng.integers(8, 24))
        for k in range(length):
            y, x = y0 + k, x0 + k // 3
            if 0 <= y < h and 0 <= x < w:
                streaks[y, x] = 200.0
    noisy = base + streaks[:, :, None]
    return np.clip(noisy, 0, 255).astype(np.uint8)


__all__ = [
    "IMAGE_EXTS",
    "load_image",
    "save_image",
    "list_images",
    "match_pairs",
    "to_uint8",
    "to_float01",
    "resize",
    "limit_long_side",
    "resize_to_multiple",
    "pad_to_min_size",
    "preprocess_for_model",
    "postprocess_to_origin",
    "box_filter",
    "unsharp_mask",
    "median_filter",
    "directional_median",
    "min_filter",
    "max_filter",
    "guided_filter",
    "to_gray",
    "side_by_side",
    "make_demo_image",
]
