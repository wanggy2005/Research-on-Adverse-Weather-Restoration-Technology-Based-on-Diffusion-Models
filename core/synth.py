"""
合成样本生成：在没有下载真实数据集之前，先用程序生成"清晰图 + 雨/雾/雪退化图"配对，
让整个系统（复原 + 指标 + 界面）第一天就能完整跑起来并看到 PSNR/SSIM 的变化。

全部纯 numpy 实现，不依赖 opencv / 网络 / 数据集。

真实数据集到位后，把 data/samples 换成 data/raindrop 等真实目录即可，代码无需改动。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .image_utils import box_filter, to_gray

__all__ = [
    "make_clean_scene",
    "add_rain",
    "add_haze",
    "add_snow",
    "make_pair",
    "WEATHERS",
]

WEATHERS = ("rain", "haze", "snow")


# --------------------------------------------------------------------------- #
# 干净场景（充当 GT）
# --------------------------------------------------------------------------- #
def make_clean_scene(size: Tuple[int, int] = (384, 512), seed: int = 0) -> np.ndarray:
    """
    程序化生成一张"城市街景"风格的清晰图：天空渐变 + 阳光 + 地面 + 楼房窗格 + 树 + 车道线。
    含有大量高频边缘（窗格、车道线），便于观察复原前后的细节差异与指标变化。
    """
    h, w = size
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    horizon = int(h * 0.58)

    img = np.zeros((h, w, 3), dtype=np.float32)

    # 天空：竖直渐变
    t = np.clip(yy / max(1.0, float(horizon)), 0.0, 1.0)
    top = np.array([62, 104, 176], dtype=np.float32)
    bottom = np.array([196, 214, 234], dtype=np.float32)
    img += top[None, None, :] * (1.0 - t)[:, :, None] + bottom[None, None, :] * t[:, :, None]

    # 阳光光晕
    sx, sy = w * 0.74, horizon * 0.32
    dist = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
    glow = np.exp(-((dist / (0.30 * w)) ** 2))
    img += glow[:, :, None] * np.array([95, 78, 38], dtype=np.float32)

    # 地面
    ground_mask = yy >= horizon
    depth = np.clip((yy - horizon) / max(1.0, float(h - horizon)), 0.0, 1.0)
    ground = np.stack(
        [72 + 48 * depth, 84 + 42 * depth, 66 + 34 * depth], axis=-1
    ).astype(np.float32)
    img = np.where(ground_mask[:, :, None], ground, img)

    # 楼房 + 窗格
    n_buildings = 7
    edges = np.linspace(0, w, n_buildings + 1).astype(int)
    for i in range(n_buildings):
        x0, x1 = edges[i], edges[i + 1]
        bw = x1 - x0
        if bw < 12:
            continue
        bh = int(rng.integers(int(h * 0.16), int(h * 0.42)))
        y0 = horizon - bh
        tone = float(rng.integers(70, 135))
        img[y0:horizon, x0:x1] = np.array([tone, tone * 0.97, tone * 0.92], dtype=np.float32)
        # 楼体边缘描深
        img[y0:horizon, x0:x0 + 2] *= 0.75
        img[y0:y0 + 2, x0:x1] *= 0.8
        # 窗格
        step_y, step_x = 12, 10
        lit = float(rng.integers(180, 245))
        for wy in range(y0 + 6, horizon - 6, step_y):
            for wx in range(x0 + 4, x1 - 5, step_x):
                if rng.random() < 0.55:
                    img[wy:wy + 6, wx:wx + 5] = np.array(
                        [lit, lit * 0.95, lit * 0.7], dtype=np.float32
                    )
                else:
                    img[wy:wy + 6, wx:wx + 5] = np.array([40, 46, 58], dtype=np.float32)

    # 路面与车道线（强边缘，便于看细节）
    road_top = horizon + int((h - horizon) * 0.25)
    img[road_top:, :] = np.array([58, 58, 62], dtype=np.float32)
    center = w // 2
    for y in range(road_top, h, 26):
        half = 3 + int((y - road_top) * 0.05)
        img[y:y + 13, center - half:center + half] = np.array([236, 232, 210], dtype=np.float32)

    # 路边的树（三角形树冠 + 树干）
    for _ in range(4):
        tx = int(rng.integers(int(w * 0.05), int(w * 0.95)))
        th = int(rng.integers(30, 60))
        base_y = horizon + int(rng.integers(0, max(1, int((h - horizon) * 0.2))))
        for k in range(th):
            half = int((th - k) * 0.42) + 1
            y = base_y - k
            if 0 <= y < h:
                x0 = max(0, tx - half)
                x1 = min(w, tx + half)
                img[y, x0:x1] = np.array([34, 92, 48], dtype=np.float32)
        img[base_y:base_y + 10, max(0, tx - 2):tx + 2] = np.array([72, 52, 34], dtype=np.float32)

    # 轻微传感器噪声，避免图像过于"干净"
    img += rng.normal(0.0, 1.8, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 三种天气退化
# --------------------------------------------------------------------------- #
def add_rain(
    image: np.ndarray,
    seed: int = 0,
    n_streaks: int = 120,
    veil: float = 0.03,
) -> np.ndarray:
    """雨：短雨丝 + 少量水滴 + 轻微雨幕，模拟 RainDrop/Outdoor-Rain 真实雨形态。
    
    与之前版本的关键区别:
      - 雨丝数量进一步降低(200→120)，避免过密
      - 雨丝强度上限降低(0.85→0.65)，避免过亮
      - 雨幕更轻(0.04→0.03)
    """
    rng = np.random.default_rng(seed + 1000)
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    angle = rng.uniform(-0.15, 0.15)          # 雨的整体倾斜（真实雨接近垂直）
    for _ in range(int(n_streaks)):
        x0 = float(rng.integers(0, w))
        y0 = float(rng.integers(0, h))
        length = int(rng.integers(4, 14))     # 短雨丝
        strength = float(rng.uniform(0.2, 0.65))  # 降低上限，避免过亮
        jitter = angle + rng.uniform(-0.05, 0.05)
        width = rng.integers(1, 3)            # 1~2 像素宽
        for k in range(length):
            y = int(y0 + k)
            x = int(x0 + k * jitter)
            for dw in range(width):
                xx = x + dw
                if 0 <= y < h and 0 <= xx < w:
                    mask[y, xx] = max(mask[y, xx], strength)

    # 少量圆形水滴（模拟镜头上的雨滴）
    n_drops = int(n_streaks * 0.15)
    for _ in range(n_drops):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        radius = float(rng.uniform(1.5, 4.0))
        strength = float(rng.uniform(0.4, 0.9))
        r = int(np.ceil(radius)) + 1
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        ly, lx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.sqrt((ly - cy) ** 2 + (lx - cx) ** 2)
        blob = np.clip(1.0 - d / (radius + 0.5), 0.0, 1.0) * strength
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], blob)

    mask = np.clip(box_filter(mask, 1) * 1.4, 0.0, 1.0)      # 轻微柔化
    arr = image.astype(np.float32)
    rain_color = np.array([232, 236, 242], dtype=np.float32)
    out = arr * (1.0 - mask[:, :, None]) + rain_color[None, None, :] * mask[:, :, None]
    # 雨幕：整体轻微发灰
    out = out * (1.0 - veil) + 205.0 * veil
    return np.clip(out, 0, 255).astype(np.uint8)


def add_haze(image: np.ndarray, seed: int = 0, beta: float = 1.6, airlight: float = 226.0) -> np.ndarray:
    """雾：按大气散射模型 I = J*t + A*(1-t)，透射率随"深度"指数衰减。"""
    rng = np.random.default_rng(seed + 2000)
    h, w = image.shape[:2]
    yy, _ = np.mgrid[0:h, 0:w].astype(np.float32)
    horizon = h * 0.58
    # 越靠近地平线越远 -> 雾越浓；地面近处雾略薄
    depth = 1.0 - np.abs(yy - horizon) / max(1.0, float(h))
    depth = np.clip(depth, 0.05, 1.0)
    # 加一点非均匀雾团
    blobs = rng.normal(0.0, 1.0, (h // 16 + 1, w // 16 + 1)).astype(np.float32)
    blobs = np.kron(blobs, np.ones((16, 16), dtype=np.float32))[:h, :w]
    depth = np.clip(depth + 0.12 * box_filter(blobs, 6), 0.05, 1.2)

    trans = np.exp(-beta * depth)
    arr = image.astype(np.float32)
    out = arr * trans[:, :, None] + airlight * (1.0 - trans)[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def add_snow(image: np.ndarray, seed: int = 0, n_flakes: int = 700) -> np.ndarray:
    """雪：大小不一的白色雪花（软边圆斑）+ 少量运动拉丝 + 轻微白幕。"""
    rng = np.random.default_rng(seed + 3000)
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    for _ in range(int(n_flakes)):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        radius = float(rng.uniform(0.8, 3.4))
        strength = float(rng.uniform(0.55, 1.0))
        r = int(np.ceil(radius)) + 1
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        ly, lx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.sqrt((ly - cy) ** 2 + (lx - cx) ** 2)
        blob = np.clip(1.0 - d / (radius + 0.6), 0.0, 1.0) * strength
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], blob)

    # 部分雪花带运动拉丝
    for _ in range(int(n_flakes * 0.12)):
        x0 = float(rng.integers(0, w))
        y0 = float(rng.integers(0, h))
        length = int(rng.integers(6, 18))
        slope = rng.uniform(-0.5, 0.5)
        for k in range(length):
            y = int(y0 + k)
            x = int(x0 + k * slope)
            if 0 <= y < h and 0 <= x < w:
                mask[y, x] = max(mask[y, x], 0.85)

    mask = np.clip(mask, 0.0, 1.0)
    arr = image.astype(np.float32)
    out = arr * (1.0 - mask[:, :, None]) + 249.0 * mask[:, :, None]
    out = out * 0.94 + 240.0 * 0.06     # 轻微白幕
    return np.clip(out, 0, 255).astype(np.uint8)


def make_pair(
    weather: str,
    size: Tuple[int, int] = (384, 512),
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成一对 (退化图, 清晰图)。

    :param weather: rain / haze / snow / mixed
    """
    clean = make_clean_scene(size=size, seed=seed)
    key = (weather or "").lower()
    if key == "rain":
        degraded = add_rain(clean, seed=seed)
    elif key == "haze":
        degraded = add_haze(clean, seed=seed)
    elif key == "snow":
        degraded = add_snow(clean, seed=seed)
    elif key == "mixed":
        degraded = add_snow(add_haze(clean, seed=seed, beta=1.0), seed=seed, n_flakes=400)
    else:
        raise ValueError(f"未知天气类型: {weather}，可选 {WEATHERS} 或 mixed")
    return degraded, clean


def degradation_strength(degraded: np.ndarray, clean: np.ndarray) -> float:
    """粗略衡量退化强度（灰度均值差 + 标准差比），生成样本时打印用。"""
    d = to_gray(degraded)
    c = to_gray(clean)
    return float(abs(d.mean() - c.mean()) + abs(d.std() - c.std()))
