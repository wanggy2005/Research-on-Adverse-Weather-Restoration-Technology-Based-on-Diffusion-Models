"""
=============================================================================
指标模块（评测组主战场）
=============================================================================

有参考指标（需要 GT）:  PSNR、SSIM、LPIPS
无参考指标（不需要 GT）: NIQE、BRISQUE
数据集级指标:            FID

设计原则：
  - 所有函数输入统一为 RGB uint8 (H, W, 3)
  - 第三方库全部**延迟导入**，没装也不会影响系统启动，只是对应指标不可用
  - 不可用时抛 MetricUnavailable，由上层决定是跳过还是提示
=============================================================================
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .image_utils import to_float01, to_uint8

#: 有参考指标名
FULL_REFERENCE_METRICS: List[str] = ["psnr", "ssim", "lpips"]
#: 无参考指标名
NO_REFERENCE_METRICS: List[str] = ["niqe", "brisque"]
#: 指标是"越大越好"还是"越小越好"，界面画图和排序要用
HIGHER_IS_BETTER: Dict[str, bool] = {
    "psnr": True,
    "ssim": True,
    "lpips": False,
    "niqe": False,
    "brisque": False,
    "fid": False,
}
#: 界面表头显示名
METRIC_LABELS: Dict[str, str] = {
    "psnr": "PSNR↑",
    "ssim": "SSIM↑",
    "lpips": "LPIPS↓",
    "niqe": "NIQE↓",
    "brisque": "BRISQUE↓",
    "fid": "FID↓",
}


class MetricUnavailable(RuntimeError):
    """指标依赖库未安装或不可用。"""


# --------------------------------------------------------------------------- #
# 有参考指标
# --------------------------------------------------------------------------- #
def psnr(pred: np.ndarray, gt: np.ndarray, data_range: float = 255.0) -> float:
    """峰值信噪比。纯 numpy 实现，无第三方依赖。"""
    a = np.asarray(to_uint8(pred), dtype=np.float64)
    b = np.asarray(to_uint8(gt), dtype=np.float64)
    _check_same_shape(a, b)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """结构相似度。优先用 scikit-image，未安装则用内置实现兜底。"""
    a = to_uint8(pred)
    b = to_uint8(gt)
    _check_same_shape(a, b)
    try:
        from skimage.metrics import structural_similarity  # noqa: PLC0415

        return float(
            structural_similarity(a, b, channel_axis=2, data_range=255)
        )
    except ImportError:
        return _ssim_fallback(a, b)


def lpips_score(pred: np.ndarray, gt: np.ndarray, net: str = "alex") -> float:
    """LPIPS 感知相似度（越小越好）。需要 pip install lpips torch。"""
    model = _get_lpips_model(net)
    import torch  # noqa: PLC0415

    def prep(x: np.ndarray) -> "torch.Tensor":
        t = torch.from_numpy(to_float01(x)).permute(2, 0, 1).unsqueeze(0)
        return (t * 2.0 - 1.0).to(_lpips_device())

    with torch.no_grad():
        return float(model(prep(pred), prep(gt)).item())


# --------------------------------------------------------------------------- #
# 无参考指标
# --------------------------------------------------------------------------- #
def niqe(image: np.ndarray) -> float:
    """NIQE（越小越好）。需要 pip install pyiqa。"""
    return _pyiqa_score("niqe", image)


def brisque(image: np.ndarray) -> float:
    """BRISQUE（越小越好）。需要 pip install pyiqa。"""
    return _pyiqa_score("brisque", image)


# --------------------------------------------------------------------------- #
# 数据集级指标
# --------------------------------------------------------------------------- #
def fid(pred_dir: str | os.PathLike, gt_dir: str | os.PathLike, batch_size: int = 16) -> float:
    """
    FID（越小越好），比较两个目录的分布差异。需要 pip install pytorch-fid。
    注意：每个目录建议至少 50 张图，否则数值不可靠。
    """
    try:
        from pytorch_fid import fid_score  # noqa: PLC0415
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise MetricUnavailable("FID 需要 pytorch-fid: pip install pytorch-fid") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return float(
        fid_score.calculate_fid_given_paths(
            [str(pred_dir), str(gt_dir)], batch_size=batch_size, device=device, dims=2048
        )
    )


# --------------------------------------------------------------------------- #
# 组合入口：engine/evaluate.py 与界面都调这两个
# --------------------------------------------------------------------------- #
def compute_metrics(
    pred: np.ndarray,
    gt: Optional[np.ndarray] = None,
    metrics: Optional[Sequence[str]] = None,
    silent: bool = True,
) -> Dict[str, Optional[float]]:
    """
    计算单张图的所有指标。

    :param pred:    复原结果 RGB uint8
    :param gt:      GT，可为 None（此时只算无参考指标）
    :param metrics: 指定要算的指标，默认 PSNR/SSIM + NIQE/BRISQUE
    :param silent:  True 时某个指标不可用只记 None，不抛异常
    :return:        {指标名: 数值或 None}
    """
    if metrics is None:
        metrics = ["psnr", "ssim", "niqe", "brisque"]

    result: Dict[str, Optional[float]] = {}
    for name in metrics:
        key = name.lower()
        try:
            if key in FULL_REFERENCE_METRICS:
                if gt is None:
                    result[key] = None
                    continue
                if key == "psnr":
                    result[key] = psnr(pred, gt)
                elif key == "ssim":
                    result[key] = ssim(pred, gt)
                elif key == "lpips":
                    result[key] = lpips_score(pred, gt)
            elif key == "niqe":
                result[key] = niqe(pred)
            elif key == "brisque":
                result[key] = brisque(pred)
            else:
                raise MetricUnavailable(f"未知指标: {name}")
        except Exception as exc:  # noqa: BLE001
            if not silent:
                raise
            result[key] = None
            _warn_once(f"[metrics] 指标 {key} 计算失败/不可用: {exc}")
    return result


def summarize(rows: Sequence[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    """对多张图的指标求均值（自动忽略 None 与 inf）。"""
    if not rows:
        return {}
    keys = [k for k in rows[0] if k in HIGHER_IS_BETTER]
    out: Dict[str, Optional[float]] = {}
    for k in keys:
        vals = [
            float(r[k])
            for r in rows
            if r.get(k) is not None and np.isfinite(float(r[k]))
        ]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def format_value(value: Optional[float], digits: int = 4) -> str:
    """界面/CSV 统一的数值格式化。"""
    if value is None:
        return "-"
    if not np.isfinite(value):
        return "inf"
    return f"{value:.{digits}f}"


def available_metrics() -> Dict[str, bool]:
    """探测各指标当前是否可用，界面"关于"页可以展示。"""
    status = {"psnr": True, "ssim": True}
    try:
        import skimage  # noqa: F401,PLC0415
        status["ssim"] = True
    except ImportError:
        status["ssim"] = True  # 有兜底实现
    for name, module in (("lpips", "lpips"), ("niqe", "pyiqa"), ("brisque", "pyiqa"), ("fid", "pytorch_fid")):
        try:
            __import__(module)
            status[name] = True
        except ImportError:
            status[name] = False
    return status


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
_WARNED: set = set()
_LPIPS_CACHE: Dict[str, object] = {}
_PYIQA_CACHE: Dict[str, object] = {}


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(msg)


def _check_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"预测与 GT 尺寸不一致: {a.shape} vs {b.shape}，请先对齐尺寸")


def _lpips_device() -> str:
    try:
        import torch  # noqa: PLC0415

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_lpips_model(net: str):
    if net in _LPIPS_CACHE:
        return _LPIPS_CACHE[net]
    try:
        import lpips  # noqa: PLC0415
    except ImportError as exc:
        raise MetricUnavailable("LPIPS 需要: pip install lpips") from exc
    model = lpips.LPIPS(net=net).to(_lpips_device()).eval()
    _LPIPS_CACHE[net] = model
    return model


def _pyiqa_score(metric_name: str, image: np.ndarray) -> float:
    try:
        import pyiqa  # noqa: PLC0415
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise MetricUnavailable(f"{metric_name.upper()} 需要: pip install pyiqa") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if metric_name not in _PYIQA_CACHE:
        _PYIQA_CACHE[metric_name] = pyiqa.create_metric(metric_name, device=device)
    metric = _PYIQA_CACHE[metric_name]
    tensor = torch.from_numpy(to_float01(image)).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        score = metric(tensor)
    return float(score.item() if hasattr(score, "item") else float(score))


def _ssim_fallback(a: np.ndarray, b: np.ndarray) -> float:
    """scikit-image 缺失时的 SSIM 兜底实现（7x7 均值窗口，与官方实现有细微差异）。"""
    from .image_utils import box_filter  # noqa: PLC0415

    x = a.astype(np.float64)
    y = b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    r = 3
    mu_x = box_filter(x, r)
    mu_y = box_filter(y, r)
    sigma_x = box_filter(x * x, r) - mu_x ** 2
    sigma_y = box_filter(y * y, r) - mu_y ** 2
    sigma_xy = box_filter(x * y, r) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return float(np.mean(num / np.maximum(den, 1e-12)))


__all__ = [
    "FULL_REFERENCE_METRICS",
    "NO_REFERENCE_METRICS",
    "HIGHER_IS_BETTER",
    "METRIC_LABELS",
    "MetricUnavailable",
    "psnr",
    "ssim",
    "lpips_score",
    "niqe",
    "brisque",
    "fid",
    "compute_metrics",
    "summarize",
    "format_value",
    "available_metrics",
]
