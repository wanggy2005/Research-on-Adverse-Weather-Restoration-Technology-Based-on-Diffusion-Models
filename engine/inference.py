"""
推理编排：单图 / 单文件 / 批量。

界面层与命令行都调用这里的函数，保证行为一致。
ModelCache 负责"同一个配置只加载一次权重"，这是界面体验的关键。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.base import BaseRestorer, CancelCallback, ProgressCallback, RestoreResult
from core.config import get_config, load_config
from core.image_utils import load_image, save_image
from core.registry import build_restorer


class ModelCache:
    """
    权重缓存：按配置名缓存已加载的复原器实例。

    界面上切换天气类型时不会重复读盘、重复占显存；
    通过 release_others() 可以只保留当前使用的模型，控制显存。
    """

    def __init__(self) -> None:
        self._cache: Dict[str, BaseRestorer] = {}

    def get(self, weather: str, overrides: Optional[Dict[str, Any]] = None) -> BaseRestorer:
        """
        取一个已加载好的复原器。

        :param weather:   配置名，如 rain / haze / snow / dcp / classical
        :param overrides: 运行时参数覆盖，如 {"sampling": {"timesteps": 10}}
        """
        key = self._make_key(weather, overrides)
        if key not in self._cache:
            cfg = get_config(weather)
            if overrides:
                cfg = _deep_update(cfg, overrides)
            restorer = build_restorer(cfg)
            restorer.load_model()
            self._cache[key] = restorer
        elif overrides:
            # 采样步数/grid_r 这类参数变了不需要重新加载权重，直接更新配置即可
            cached = self._cache[key]
            cached.config = _deep_update(cached.config, overrides)
        return self._cache[key]

    def release_others(self, keep_weather: Optional[str] = None) -> None:
        for key in list(self._cache):
            if keep_weather is not None and key.startswith(f"{keep_weather}|"):
                continue
            self._cache.pop(key).unload()

    def clear(self) -> None:
        for key in list(self._cache):
            self._cache.pop(key).unload()

    @staticmethod
    def _make_key(weather: str, overrides: Optional[Dict[str, Any]]) -> str:
        # 采样步数等参数变化不需要重新加载权重，所以 key 只带影响权重的部分
        return f"{weather}|"


def restore_array(
    restorer: BaseRestorer,
    image: np.ndarray,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_cb: Optional[CancelCallback] = None,
) -> RestoreResult:
    """对内存中的一张图做复原。"""
    return restorer.restore_with_stats(image, progress_cb=progress_cb, cancel_cb=cancel_cb)


def restore_file(
    restorer: BaseRestorer,
    input_path: str | os.PathLike,
    output_path: Optional[str | os.PathLike] = None,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_cb: Optional[CancelCallback] = None,
) -> Tuple[np.ndarray, RestoreResult]:
    """
    对一个图片文件做复原。

    :return: (原始输入图, 复原结果)
    """
    image = load_image(input_path)
    result = restore_array(restorer, image, progress_cb=progress_cb, cancel_cb=cancel_cb)
    if output_path:
        save_image(output_path, result.image)
    return image, result


def restore_batch(
    restorer: BaseRestorer,
    files: Sequence[str | os.PathLike],
    output_dir: Optional[str | os.PathLike] = None,
    on_item: Optional[Callable[[int, int, Path, Optional[RestoreResult], Optional[str]], None]] = None,
    cancel_cb: Optional[CancelCallback] = None,
    step_cb: Optional[ProgressCallback] = None,
) -> List[Tuple[Path, Optional[RestoreResult], Optional[str]]]:
    """
    批量复原。

    :param on_item:  每处理完一张回调 (序号, 总数, 文件路径, 结果, 错误信息)
    :param step_cb:  单张图内部的采样进度回调
    :return: [(文件路径, 结果或 None, 错误信息或 None)]
    """
    results: List[Tuple[Path, Optional[RestoreResult], Optional[str]]] = []
    total = len(files)
    for idx, f in enumerate(files, start=1):
        path = Path(f)
        if cancel_cb is not None and cancel_cb():
            break
        try:
            out_path = Path(output_dir) / path.name if output_dir else None
            _, result = restore_file(
                restorer, path, out_path, progress_cb=step_cb, cancel_cb=cancel_cb
            )
            results.append((path, result, None))
            if on_item:
                on_item(idx, total, path, result, None)
        except Exception as exc:  # noqa: BLE001 单张失败不应中断整批
            results.append((path, None, str(exc)))
            if on_item:
                on_item(idx, total, path, None, str(exc))
    return results


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并配置（不修改原字典）。"""
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


__all__ = [
    "ModelCache",
    "restore_array",
    "restore_file",
    "restore_batch",
    "load_config",
]
