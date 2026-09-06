"""
=============================================================================
全项目最重要的文件：统一复原接口契约（Interface Contract）
=============================================================================

所有复原模型（扩散模型、传统算法）都必须继承 BaseRestorer。
UI 层与评测层只依赖本文件定义的接口，不关心底层是什么模型。

*** 修改本文件的任何签名，必须先在群里同步并由组长确认。 ***

约定（非常重要，三组都按这个来）:
  - 图像统一格式: numpy.ndarray, dtype=uint8, shape=(H, W, 3), 通道顺序 RGB
  - restore() 的输出尺寸必须与输入尺寸完全一致
  - 进度回调 progress_cb(step, total, preview)：
        step    当前完成的步数（1-based）
        total   总步数
        preview 中间结果预览图（RGB uint8）或 None
  - 取消回调 cancel_cb() -> bool：返回 True 时实现方必须尽快抛出 RestoreCancelled
=============================================================================
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

# 进度回调: (当前步, 总步数, 中间预览图或 None) -> None
ProgressCallback = Callable[[int, int, Optional[np.ndarray]], None]
# 取消回调: () -> 是否需要取消
CancelCallback = Callable[[], bool]


class RestoreCancelled(Exception):
    """用户主动取消了推理。UI 捕获后只需提示"已取消"，不算错误。"""


class WeightNotFoundError(FileNotFoundError):
    """权重文件缺失。UI 捕获后应提示用户去 weights/ 目录放权重。"""


@dataclass
class RestoreResult:
    """一次复原的完整结果，UI 和评测都用它，避免各自定义结构。"""

    image: np.ndarray                    # 复原后图像, RGB uint8
    elapsed: float = 0.0                 # 耗时(秒)
    steps: int = 0                       # 实际采样步数
    device: str = "cpu"                  # 实际使用的设备
    restorer: str = ""                   # 模型标识, 如 weatherdiff / classical / dcp
    extra: Dict[str, Any] = field(default_factory=dict)  # 其他信息(显存占用等)

    @property
    def elapsed_text(self) -> str:
        return f"{self.elapsed:.2f} s"


class BaseRestorer(abc.ABC):
    """
    复原器抽象基类。

    子类必须实现:
        load_model()  加载权重（只会被调用一次，之后由上层缓存实例）
        restore()     执行复原

    子类可选覆盖:
        unload()      释放显存
        default_steps 属性，用于 UI 显示进度条总步数的预估值
    """

    #: 注册名，与 configs/*.yaml 里的 restorer 字段对应
    name: str = "base"
    #: 界面上显示的名字
    display_name: str = "未命名模型"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = config or {}
        self.loaded: bool = False
        self.device: str = "cpu"

    # ------------------------------------------------------------------ #
    # 子类必须实现
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def load_model(self) -> None:
        """加载权重并把 self.loaded 置 True。权重不存在应抛 WeightNotFoundError。"""

    @abc.abstractmethod
    def restore(
        self,
        image: np.ndarray,
        progress_cb: Optional[ProgressCallback] = None,
        cancel_cb: Optional[CancelCallback] = None,
    ) -> np.ndarray:
        """
        对单张退化图做复原。

        :param image:       RGB uint8, (H, W, 3)
        :param progress_cb: 进度回调，可为 None
        :param cancel_cb:   取消回调，可为 None
        :return:            RGB uint8, (H, W, 3)，尺寸与输入一致
        :raises RestoreCancelled: 用户取消
        """

    # ------------------------------------------------------------------ #
    # 通用能力，子类一般不需要改
    # ------------------------------------------------------------------ #
    def restore_with_stats(
        self,
        image: np.ndarray,
        progress_cb: Optional[ProgressCallback] = None,
        cancel_cb: Optional[CancelCallback] = None,
    ) -> RestoreResult:
        """带计时的复原入口。**UI 与评测统一调用这个方法**，不要直接调 restore()。"""
        self.validate_image(image)
        if not self.loaded:
            self.load_model()

        t0 = time.perf_counter()
        output = self.restore(image, progress_cb=progress_cb, cancel_cb=cancel_cb)
        elapsed = time.perf_counter() - t0

        self.validate_image(output, name="输出图像")
        if output.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"复原输出尺寸 {output.shape[:2]} 与输入 {image.shape[:2]} 不一致，"
                f"请在模型内部恢复到原始尺寸"
            )
        return RestoreResult(
            image=output,
            elapsed=elapsed,
            steps=self.default_steps,
            device=self.device,
            restorer=self.name,
        )

    def unload(self) -> None:
        """释放资源。默认只置标记，需要清显存的子类自行覆盖。"""
        self.loaded = False

    @property
    def default_steps(self) -> int:
        """进度条总步数预估值，默认取配置里的采样步数。"""
        sampling = self.config.get("sampling") or {}
        return int(sampling.get("timesteps", 1) or 1)

    # ------------------------------------------------------------------ #
    # 给子类用的小工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def validate_image(image: Any, name: str = "输入图像") -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"{name} 必须是 numpy.ndarray, 实际为 {type(image)}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{name} 必须是 (H, W, 3) 的 RGB 图, 实际 shape={image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"{name} 必须是 uint8, 实际 dtype={image.dtype}")

    @staticmethod
    def check_cancel(cancel_cb: Optional[CancelCallback]) -> None:
        """在耗时循环里定期调用；被取消时抛 RestoreCancelled。"""
        if cancel_cb is not None and cancel_cb():
            raise RestoreCancelled("用户取消了复原任务")

    @staticmethod
    def report(
        progress_cb: Optional[ProgressCallback],
        step: int,
        total: int,
        preview: Optional[np.ndarray] = None,
    ) -> None:
        """安全地上报进度：回调内部异常不允许影响推理主流程。"""
        if progress_cb is None:
            return
        try:
            progress_cb(step, total, preview)
        except Exception:  # noqa: BLE001  回调异常不能中断推理
            pass

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__} name={self.name} loaded={self.loaded}>"
