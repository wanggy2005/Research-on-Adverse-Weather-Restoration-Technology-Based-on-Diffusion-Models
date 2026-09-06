"""
=============================================================================
后台线程（界面绝不允许在主线程跑推理，否则一定卡死）
=============================================================================

提供两个 QThread:
  RestoreWorker  单图复原
  EvalWorker     批量评测

信号约定（界面只连信号，不碰模型）:
  loading()                     开始加载权重
  progress(int step, int total) 采样进度
  preview(object ndarray)       中间预览图
  finished(object RestoreResult / list[EvalRow])
  failed(str message)
  cancelled()
=============================================================================
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from core.base import RestoreCancelled, RestoreResult, WeightNotFoundError
from engine.evaluate import EvalRow, evaluate_dataset, write_csv
from engine.inference import ModelCache, restore_array


class RestoreWorker(QThread):
    """单图复原线程。"""

    loading = pyqtSignal()
    progress = pyqtSignal(int, int)
    preview = pyqtSignal(object)
    finished_ok = pyqtSignal(object)     # RestoreResult
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        cache: ModelCache,
        weather: str,
        image: np.ndarray,
        overrides: Optional[Dict[str, Any]] = None,
        emit_preview: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._cache = cache
        self._weather = weather
        self._image = image
        self._overrides = overrides or {}
        self._emit_preview = emit_preview
        self._cancel = False

    def cancel(self) -> None:
        """主线程调用；worker 内部会在下一个采样步检测到并退出。"""
        self._cancel = True

    def run(self) -> None:  # noqa: D102
        try:
            self.loading.emit()
            restorer = self._cache.get(self._weather, self._overrides)

            def on_progress(step: int, total: int, preview: Optional[np.ndarray]) -> None:
                self.progress.emit(int(step), int(total))
                if self._emit_preview and preview is not None:
                    self.preview.emit(preview)

            result: RestoreResult = restore_array(
                restorer,
                self._image,
                progress_cb=on_progress,
                cancel_cb=lambda: self._cancel,
            )
            self.finished_ok.emit(result)
        except RestoreCancelled:
            self.cancelled.emit()
        except WeightNotFoundError as exc:
            self.failed.emit(f"权重缺失：\n{exc}")
        except Exception as exc:  # noqa: BLE001 线程里的异常必须自己兜住
            import traceback

            traceback.print_exc()
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class EvalWorker(QThread):
    """批量评测线程。"""

    loading = pyqtSignal()
    item_done = pyqtSignal(int, int, object)   # (已完成, 总数, EvalRow)
    finished_ok = pyqtSignal(object, object)   # (List[EvalRow], summary dict)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        cache: ModelCache,
        weather: str,
        input_dir: str,
        gt_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        metrics: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        csv_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._cache = cache
        self._weather = weather
        self._input_dir = input_dir
        self._gt_dir = gt_dir or None
        self._output_dir = output_dir or None
        self._metrics = list(metrics) if metrics else None
        self._limit = limit
        self._csv_path = csv_path
        self._overrides = overrides or {}
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D102
        try:
            self.loading.emit()
            restorer = self._cache.get(self._weather, self._overrides)

            def on_progress(done: int, total: int, row: Optional[EvalRow]) -> None:
                self.item_done.emit(int(done), int(total), row)

            rows, summary = evaluate_dataset(
                restorer,
                input_dir=self._input_dir,
                gt_dir=self._gt_dir,
                output_dir=self._output_dir,
                metrics=self._metrics,
                limit=self._limit,
                weather=self._weather,
                save_outputs=bool(self._output_dir),
                progress_cb=on_progress,
                cancel_cb=lambda: self._cancel,
            )
            if self._csv_path:
                write_csv(rows, self._csv_path)
            if self._cancel:
                self.cancelled.emit()
            else:
                self.finished_ok.emit(rows, summary)
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            self.failed.emit(f"{type(exc).__name__}: {exc}")


__all__ = ["RestoreWorker", "EvalWorker"]
