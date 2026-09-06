"""
批量评测：遍历数据集 -> 复原 -> 计算指标 -> 汇总 -> 导出 CSV。

命令行 (scripts/run_eval.py) 与界面"批量评测"页共用本模块，
所以这里不允许出现任何 print 之外的界面代码。
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.base import BaseRestorer, CancelCallback
from core.image_utils import load_image, match_pairs, resize, save_image
from core.metrics import HIGHER_IS_BETTER, compute_metrics, format_value, summarize

#: 评测进度回调 (已完成数, 总数, 当前行或 None)
EvalProgress = Callable[[int, int, Optional["EvalRow"]], None]


@dataclass
class EvalRow:
    """一张图的评测结果，界面表格一行 = 一个 EvalRow。"""

    filename: str
    weather: str = ""
    restorer: str = ""
    has_gt: bool = False
    elapsed: float = 0.0
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    error: Optional[str] = None

    def to_flat_dict(self) -> Dict[str, Any]:
        """摊平成 CSV 一行。"""
        row: Dict[str, Any] = {
            "filename": self.filename,
            "weather": self.weather,
            "restorer": self.restorer,
            "has_gt": int(self.has_gt),
            "elapsed_s": round(self.elapsed, 3),
        }
        for k, v in self.metrics.items():
            row[k] = format_value(v)
        row["error"] = self.error or ""
        return row


def evaluate_dataset(
    restorer: BaseRestorer,
    input_dir: str | os.PathLike,
    gt_dir: Optional[str | os.PathLike] = None,
    output_dir: Optional[str | os.PathLike] = None,
    metrics: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    weather: str = "",
    save_outputs: bool = True,
    progress_cb: Optional[EvalProgress] = None,
    cancel_cb: Optional[CancelCallback] = None,
) -> Tuple[List[EvalRow], Dict[str, Optional[float]]]:
    """
    对一个目录做批量评测。

    :param input_dir:  退化图目录
    :param gt_dir:     GT 目录，None 表示只算无参考指标
    :param output_dir: 复原结果保存目录（算 FID 需要它）
    :param metrics:    指标列表，默认 ["psnr", "ssim", "niqe"]
    :param limit:      只跑前 N 张（调试用）
    :return: (每张图的结果列表, 均值汇总)
    """
    if metrics is None:
        metrics = ["psnr", "ssim", "niqe"]

    pairs = match_pairs(input_dir, gt_dir)
    if limit:
        pairs = pairs[: int(limit)]
    total = len(pairs)

    rows: List[EvalRow] = []
    for idx, (img_path, gt_path) in enumerate(pairs, start=1):
        if cancel_cb is not None and cancel_cb():
            break

        row = EvalRow(
            filename=img_path.name,
            weather=weather,
            restorer=restorer.name,
            has_gt=gt_path is not None,
        )
        try:
            degraded = load_image(img_path)
            result = restorer.restore_with_stats(degraded, cancel_cb=cancel_cb)
            row.elapsed = result.elapsed

            if save_outputs and output_dir:
                save_image(Path(output_dir) / img_path.name, result.image)

            gt = None
            if gt_path is not None:
                gt = load_image(gt_path)
                if gt.shape[:2] != result.image.shape[:2]:
                    # 少数数据集 GT 与输入尺寸不同，统一缩放到复原结果尺寸
                    gt = resize(gt, (result.image.shape[1], result.image.shape[0]))
            row.metrics = compute_metrics(result.image, gt, metrics=metrics)
        except Exception as exc:  # noqa: BLE001 单张失败不中断整批
            row.error = str(exc)

        rows.append(row)
        if progress_cb:
            progress_cb(idx, total, row)

    summary = summarize([r.metrics for r in rows if not r.error])
    summary["elapsed_s"] = _mean([r.elapsed for r in rows if not r.error])
    return rows, summary


def write_csv(rows: Sequence[EvalRow], path: str | os.PathLike) -> str:
    """导出逐图指标 CSV（Excel 打开中文不乱码，用 utf-8-sig）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    flat = [r.to_flat_dict() for r in rows]
    if not flat:
        p.write_text("", encoding="utf-8-sig")
        return str(p)
    fieldnames = list(flat[0].keys())
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)
    return str(p)


def write_summary_csv(
    summaries: Dict[str, Dict[str, Optional[float]]], path: str | os.PathLike
) -> str:
    """
    导出对比汇总 CSV。

    :param summaries: {"rain/weatherdiff": {"psnr": 30.1, ...}, "rain/dcp": {...}}
                      正好是答辩要用的"方法 x 指标"对比表
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    metric_keys: List[str] = []
    for s in summaries.values():
        for k in s:
            if k not in metric_keys:
                metric_keys.append(k)
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["setting"] + metric_keys)
        for name, s in summaries.items():
            writer.writerow([name] + [format_value(s.get(k)) for k in metric_keys])
    return str(p)


def better_of(metric: str, a: Optional[float], b: Optional[float]) -> Optional[float]:
    """按指标方向取更优值，界面高亮"最优方法"时用。"""
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b) if HIGHER_IS_BETTER.get(metric, True) else min(a, b)


def _mean(values: Sequence[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return float(sum(values) / len(values)) if values else None


__all__ = [
    "EvalRow",
    "EvalProgress",
    "evaluate_dataset",
    "write_csv",
    "write_summary_csv",
    "better_of",
]
