"""
传统方法参数调优（评测组用）。

对 data/samples 里的样本做小规模网格搜索，找出 configs/classical.yaml 里
spike_tau / 中值窗口 / 是否去雾 的最佳组合，并打印 PSNR/SSIM 对比。

用法:
    python scripts/tune_classical.py                 # 三类天气都调
    python scripts/tune_classical.py --weather rain  # 只调雨
    python scripts/tune_classical.py --limit 2
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import DATA_DIR, get_config  # noqa: E402
from core.image_utils import list_images, load_image  # noqa: E402
from core.metrics import psnr, ssim  # noqa: E402
from core.registry import build_restorer  # noqa: E402
from core.synth import WEATHERS  # noqa: E402

SAMPLES_DIR = DATA_DIR / "samples"

# 搜索空间（想扩大范围直接改这里）
GRID = {
    "dehaze": ["auto", "true", "false"],
    "spike_tau": [0.008, 0.015, 0.025],
    "detail_median": [3, 5],
    "second_median": [0, 5],
    "sharpen": [0.0, 0.25],
}


def evaluate(weather: str, params: dict, limit: int) -> tuple:
    cfg = get_config("classical")
    cfg["classical"] = {**(cfg.get("classical") or {}), **params}
    restorer = build_restorer(cfg)
    restorer.load_model()

    in_dir = SAMPLES_DIR / weather / "input"
    gt_dir = SAMPLES_DIR / weather / "gt"
    ps: List[float] = []
    ss: List[float] = []
    for f in list_images(in_dir)[:limit]:
        gt_file = gt_dir / f.name
        if not gt_file.is_file():
            continue
        out = restorer.restore_with_stats(load_image(f)).image
        clean = load_image(gt_file)
        ps.append(psnr(out, clean))
        ss.append(ssim(out, clean))
    restorer.unload()
    if not ps:
        return 0.0, 0.0
    return sum(ps) / len(ps), sum(ss) / len(ss)


def main() -> int:
    parser = argparse.ArgumentParser(description="传统方法参数网格搜索")
    parser.add_argument("--weather", default=",".join(WEATHERS))
    parser.add_argument("--limit", type=int, default=2, help="每类用几张图评估")
    parser.add_argument("--top", type=int, default=5, help="打印前几名")
    args = parser.parse_args()

    keys = list(GRID)
    combos = [dict(zip(keys, values)) for values in product(*(GRID[k] for k in keys))]
    print(f"搜索空间: {len(combos)} 组参数\n")

    for weather in [w.strip() for w in args.weather.split(",") if w.strip()]:
        in_dir = SAMPLES_DIR / weather / "input"
        if not list_images(in_dir):
            print(f"跳过 {weather}: 没有样本，请先跑 scripts/make_samples.py")
            continue

        # 退化图基线
        gt_dir = SAMPLES_DIR / weather / "gt"
        base = []
        for f in list_images(in_dir)[: args.limit]:
            gt_file = gt_dir / f.name
            if gt_file.is_file():
                base.append(psnr(load_image(f), load_image(gt_file)))
        baseline = sum(base) / len(base) if base else 0.0

        print(f"{'=' * 78}")
        print(f"天气 {weather}  (退化图基线 PSNR={baseline:.2f})")
        print(f"{'=' * 78}")

        results = []
        for i, params in enumerate(combos, start=1):
            p, s = evaluate(weather, params, args.limit)
            results.append((p, s, params))
            print(f"\r  进度 {i}/{len(combos)}  当前最好 PSNR="
                  f"{max(r[0] for r in results):.2f}".ljust(70), end="")
        print()

        results.sort(key=lambda r: r[0], reverse=True)
        for rank, (p, s, params) in enumerate(results[: args.top], start=1):
            gain = p - baseline
            print(f"  #{rank}  PSNR={p:6.2f} (+{gain:5.2f})  SSIM={s:.4f}  {params}")
        print()

    print("把最好的参数写回 configs/classical.yaml 的 classical: 段即可。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
