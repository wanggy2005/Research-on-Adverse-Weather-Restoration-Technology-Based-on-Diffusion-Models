"""
一键演示脚本：不需要权重、不需要 GPU，跑完直接得到指标对比表和对比图。

它做四件事:
  1. 样本不存在时自动生成（调用 make_samples 的逻辑）
  2. 对雨/雾/雪三类样本，依次用各方法复原
  3. 打印"退化前 vs 各方法"的 PSNR / SSIM 对比表
  4. 把 [退化图 | 复原图 | GT] 三联对比图保存到 outputs/demo/

用法:
    python scripts/demo.py                                   # 默认: 传统方法 + DCP
    python scripts/demo.py --methods classical,dcp,rain      # 加上扩散模型（需权重）
    python scripts/demo.py --limit 2                         # 每类只跑 2 张
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import DATA_DIR, OUTPUT_DIR, ensure_dir, get_config, weight_path  # noqa: E402
from core.image_utils import list_images, load_image, save_image, side_by_side  # noqa: E402
from core.metrics import format_value, psnr, ssim  # noqa: E402
from core.registry import build_restorer  # noqa: E402
from core.synth import WEATHERS  # noqa: E402

SAMPLES_DIR = DATA_DIR / "samples"
DEMO_OUT = OUTPUT_DIR / "demo"


def ensure_samples(count: int = 4) -> None:
    """样本不存在就自动生成。"""
    if SAMPLES_DIR.exists() and any(SAMPLES_DIR.rglob("*.png")):
        return
    print("未发现样本数据，正在自动生成...")
    from core.synth import make_pair  # noqa: PLC0415

    for weather in WEATHERS:
        in_dir = ensure_dir(SAMPLES_DIR / weather / "input")
        gt_dir = ensure_dir(SAMPLES_DIR / weather / "gt")
        for i in range(count):
            degraded, clean = make_pair(weather, seed=i)
            save_image(in_dir / f"{i:02d}.png", degraded)
            save_image(gt_dir / f"{i:02d}.png", clean)
    real_dir = ensure_dir(SAMPLES_DIR / "real" / "input")
    for i in range(3):
        degraded, _ = make_pair("mixed", seed=100 + i)
        save_image(real_dir / f"mixed_{i:02d}.png", degraded)
    print(f"样本已生成: {SAMPLES_DIR}\n")


def available_methods(requested: List[str]) -> List[str]:
    """过滤掉缺权重的方法，避免演示中途报错。"""
    ok: List[str] = []
    for name in requested:
        try:
            cfg = get_config(name)
        except FileNotFoundError:
            print(f"跳过 {name}: 找不到 configs/{name}.yaml")
            continue
        wp = weight_path(cfg)
        if wp and not Path(wp).is_file():
            print(f"跳过 {name}: 缺少权重 {wp}（先跑 scripts/download_weights.py）")
            continue
        ok.append(name)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="一键演示")
    parser.add_argument("--methods", default="classical,dcp", help="逗号分隔的配置名")
    parser.add_argument("--limit", type=int, default=2, help="每类天气跑几张")
    parser.add_argument("--weathers", default=",".join(WEATHERS), help="要跑的天气目录")
    args = parser.parse_args()

    ensure_samples()
    methods = available_methods([m.strip() for m in args.methods.split(",") if m.strip()])
    if not methods:
        print("没有可用的方法，退出")
        return 1

    ensure_dir(DEMO_OUT)
    # 结果表: rows[(weather, method)] = {"psnr":.., "ssim":.., "time":..}
    table: Dict[str, Dict[str, Optional[float]]] = {}

    for weather in [w.strip() for w in args.weathers.split(",") if w.strip()]:
        in_dir = SAMPLES_DIR / weather / "input"
        gt_dir = SAMPLES_DIR / weather / "gt"
        files = list_images(in_dir)[: args.limit]
        if not files:
            print(f"跳过 {weather}: {in_dir} 下没有图片")
            continue

        print(f"\n{'=' * 70}\n天气: {weather}   样本 {len(files)} 张\n{'=' * 70}")

        # 基线：退化图直接与 GT 比
        base_p, base_s = [], []
        for f in files:
            gt_file = gt_dir / f.name
            if not gt_file.is_file():
                continue
            degraded, clean = load_image(f), load_image(gt_file)
            base_p.append(psnr(degraded, clean))
            base_s.append(ssim(degraded, clean))
        if base_p:
            table[f"{weather} / 退化图(未处理)"] = {
                "psnr": sum(base_p) / len(base_p),
                "ssim": sum(base_s) / len(base_s),
                "time": 0.0,
            }
            print(f"  退化图基线      PSNR={sum(base_p)/len(base_p):6.2f}  "
                  f"SSIM={sum(base_s)/len(base_s):.4f}")

        for method in methods:
            restorer = build_restorer(get_config(method))
            restorer.load_model()
            ps, ss, ts = [], [], []
            for f in files:
                degraded = load_image(f)
                result = restorer.restore_with_stats(degraded)
                ts.append(result.elapsed)
                gt_file = gt_dir / f.name
                if gt_file.is_file():
                    clean = load_image(gt_file)
                    ps.append(psnr(result.image, clean))
                    ss.append(ssim(result.image, clean))
                    triptych = side_by_side(side_by_side(degraded, result.image), clean)
                else:
                    triptych = side_by_side(degraded, result.image)
                save_image(DEMO_OUT / f"{weather}_{method}_{f.stem}.png", triptych)

            key = f"{weather} / {method}"
            table[key] = {
                "psnr": (sum(ps) / len(ps)) if ps else None,
                "ssim": (sum(ss) / len(ss)) if ss else None,
                "time": (sum(ts) / len(ts)) if ts else None,
            }
            print(f"  {method:<14} PSNR={format_value(table[key]['psnr'], 2):>6}  "
                  f"SSIM={format_value(table[key]['ssim'])}  "
                  f"平均耗时={format_value(table[key]['time'], 2)}s")
            restorer.unload()

    # 汇总表
    print(f"\n{'=' * 70}\n汇总（PSNR/SSIM 越高越好）\n{'=' * 70}")
    print(f"{'设置':<32}{'PSNR':>10}{'SSIM':>10}{'耗时(s)':>12}")
    print("-" * 70)
    for key, val in table.items():
        print(
            f"{key:<32}{format_value(val['psnr'], 2):>10}"
            f"{format_value(val['ssim']):>10}{format_value(val['time'], 2):>12}"
        )

    csv_path = DEMO_OUT / "summary.csv"
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write("setting,psnr,ssim,elapsed_s\n")
        for key, val in table.items():
            f.write(
                f"{key},{format_value(val['psnr'], 3)},"
                f"{format_value(val['ssim'], 4)},{format_value(val['time'], 3)}\n"
            )

    print(f"\n对比图（退化 | 复原 | GT）已保存到: {DEMO_OUT}")
    print(f"汇总 CSV: {csv_path}")
    print("\n下一步: python app.py 打开可视化界面")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
