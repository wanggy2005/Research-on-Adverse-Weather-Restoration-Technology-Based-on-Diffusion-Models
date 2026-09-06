"""
生成合成样本数据（第一次运行系统时先执行这个）。

会在 data/samples/ 下生成"退化图 + GT"配对，让整个系统在没有下载
任何真实数据集、没有任何权重的情况下就能完整跑起来并给出 PSNR/SSIM。

用法:
    python scripts/make_samples.py                 # 每类天气 4 张
    python scripts/make_samples.py --count 8       # 每类 8 张
    python scripts/make_samples.py --size 512x768  # 指定分辨率
    python scripts/make_samples.py --force         # 覆盖已有样本

生成结构:
    data/samples/rain/input   rain/gt
    data/samples/haze/input   haze/gt
    data/samples/snow/input   snow/gt
    data/samples/real/input            (混合退化，无 GT，用于无参考指标演示)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import DATA_DIR, ensure_dir  # noqa: E402
from core.image_utils import save_image  # noqa: E402
from core.metrics import psnr, ssim  # noqa: E402
from core.synth import WEATHERS, make_pair  # noqa: E402

SAMPLES_DIR = DATA_DIR / "samples"


def parse_size(text: str):
    if "x" not in text.lower():
        raise argparse.ArgumentTypeError("分辨率格式应为 高x宽，例如 384x512")
    h, w = text.lower().split("x")
    return int(h), int(w)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成合成天气退化样本")
    parser.add_argument("--count", type=int, default=4, help="每类天气生成几张")
    parser.add_argument("--size", type=parse_size, default=(384, 512), help="分辨率 高x宽")
    parser.add_argument("--force", action="store_true", help="已存在时也重新生成")
    args = parser.parse_args()

    if SAMPLES_DIR.exists() and any(SAMPLES_DIR.rglob("*.png")) and not args.force:
        print(f"样本已存在: {SAMPLES_DIR}")
        print("如需重新生成请加 --force")
        return 0

    print(f"生成样本到: {SAMPLES_DIR}")
    print(f"分辨率: {args.size[0]}x{args.size[1]}   每类数量: {args.count}\n")

    for weather in WEATHERS:
        in_dir = ensure_dir(SAMPLES_DIR / weather / "input")
        gt_dir = ensure_dir(SAMPLES_DIR / weather / "gt")
        print(f"[{weather}]")
        for i in range(args.count):
            degraded, clean = make_pair(weather, size=args.size, seed=i)
            name = f"{i:02d}.png"
            save_image(in_dir / name, degraded)
            save_image(gt_dir / name, clean)
            # 打印退化图相对 GT 的指标，作为"复原前"的基线参考
            print(
                f"  {name}  退化前基线: PSNR={psnr(degraded, clean):5.2f} dB  "
                f"SSIM={ssim(degraded, clean):.4f}"
            )

    # 无 GT 的"真实场景"样例：多种退化叠加
    real_dir = ensure_dir(SAMPLES_DIR / "real" / "input")
    print("\n[real] 混合退化(无 GT，仅用于 NIQE/BRISQUE 演示)")
    for i in range(3):
        degraded, _ = make_pair("mixed", size=args.size, seed=100 + i)
        save_image(real_dir / f"mixed_{i:02d}.png", degraded)
        print(f"  mixed_{i:02d}.png")

    print("\n完成。接下来可以:")
    print("  python scripts/demo.py            # 命令行一键演示 + 指标对比表")
    print("  python app.py                     # 打开可视化界面")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
