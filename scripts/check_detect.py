"""
退化类型自动判别的准确率检查（传统方法的 profile 自动识别）。

用法:
    python scripts/check_detect.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.baseline.classical import ClassicalRestorer  # noqa: E402
from core.config import DATA_DIR  # noqa: E402
from core.image_utils import list_images, load_image, to_float01  # noqa: E402
from core.synth import WEATHERS  # noqa: E402

SAMPLES_DIR = DATA_DIR / "samples"


def main() -> int:
    total = 0
    correct = 0
    print(f"{'文件':<28}{'真实':<10}{'判别':<10}{'结果'}")
    print("-" * 60)
    for weather in WEATHERS:
        for f in list_images(SAMPLES_DIR / weather / "input"):
            img = to_float01(load_image(f))
            got = ClassicalRestorer.detect_profile(img)
            ok = got == weather
            total += 1
            correct += int(ok)
            print(f"{weather + '/' + f.name:<28}{weather:<10}{got:<10}{'OK' if ok else 'X'}")

    real_dir = SAMPLES_DIR / "real" / "input"
    for f in list_images(real_dir):
        got = ClassicalRestorer.detect_profile(to_float01(load_image(f)))
        print(f"{'real/' + f.name:<28}{'mixed':<10}{got:<10}-")

    print("-" * 60)
    if total:
        print(f"准确率: {correct}/{total} = {correct / total * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
