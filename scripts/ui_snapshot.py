"""
界面渲染自检：不弹窗，直接把三个 Tab 渲染成 PNG，方便远程/无人值守验证 UI。

用法:
    python scripts/ui_snapshot.py                 # 用系统平台（字体正常，窗口会一闪）
    python scripts/ui_snapshot.py --offscreen      # 无桌面环境（可能缺字体）
    python scripts/ui_snapshot.py --theme light
输出:
    outputs/ui_preview/tab1_单图复原_dark.png 等
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import OUTPUT_DIR, ensure_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme", default="dark", choices=["dark", "light"])
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=880)
    parser.add_argument("--offscreen", action="store_true", help="无桌面环境下渲染（字体可能缺失）")
    args = parser.parse_args()

    if args.offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from PyQt5.QtCore import Qt  # noqa: PLC0415
    from PyQt5.QtWidgets import QApplication  # noqa: PLC0415

    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)

    from ui.main_window import MainWindow  # noqa: PLC0415
    from ui.theme import apply_theme  # noqa: PLC0415

    apply_theme(app, dark=args.theme == "dark")
    window = MainWindow()
    window.resize(args.width, args.height)
    window.show()
    app.processEvents()

    # 单图页先载入一张示例，截图才有内容
    try:
        window.single_page.load_next_sample()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 载入示例失败（可忽略）: {exc}")
    app.processEvents()

    out_dir = ensure_dir(OUTPUT_DIR / "ui_preview")
    for i in range(window.tabs.count()):
        window.tabs.setCurrentIndex(i)
        app.processEvents()
        name = window.tabs.tabText(i)
        path = out_dir / f"tab{i + 1}_{name}_{args.theme}.png"
        window.grab().save(str(path))
        print(f"[ok] {path}")

    print(f"\n共 {window.tabs.count()} 张，输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
