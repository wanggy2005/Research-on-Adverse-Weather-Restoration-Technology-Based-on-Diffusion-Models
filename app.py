"""
恶劣天气图像复原系统 —— 界面启动入口。

用法:
    python app.py                          # 正常启动（默认深色主题，自动选 GPU/CPU）
    set AWR_THEME=light && python app.py   # 浅色主题
    set AWR_DEVICE=cpu && python app.py    # 强制用 CPU
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 保证从任意目录运行都能 import 到项目包
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        print("缺少 PyQt5，请执行: pip install PyQt5")
        return 1

    # 高分屏缩放必须在创建 QApplication 之前设置
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    from ui.main_window import MainWindow
    from ui.theme import apply_theme

    app = QApplication(sys.argv)
    app.setApplicationName("恶劣天气图像复原系统")
    apply_theme(app, dark=os.environ.get("AWR_THEME", "dark").lower() != "light")

    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
