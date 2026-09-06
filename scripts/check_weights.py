"""
权重体检脚本（模型组用）。

不需要 UNet 实现完成就能跑：它只读取 checkpoint，打印里面的结构，
帮助确认 state_dict 的 key 命名、是否有 module. 前缀、是否带 EMA。
这是移植 unet.py 之前必须先做的一步。

用法:
    python scripts/check_weights.py
    python scripts/check_weights.py --config rain --limit 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config, weight_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="检查权重文件结构")
    parser.add_argument("--config", default="rain", help="配置名: rain/haze/snow/allweather")
    parser.add_argument("--limit", type=int, default=20, help="打印前 N 个参数名")
    parser.add_argument("--try-build", action="store_true", help="尝试真正构建模型并加载权重")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = weight_path(cfg)
    print(f"配置: {cfg.get('_config_path')}")
    print(f"权重: {ckpt_path}")
    if not ckpt_path or not Path(ckpt_path).is_file():
        print("权重不存在，请先执行: python scripts/download_weights.py")
        return 1

    try:
        import torch
    except ImportError:
        print("需要安装 torch 才能读取权重")
        return 1

    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"\ncheckpoint 类型: {type(ckpt)}")
    if isinstance(ckpt, dict):
        print(f"顶层键: {list(ckpt.keys())}")
        for key in ("epoch", "step"):
            if key in ckpt:
                print(f"  {key} = {ckpt[key]}")

    state = None
    for key in ("state_dict", "model", "net", "params"):
        if isinstance(ckpt, dict) and isinstance(ckpt.get(key), dict):
            state = ckpt[key]
            print(f"\n参数字典位于: ckpt['{key}']，共 {len(state)} 项")
            break
    if state is None and isinstance(ckpt, dict):
        state = ckpt
        print(f"\n直接把顶层当作参数字典，共 {len(state)} 项")

    if state:
        has_module = any(str(k).startswith("module.") for k in state)
        print(f"是否带 DataParallel 的 module. 前缀: {has_module}")
        print(f"\n前 {args.limit} 个参数名:")
        for i, (k, v) in enumerate(state.items()):
            if i >= args.limit:
                break
            shape = tuple(v.shape) if hasattr(v, "shape") else "-"
            print(f"  {k:<60} {shape}")

    if isinstance(ckpt, dict) and "ema_helper" in ckpt:
        ema = ckpt["ema_helper"]
        n = len(ema.get("shadow", ema)) if isinstance(ema, dict) else "?"
        print(f"\n包含 EMA 权重，影子参数数量: {n}（推理时应用 EMA 效果更好）")

    if args.try_build:
        print("\n尝试构建模型并加载权重...")
        from core.registry import build_restorer

        restorer = build_restorer(cfg)
        restorer.load_model()
        print("加载成功！可以开始跑推理了。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
