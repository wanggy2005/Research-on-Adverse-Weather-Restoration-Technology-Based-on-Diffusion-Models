"""
批量评测命令行入口。

用法示例:
    # 传统方法（无需权重/GPU，适合先跑通流程）
    python scripts/run_eval.py --config classical --input data/samples/rain/input --gt data/samples/rain/gt

    # 扩散模型 + 三类天气
    python scripts/run_eval.py --config rain --input data/raindrop/test_a/data --gt data/raindrop/test_a/gt

    # 一次跑多个方法做对照，自动生成汇总对比表
    python scripts/run_eval.py --config rain,dcp --input data/rainfog/input --gt data/rainfog/gt --metrics psnr,ssim,niqe

    # 顺便算 FID（需要 pytorch-fid，且 GT 目录图片数量足够）
    python scripts/run_eval.py --config rain --input ... --gt ... --fid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import OUTPUT_DIR, ensure_dir, get_config  # noqa: E402
from core.metrics import METRIC_LABELS, format_value  # noqa: E402
from core.registry import build_restorer  # noqa: E402
from engine.evaluate import evaluate_dataset, write_csv, write_summary_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="批量评测")
    parser.add_argument("--config", default="classical", help="配置名，逗号分隔可一次跑多个方法")
    parser.add_argument("--input", required=True, help="退化图目录")
    parser.add_argument("--gt", default=None, help="GT 目录（不给则只算无参考指标）")
    parser.add_argument("--out", default=None, help="结果输出根目录，默认 outputs/eval")
    parser.add_argument("--metrics", default="psnr,ssim", help="指标: psnr,ssim,lpips,niqe,brisque")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 张，0 为全部")
    parser.add_argument("--steps", type=int, default=0, help="覆盖采样步数")
    parser.add_argument("--grid-r", type=int, default=0, help="覆盖滑窗步长")
    parser.add_argument("--fid", action="store_true", help="额外计算 FID（需要 GT 目录）")
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    out_root = Path(args.out) if args.out else OUTPUT_DIR / "eval"
    ensure_dir(out_root)

    summaries = {}
    for name in [c.strip() for c in args.config.split(",") if c.strip()]:
        cfg = get_config(name)
        if args.steps:
            cfg.setdefault("sampling", {})["timesteps"] = args.steps
        if args.grid_r:
            cfg.setdefault("sampling", {})["grid_r"] = args.grid_r

        restorer = build_restorer(cfg)
        print(f"\n===== {name} ({restorer.display_name}) =====")
        restorer.load_model()

        pred_dir = out_root / name
        rows, summary = evaluate_dataset(
            restorer,
            input_dir=args.input,
            gt_dir=args.gt,
            output_dir=pred_dir,
            metrics=metrics,
            limit=args.limit or None,
            weather=name,
            progress_cb=_print_progress,
        )
        csv_path = write_csv(rows, out_root / f"{name}_per_image.csv")

        if args.fid and args.gt:
            try:
                from core.metrics import fid as compute_fid

                summary["fid"] = compute_fid(pred_dir, args.gt)
            except Exception as exc:  # noqa: BLE001
                print(f"  FID 计算失败: {exc}")

        summaries[name] = summary
        print(f"\n  逐图结果: {csv_path}")
        print("  均值: " + _fmt_summary(summary))
        restorer.unload()

    if summaries:
        path = write_summary_csv(summaries, out_root / "summary.csv")
        print(f"\n对比汇总表: {path}")
        print("\n" + _fmt_table(summaries))
    return 0


def _print_progress(done: int, total: int, row) -> None:
    tail = ""
    if row is not None and row.metrics:
        tail = " ".join(f"{k}={format_value(v, 2)}" for k, v in row.metrics.items() if v is not None)
    print(f"\r  [{done}/{total}] {row.filename if row else ''} {tail}".ljust(100), end="")
    if done == total:
        print()


def _fmt_summary(summary) -> str:
    return "  ".join(
        f"{METRIC_LABELS.get(k, k)}={format_value(v, 2 if k in ('psnr', 'fid') else 4)}"
        for k, v in summary.items()
        if v is not None
    )


def _fmt_table(summaries) -> str:
    keys = []
    for s in summaries.values():
        for k in s:
            if k not in keys:
                keys.append(k)
    header = "方法".ljust(14) + "".join(METRIC_LABELS.get(k, k).ljust(12) for k in keys)
    lines = [header, "-" * len(header)]
    for name, s in summaries.items():
        lines.append(name.ljust(14) + "".join(format_value(s.get(k), 3).ljust(12) for k in keys))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
