"""
环境自检 / 全链路冒烟测试（部署后第一件事就跑这个）。

它不需要权重、不需要 GPU，只验证：
  1. 配置能否加载
  2. 传统方法能否跑通 restore（含进度回调）
  3. 取消机制是否生效
  4. DCP 真实算法与传统方法的实际提升
  5. PSNR / SSIM 能否算出来
  6. 批量评测与 CSV 导出能否跑通
  7. 界面依赖是否装好
  8. 扩散模型权重与 GPU 状态（缺失只告警，不算失败）

用法:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.base import RestoreCancelled  # noqa: E402
from core.config import list_weather_configs, load_config, weight_path  # noqa: E402
from core.image_utils import make_demo_image, save_image  # noqa: E402
from core.metrics import available_metrics, psnr, ssim  # noqa: E402
from core.registry import build_restorer  # noqa: E402
from engine.evaluate import evaluate_dataset, write_csv  # noqa: E402

PASS = "[ OK ]"
FAIL = "[FAIL]"


def main() -> int:
    failures = 0

    print("=" * 64)
    print("恶劣天气图像复原系统 - 环境自检")
    print("=" * 64)

    # 1. 配置
    configs = list_weather_configs()
    print(f"\n1) 发现 {len(configs)} 个配置: {', '.join(configs)}")
    if not configs:
        print(f"{FAIL} configs/ 目录里没有 yaml")
        failures += 1

    # 2. 传统方法跑通主流程（含进度回调）
    print("\n2) 传统方法全流程（无需权重）")
    image = make_demo_image((192, 256), seed=1)
    steps = []
    try:
        restorer = build_restorer(load_config("classical"))
        result = restorer.restore_with_stats(
            image, progress_cb=lambda s, t, p: steps.append(s)
        )
        assert result.image.shape == image.shape, "输出尺寸必须与输入一致"
        print(f"{PASS} 复原完成 shape={result.image.shape} 进度回调 {len(steps)} 次 "
              f"耗时 {result.elapsed:.3f}s")
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} 传统方法失败: {exc}")
        failures += 1

    # 3. 取消机制
    print("\n3) 取消机制")
    try:
        restorer = build_restorer(load_config("classical"))
        try:
            restorer.restore_with_stats(image, cancel_cb=lambda: True)
            print(f"{FAIL} 取消回调没有生效")
            failures += 1
        except RestoreCancelled:
            print(f"{PASS} 正确抛出 RestoreCancelled")
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} {exc}")
        failures += 1

    # 4. DCP 真实算法
    print("\n4) DCP 暗通道先验（真实算法，无需权重）")
    try:
        restorer = build_restorer(load_config("dcp"))
        result = restorer.restore_with_stats(image)
        print(f"{PASS} DCP 完成 耗时 {result.elapsed:.3f}s")
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} DCP 失败: {exc}")
        failures += 1

    # 4b. 传统方法 + 合成样本：验证"复原后指标确实变好"
    print("\n4b) 传统方法在合成样本上的实际提升")
    try:
        from core.baseline.classical import ClassicalRestorer  # noqa: PLC0415
        from core.synth import WEATHERS, make_pair  # noqa: PLC0415

        restorer = build_restorer(load_config("classical"))
        for weather in WEATHERS:
            degraded, clean = make_pair(weather, size=(192, 256), seed=0)
            detected = ClassicalRestorer.detect_profile(degraded.astype("float32") / 255.0)
            out = restorer.restore_with_stats(degraded).image
            before, after = psnr(degraded, clean), psnr(out, clean)
            flag = PASS if after > before else "[WARN]"
            print(
                f"{flag} {weather:<5} 判别={detected:<5} "
                f"PSNR {before:5.2f} -> {after:5.2f} dB  "
                f"({'+' if after >= before else ''}{after - before:.2f})"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} 传统方法失败: {exc}")
        failures += 1

    # 5. 指标
    print("\n5) 指标")
    try:
        clean = make_demo_image((192, 256), seed=1)
        print(f"{PASS} PSNR(自己vs自己) = {psnr(clean, clean)}")
        print(f"{PASS} SSIM(自己vs自己) = {ssim(clean, clean):.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} 指标计算失败: {exc}")
        failures += 1
    print("   指标可用性: " + ", ".join(
        f"{k}={'可用' if v else '缺依赖'}" for k, v in available_metrics().items()
    ))

    # 6. 批量评测 + CSV
    print("\n6) 批量评测与 CSV 导出")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_dir = tmp_path / "input"
            gt_dir = tmp_path / "gt"
            for i in range(3):
                save_image(in_dir / f"img{i}.png", make_demo_image((128, 128), seed=i))
                save_image(gt_dir / f"img{i}.png", make_demo_image((128, 128), seed=i + 100))
            cfg = load_config("classical")
            restorer = build_restorer(cfg)
            rows, summary = evaluate_dataset(
                restorer,
                input_dir=in_dir,
                gt_dir=gt_dir,
                output_dir=tmp_path / "out",
                metrics=["psnr", "ssim"],
            )
            csv_path = write_csv(rows, tmp_path / "result.csv")
            assert len(rows) == 3 and Path(csv_path).is_file()
            print(f"{PASS} 评测 {len(rows)} 张，均值 {summary}")
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} 批量评测失败: {exc}")
        failures += 1

    # 7. PyQt5
    print("\n7) 界面依赖")
    try:
        import PyQt5  # noqa: F401

        from ui.compare_view import CompareView  # noqa: F401
        from ui.worker import RestoreWorker  # noqa: F401

        print(f"{PASS} PyQt5 与界面模块导入正常，可执行 python app.py")
    except ImportError as exc:
        print(f"[WARN] 界面依赖缺失({exc})，请执行 pip install PyQt5")

    # 8. 扩散模型权重与 GPU（缺失不算失败，传统方法不依赖它）
    print("\n8) 扩散模型权重与 GPU")
    try:
        ckpt = weight_path(load_config("rain"))
        if ckpt and Path(ckpt).is_file():
            size_mb = Path(ckpt).stat().st_size / 1024 / 1024
            print(f"{PASS} 权重已就绪 {Path(ckpt).name} ({size_mb:.0f} MB)")
        else:
            print("[WARN] 未找到扩散模型权重，请执行 python scripts/download_weights.py")
            print("       （不影响传统方法 classical / dcp 的使用）")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] 权重检查跳过: {exc}")

    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            print(f"{PASS} GPU 可用: {torch.cuda.get_device_name(0)} (torch {torch.__version__})")
        else:
            print(f"[WARN] torch {torch.__version__} 已装但 CUDA 不可用，扩散模型将跑在 CPU（很慢）")
    except ImportError:
        print("[WARN] 未安装 torch，扩散模型不可用（传统方法不受影响）")

    print("\n" + "=" * 64)
    if failures:
        print(f"自检结束：{failures} 项失败，请先解决再使用")
    else:
        print("自检全部通过")
        print("\n下一步:")
        print("  python scripts/make_samples.py    # 生成样本数据")
        print("  python scripts/demo.py            # 命令行演示 + 指标对比表")
        print("  python app.py                     # 打开可视化界面")
    print("=" * 64)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
