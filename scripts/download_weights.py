"""
预训练权重下载脚本（模型组第一天执行）。

WeatherDiffusion 官方权重是 TU Graz 的 HTTP 直链，不需要 Google Drive。

用法:
    python scripts/download_weights.py                # 下载默认的 WeatherDiff64
    python scripts/download_weights.py --all          # 两个权重都下
    python scripts/download_weights.py --list         # 只看有哪些权重（含完整性状态）
    python scripts/download_weights.py --verify       # 只校验本地文件，不下载
    python scripts/download_weights.py --force        # 删掉本地文件重新下载
    python scripts/download_weights.py --url <URL>    # 下载自定义地址(备选方案)

特性:
  - 断点续传：中断后重跑会从已下载的字节继续（包括上次没下完的残缺文件）
  - 完整性校验：下载后检查字节数 + zip 结构，避免出现
    "PytorchStreamReader failed reading zip archive" 这类损坏错误
  - SSL 兜底：部分网络环境证书链不全，会自动降级并给出提示
"""

from __future__ import annotations

import argparse
import os
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import WEIGHTS_DIR, ensure_dir  # noqa: E402

WEIGHTS = {
    "WeatherDiff64": {
        "url": "https://igi-web.tugraz.at/download/OzdenizciLegensteinTPAMI2023/WeatherDiff64.pth.tar",
        "filename": "WeatherDiff64.pth.tar",
        "size": 1327834207,  # 官方文件精确字节数，用于判断是否下完
        "note": "patch=64，配置 configs/*.yaml 默认用它，8~12G 显存推荐",
    },
    "WeatherDiff128": {
        "url": "https://igi-web.tugraz.at/download/OzdenizciLegensteinTPAMI2023/WeatherDiff128.pth.tar",
        "filename": "WeatherDiff128.pth.tar",
        "size": None,  # 未记录，靠服务器 Content-Length 判断
        "note": "patch=128，质量略好但更慢更吃显存，需要把 data.image_size 改成 128",
    },
}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


# --------------------------------------------------------------------------- #
# SSL：优先正常校验，证书链不全时降级（打印告警）
# --------------------------------------------------------------------------- #
def _ssl_context(insecure: bool = False) -> Optional[ssl.SSLContext]:
    if not insecure:
        try:
            import certifi  # noqa: PLC0415

            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            return None  # 用 Python 默认
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _urlopen(req: urllib.request.Request, timeout: int = 60):
    """先按正常 SSL 打开，证书失败则降级重试一次。"""
    try:
        return urllib.request.urlopen(req, timeout=timeout, context=_ssl_context(False))
    except urllib.error.URLError as exc:
        if not isinstance(getattr(exc, "reason", None), ssl.SSLCertVerificationError):
            raise
        print("  [警告] SSL 证书校验失败，降级为不校验证书继续下载")
        print("         如需彻底修复: pip install -U certifi")
        return urllib.request.urlopen(req, timeout=timeout, context=_ssl_context(True))


def remote_size(url: str) -> Optional[int]:
    """查询服务器上文件的字节数，失败返回 None。"""
    try:
        with _urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as resp:
            value = resp.headers.get("Content-Length")
            return int(value) if value else None
    except Exception:  # noqa: BLE001 查不到就算了，不影响下载
        return None


# --------------------------------------------------------------------------- #
# 完整性校验
# --------------------------------------------------------------------------- #
def verify_checkpoint(path: Path, expect_size: Optional[int] = None) -> Tuple[bool, str]:
    """
    检查权重文件是否完整可读。

    :return: (是否通过, 说明文字)
    """
    if not path.is_file():
        return False, "文件不存在"

    size = path.stat().st_size
    if size == 0:
        return False, "文件为空"

    if expect_size and size != expect_size:
        pct = size / expect_size * 100
        return False, (
            f"字节数不符: 本地 {human(size)} / 应为 {human(expect_size)}"
            f"（仅完成 {pct:.1f}%，需要续传）"
        )

    # PyTorch 新格式权重本质是 zip，先做结构校验（比 torch.load 快得多）
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic[:2] == b"PK":
        if not zipfile.is_zipfile(path):
            return False, "zip 结构损坏（找不到中央目录，通常是没下完或传输中断）"
        try:
            with zipfile.ZipFile(path) as zf:
                bad = zf.testzip()
            if bad:
                return False, f"zip 内部数据损坏: {bad}"
        except zipfile.BadZipFile as exc:
            return False, f"zip 解析失败: {exc}"

    return True, f"完整 ({human(size)})"


def verify_loadable(path: Path) -> Tuple[bool, str]:
    """进一步用 torch 真正加载一次（慢，但最可靠）。"""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return True, "跳过 torch 加载测试（未安装 torch）"

    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return False, f"torch.load 失败: {type(exc).__name__}: {exc}"

    if isinstance(ckpt, dict):
        keys = list(ckpt.keys())[:6]
        return True, f"torch.load 成功，顶层键: {keys}"
    return True, f"torch.load 成功，类型 {type(ckpt).__name__}"


# --------------------------------------------------------------------------- #
# 下载
# --------------------------------------------------------------------------- #
def download(url: str, dest: Path, expect_size: Optional[int] = None, chunk: int = 1 << 20) -> None:
    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")

    # 上次下到一半但被改成最终名的残缺文件 -> 挪回 .part 继续续传
    if dest.exists() and not tmp.exists():
        ok, _ = verify_checkpoint(dest, expect_size)
        if not ok:
            print(f"  发现残缺的 {dest.name}，转为续传模式")
            dest.replace(tmp)

    done = tmp.stat().st_size if tmp.exists() else 0

    req = urllib.request.Request(url)
    if done:
        req.add_header("Range", f"bytes={done}-")
        print(f"  从 {human(done)} 处续传")

    try:
        with _urlopen(req, timeout=60) as resp:
            if done and resp.status != 206:
                # 服务器不支持续传，只能从头下
                print("  服务器不支持断点续传，将从头开始下载")
                done = 0
                tmp.unlink(missing_ok=True)
            total = int(resp.headers.get("Content-Length", 0)) + done
            mode = "ab" if done else "wb"
            with open(tmp, mode) as f:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    pct = f"{done / total * 100:5.1f}%" if total else "  ?  "
                    print(f"\r  {pct}  {human(done)} / {human(total) if total else '?'}", end="")
        print()
    except Exception as exc:  # noqa: BLE001
        print(f"\n  下载失败: {exc}")
        print(f"  已保留进度 {human(done)}，重跑本脚本会自动续传")
        print("  备选办法: 用浏览器/迅雷直接打开下面的地址下载，然后放到 weights/ 目录")
        print(f"  {url}")
        raise

    # 落盘前先校验，避免把损坏文件写成最终名
    ok, msg = verify_checkpoint(tmp, expect_size)
    if not ok:
        print(f"  [失败] 下载的文件校验不通过: {msg}")
        print(f"  已保留 {tmp.name}，重跑本脚本会继续续传")
        raise RuntimeError(f"权重文件校验失败: {msg}")

    tmp.replace(dest)
    print(f"  完成 -> {dest}  ({human(dest.stat().st_size)})  校验通过")


# --------------------------------------------------------------------------- #
def _do_verify(targets) -> int:
    failed = 0
    for name in targets:
        info = WEIGHTS[name]
        path = WEIGHTS_DIR / info["filename"]
        expect = info.get("size") or remote_size(info["url"])
        ok, msg = verify_checkpoint(path, expect)
        print(f"  [{'OK ' if ok else 'BAD'}] {name}: {msg}")
        if ok:
            ok2, msg2 = verify_loadable(path)
            print(f"         {msg2}")
            ok = ok2
        if not ok:
            failed += 1
            print(f"         修复: python scripts/download_weights.py   # 会自动续传")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 WeatherDiffusion 预训练权重")
    parser.add_argument("--all", action="store_true", help="下载全部权重")
    parser.add_argument("--list", action="store_true", help="列出可下载的权重")
    parser.add_argument("--verify", action="store_true", help="只校验本地权重完整性")
    parser.add_argument("--force", action="store_true", help="删除本地文件重新下载")
    parser.add_argument("--url", help="下载自定义 URL（备选模型权重）")
    parser.add_argument("--out", help="保存文件名（配合 --url）")
    args = parser.parse_args()

    if args.list:
        print("可下载的权重:")
        for name, info in WEIGHTS.items():
            path = WEIGHTS_DIR / info["filename"]
            if path.exists():
                ok, msg = verify_checkpoint(path, info.get("size"))
                state = "已就绪" if ok else "损坏/未下完"
                extra = f"  <- {msg}"
            else:
                state, extra = "未下载", ""
            print(f"  [{state}] {name:<16} {info['note']}{extra}")
            print(f"           {info['url']}")
        return 0

    targets = list(WEIGHTS) if args.all else ["WeatherDiff64"]

    if args.verify:
        print("校验本地权重:")
        return _do_verify(targets)

    if args.url:
        filename = args.out or args.url.split("/")[-1]
        print(f"下载 {filename}")
        download(args.url, WEIGHTS_DIR / filename)
        return 0

    for name in targets:
        info = WEIGHTS[name]
        dest = WEIGHTS_DIR / info["filename"]
        expect = info.get("size") or remote_size(info["url"])

        if args.force and dest.exists():
            print(f"{name} --force: 删除旧文件 {dest}")
            dest.unlink()

        if dest.exists():
            ok, msg = verify_checkpoint(dest, expect)
            if ok:
                print(f"{name} 已存在且完整，跳过: {dest}  ({msg})")
                continue
            print(f"{name} 本地文件不可用: {msg}")

        print(f"下载 {name}  ({info['note']})")
        if expect:
            print(f"  目标大小 {human(expect)}")
        download(info["url"], dest, expect)

    print("\n完成。可用 python scripts/check_weights.py 验证权重能否被模型加载。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
