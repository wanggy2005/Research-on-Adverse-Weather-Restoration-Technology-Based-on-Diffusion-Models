"""配置加载：读取 configs/*.yaml，并把里面的相对路径解析成绝对路径。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("缺少依赖 pyyaml，请先执行: pip install pyyaml") from exc

#: 项目根目录（core/ 的上一级）
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
WEIGHTS_DIR: Path = PROJECT_ROOT / "weights"
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"

#: 界面下拉框里天气类型的显示顺序
WEATHER_ORDER: List[str] = ["rain", "haze", "snow", "allweather", "dcp"]


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """
    加载一个 yaml 配置。

    额外做两件事:
      1. 把 weights.path 解析成绝对路径，存入 config["weights"]["abs_path"]
      2. 注入 config["_config_path"] 便于排查问题
    """
    p = Path(path)
    if not p.is_absolute():
        # 允许传 "rain" / "rain.yaml" / "configs/rain.yaml"
        candidates = [
            PROJECT_ROOT / p,
            CONFIG_DIR / p,
            CONFIG_DIR / f"{p}.yaml",
            CONFIG_DIR / f"{p}.yml",
        ]
        for c in candidates:
            if c.is_file():
                p = c
                break
    if not p.is_file():
        raise FileNotFoundError(f"找不到配置文件: {path}")

    with open(p, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    cfg["_config_path"] = str(p)
    weights = cfg.get("weights")
    if isinstance(weights, dict) and weights.get("path"):
        wp = Path(weights["path"])
        weights["abs_path"] = str(wp if wp.is_absolute() else PROJECT_ROOT / wp)
    return cfg


def list_weather_configs() -> Dict[str, Path]:
    """
    列出所有可用配置: {配置名: yaml 路径}。
    界面下拉框直接用它填充，新增一种天气只需往 configs/ 丢一个 yaml。
    """
    if not CONFIG_DIR.is_dir():
        return {}
    found = {}
    for f in sorted(CONFIG_DIR.glob("*.y*ml")):
        found[f.stem] = f
    # 按 WEATHER_ORDER 排序，未列出的排在后面
    def sort_key(name: str) -> tuple:
        return (WEATHER_ORDER.index(name) if name in WEATHER_ORDER else 99, name)

    return {k: found[k] for k in sorted(found, key=sort_key)}


def get_config(weather: str) -> Dict[str, Any]:
    """按天气名（rain / haze / snow / allweather / dcp）取配置。"""
    return load_config(weather)


def config_display_name(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("display_name") or cfg.get("name") or "未命名")


def weight_path(cfg: Dict[str, Any]) -> Optional[str]:
    """取配置中权重的绝对路径，没有配置权重（如 DCP）返回 None。"""
    weights = cfg.get("weights")
    if not isinstance(weights, dict):
        return None
    return weights.get("abs_path") or weights.get("path")


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
