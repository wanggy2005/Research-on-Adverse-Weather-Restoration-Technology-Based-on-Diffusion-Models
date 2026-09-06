"""
core - 恶劣天气图像复原系统的核心层。

对外只暴露"接口契约"级别的对象，UI 层与评测层只应从这里导入，
不要直接 import 具体模型实现（如 core.weatherdiff.*），
以保证换模型时上层代码零改动。
"""

from .base import (
    BaseRestorer,
    RestoreResult,
    RestoreCancelled,
    WeightNotFoundError,
    ProgressCallback,
    CancelCallback,
)
from .config import (
    PROJECT_ROOT,
    CONFIG_DIR,
    load_config,
    list_weather_configs,
    get_config,
)
from .registry import register_restorer, build_restorer, available_restorers

__all__ = [
    "BaseRestorer",
    "RestoreResult",
    "RestoreCancelled",
    "WeightNotFoundError",
    "ProgressCallback",
    "CancelCallback",
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "load_config",
    "list_weather_configs",
    "get_config",
    "register_restorer",
    "build_restorer",
    "available_restorers",
]
