"""
模型注册表：按配置里的 restorer 字段动态构造复原器。

好处：
  - UI 只需要 build_restorer(cfg)，永远不 import 具体模型
  - torch 只在真正要用扩散模型时才被导入，传统方法完全不依赖 torch
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Type

from .base import BaseRestorer

#: 注册表 {name: class}
_RESTORERS: Dict[str, Type[BaseRestorer]] = {}

#: 内置实现所在模块（延迟导入，避免启动就加载 torch）
_LAZY_MODULES: Dict[str, str] = {
    "weatherdiff": "core.weatherdiff.restorer",
    "dcp": "core.baseline.dcp",
    "classical": "core.baseline.classical",
}


def register_restorer(name: str) -> Callable[[Type[BaseRestorer]], Type[BaseRestorer]]:
    """类装饰器：把复原器登记到注册表。

    用法::

        @register_restorer("weatherdiff")
        class WeatherDiffRestorer(BaseRestorer):
            ...
    """

    def deco(cls: Type[BaseRestorer]) -> Type[BaseRestorer]:
        if not issubclass(cls, BaseRestorer):
            raise TypeError(f"{cls} 必须继承 BaseRestorer")
        cls.name = name
        _RESTORERS[name] = cls
        return cls

    return deco


def _ensure_loaded(name: str) -> None:
    """按需导入实现模块，触发其中的 @register_restorer。"""
    if name in _RESTORERS:
        return
    module = _LAZY_MODULES.get(name)
    if module is None:
        raise KeyError(
            f"未知的 restorer: {name!r}。已注册: {sorted(_RESTORERS)}；"
            f"可延迟加载: {sorted(_LAZY_MODULES)}"
        )
    importlib.import_module(module)
    if name not in _RESTORERS:
        raise RuntimeError(f"模块 {module} 导入成功但没有注册名为 {name!r} 的复原器")


def available_restorers() -> List[str]:
    return sorted(set(_RESTORERS) | set(_LAZY_MODULES))


def build_restorer(config: Dict[str, Any]) -> BaseRestorer:
    """
    根据配置构造复原器实例（此时还没有加载权重）。

    :param config: load_config() 返回的字典
    """
    name = str(config.get("restorer") or "").strip()
    if not name:
        raise KeyError(
            "配置里缺少 restorer 字段，无法确定用哪个复原器。"
            f"可用: {sorted(set(_RESTORERS) | set(_LAZY_MODULES))}"
        )
    _ensure_loaded(name)
    cls = _RESTORERS[name]
    instance = cls(config)
    display = config.get("display_name")
    if display:
        instance.display_name = str(display)
    return instance
