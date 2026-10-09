"""Fixture simulating assistant's tools/registry.py for testing tool registration."""

import inspect
from typing import Any, Callable, Dict

_REGISTERED_TOOLS: Dict[str, Dict[str, Any]] = {}


def tool(func: Callable) -> Callable:
    """Decorator registering function as assistant tool."""
    sig = inspect.signature(func)
    params = {}
    for p_name, p_param in sig.parameters.items():
        params[p_name] = {
            "type": "string" if p_param.annotation is str else "integer",
            "default": p_param.default if p_param.default != inspect.Parameter.empty else None,
        }

    _REGISTERED_TOOLS[func.__name__] = {
        "function": func,
        "name": func.__name__,
        "docstring": func.__doc__,
        "parameters": params,
    }
    return func


def get_registered_tools() -> Dict[str, Dict[str, Any]]:
    """Return dict of registered tools."""
    return _REGISTERED_TOOLS
