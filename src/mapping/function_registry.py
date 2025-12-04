"""Runtime registry for Triton→Gluon mapping helpers."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Iterable, Optional

from .metadata_schema import OpMetadataSpec


class MappingFunctionRegistry:
    """Registry that maps Triton op names to Gluon helper callables."""

    def __init__(self) -> None:
        self._functions: Dict[str, Callable[..., Any]] = {}
        self._schemas: Dict[str, OpMetadataSpec] = {}
        self._builtins_loaded = False

    def register(
        self,
        triton_op: str,
        *,
        schema: OpMetadataSpec | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator used by mapping modules to register helper functions."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._functions[triton_op] = func
            if schema is not None:
                self._schemas[triton_op] = schema
            return func

        return decorator

    def lookup(self, triton_op: str) -> Optional[Callable[..., Any]]:
        """Return the helper function registered for ``triton_op``."""

        return self._functions.get(triton_op)

    def get_schema(self, triton_op: str) -> OpMetadataSpec | None:
        """Return the metadata schema registered for ``triton_op``."""

        return self._schemas.get(triton_op)

    def get_import_path(self, triton_op: str) -> Optional[str]:
        """Return an import statement for the helper registered to ``triton_op``."""

        fn = self._functions.get(triton_op)
        if fn is None:
            return None
        module = getattr(fn, "__module__", None)
        name = getattr(fn, "__name__", None)
        if not module or not name:
            return None
        return f"from {module} import {name}"

    def available_ops(self) -> Iterable[str]:
        """Yield the names of registered Triton ops."""

        return tuple(sorted(self._functions))

    def load_builtin_functions(self) -> None:
        """Import ``mapping.functions`` once to register default helpers."""

        if self._builtins_loaded:
            return
        package = __name__.rsplit(".", 1)[0]
        importlib.import_module(f"{package}.functions")
        self._builtins_loaded = True


registry = MappingFunctionRegistry()

__all__ = ["MappingFunctionRegistry", "registry"]
