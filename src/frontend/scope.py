"""Symbol-table helpers for constexpr reconstruction."""

from __future__ import annotations

import ast
import builtins
import math
import operator
import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Optional

import triton.language.core as tlc
from triton.runtime import jit as triton_jit


class SymbolType(str, Enum):
    """Classification of discovered symbols used downstream by the frontend."""

    VARIABLE = "variable"
    BUILTIN = "builtin"
    JIT_FUNCTION = "jit_function"
    CONSTEXPR = "constexpr"
    DTYPE = "dtype"


@dataclass(slots=True)
class SymbolBinding:
    """Recorded binding along with inferred symbol classification."""

    value: Any
    symbol_type: SymbolType


@dataclass(slots=True)
class SymbolTable:
    """Lightweight mapping of identifiers to their bound objects."""

    bindings: Dict[str, SymbolBinding] = field(default_factory=dict)

    def define(self, name: str, value: Any, *, symbol_type: SymbolType = SymbolType.VARIABLE) -> None:
        """Record a binding within the current scope."""

        self.bindings[name] = SymbolBinding(value=value, symbol_type=symbol_type)

    def lookup(self, name: str) -> Any:
        """Return the object bound to ``name`` if known."""

        binding = self.bindings.get(name)
        return binding.value if binding is not None else None

    def get_binding(self, name: str) -> Optional[SymbolBinding]:
        """Return the full :class:`SymbolBinding` entry if available."""

        return self.bindings.get(name)

    def merge(self, other: "SymbolTable") -> None:
        """Merge bindings from ``other`` into ``self`` (shallow copy)."""

        self.bindings.update(other.bindings)


class ScopeAnalyzer(ast.NodeVisitor):
    """Walks a kernel AST and reconstructs constexpr/global bindings."""

    def __init__(self) -> None:
        self.table = SymbolTable()
        self.globals_map: Dict[str, Any] = {}

    def analyze(self, tree: ast.AST, globals_map: Optional[Dict[str, Any]] = None) -> SymbolTable:
        """Populate :class:`SymbolTable` for ``tree``.

        TODO: Propagate nonlocal scopes, class-level constants, and imported
        modules. For now we ingest the globals map, evaluate simple assignments,
        and classify the symbols to aid later lowering steps.
        """

        self.table = SymbolTable()
        self.globals_map = dict(globals_map or {})
        if self.globals_map:
            for name, value in self.globals_map.items():
                if name == "__builtins__":
                    continue
                self.table.define(name, value, symbol_type=self._classify_symbol(value))
        self.visit(tree)
        return self.table

    # --- ast.NodeVisitor overrides -------------------------------------------------
    def visit_Assign(self, node: ast.Assign) -> Any:  # noqa: ANN401 - NodeVisitor API
        value = self._evaluate_expr(node.value)
        for target in node.targets:
            for identifier in self._collect_names(target):
                self.table.define(identifier, value, symbol_type=self._classify_symbol(value))
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:  # noqa: ANN401 - NodeVisitor API
        value = self._evaluate_expr(node.value) if node.value is not None else None
        for identifier in self._collect_names(node.target):
            self.table.define(identifier, value, symbol_type=self._classify_symbol(value))
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # noqa: ANN401
        func_obj = self._resolve_from_scope(node.name)
        self.table.define(node.name, func_obj, symbol_type=self._classify_symbol(func_obj))
        return self.generic_visit(node)

    # --- helpers -------------------------------------------------------------------
    def _collect_names(self, node: ast.AST) -> Iterable[str]:
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, (ast.Tuple, ast.List)):
            for element in node.elts:
                yield from self._collect_names(element)

    def _resolve_from_scope(self, name: str) -> Any:
        binding = self.table.get_binding(name)
        if binding is not None:
            return binding.value
        if name in self.globals_map:
            return self.globals_map[name]
        return None

    def _resolve_value(self, expr: ast.AST) -> Any:
        if isinstance(expr, ast.Constant):
            return expr.value
        if isinstance(expr, ast.Name):
            binding = self.table.get_binding(expr.id)
            if binding is not None:
                return binding.value
            if expr.id in self.globals_map:
                return self.globals_map[expr.id]
            return sys.modules.get(expr.id)
        if isinstance(expr, ast.Attribute):
            base = self._resolve_value(expr.value)
            if base is None:
                return None
            return getattr(base, expr.attr, None)
        return None

    def _evaluate_expr(self, expr: Optional[ast.AST]) -> Any:
        if expr is None:
            return None
        if isinstance(expr, ast.Constant):
            return expr.value
        if isinstance(expr, ast.IfExp):
            return self._evaluate_ifexp(expr)
        if isinstance(expr, (ast.Tuple, ast.List)):
            values = [self._evaluate_expr(element) for element in expr.elts]
            if any(value is None for value in values):
                return None
            return tuple(values) if isinstance(expr, ast.Tuple) else values
        if isinstance(expr, ast.Dict):
            evaluated_keys = []
            for key in expr.keys:
                if key is None:
                    return None
                evaluated_key = self._evaluate_expr(key)
                if evaluated_key is None:
                    return None
                evaluated_keys.append(evaluated_key)
            evaluated_values = []
            for value in expr.values:
                evaluated_value = self._evaluate_expr(value)
                if evaluated_value is None:
                    return None
                evaluated_values.append(evaluated_value)
            return dict(zip(evaluated_keys, evaluated_values))
        if isinstance(expr, ast.BoolOp):
            return self._evaluate_bool_op(expr)
        if isinstance(expr, ast.Compare):
            return self._evaluate_compare(expr)
        if isinstance(expr, (ast.Name, ast.Attribute)):
            return self._resolve_value(expr)
        if isinstance(expr, ast.UnaryOp):
            operand = self._evaluate_expr(expr.operand)
            return self._apply_unary(expr.op, operand)
        if isinstance(expr, ast.BinOp):
            left = self._evaluate_expr(expr.left)
            right = self._evaluate_expr(expr.right)
            return self._apply_binary(expr.op, left, right)
        if isinstance(expr, ast.Call):
            return self._evaluate_call(expr)
        return None

    def _evaluate_ifexp(self, expr: ast.IfExp) -> Any:
        test_value = self._evaluate_expr(expr.test)
        if test_value is None:
            return None
        branch = expr.body if bool(test_value) else expr.orelse
        return self._evaluate_expr(branch)

    def _evaluate_bool_op(self, node: ast.BoolOp) -> Any:
        if isinstance(node.op, ast.And):
            result = None
            for value_expr in node.values:
                value = self._evaluate_expr(value_expr)
                if value is None:
                    return None
                result = value
                if not value:
                    return value
            return result
        if isinstance(node.op, ast.Or):
            result = None
            for value_expr in node.values:
                value = self._evaluate_expr(value_expr)
                if value is None:
                    return None
                if value:
                    return value
                result = value
            return result
        return None

    def _evaluate_compare(self, node: ast.Compare) -> Optional[bool]:
        left = self._evaluate_expr(node.left)
        if left is None:
            return None
        for op, comparator in zip(node.ops, node.comparators):
            right = self._evaluate_expr(comparator)
            if right is None:
                return None
            outcome = self._apply_comparison(op, left, right)
            if outcome is None or not outcome:
                return outcome
            left = right
        return True

    def _evaluate_call(self, node: ast.Call) -> Any:
        func_obj = self._resolve_value(node.func)
        if func_obj is None:
            return None
        args_kwargs = self._evaluate_call_arguments(node)
        if args_kwargs is None:
            return None
        args, kwargs = args_kwargs
        if func_obj is tlc.constexpr or self._is_builtin_constructor(func_obj) or self._is_math_function(func_obj):
            return self._invoke_callable(func_obj, args, kwargs)
        return None

    def _evaluate_call_arguments(self, node: ast.Call) -> Optional[tuple[list[Any], dict[str, Any]]]:
        args: list[Any] = []
        for arg in node.args:
            value = self._evaluate_expr(arg)
            if value is None:
                return None
            args.append(value)
        kwargs: dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                return None
            kw_value = self._evaluate_expr(kw.value)
            if kw_value is None:
                return None
            kwargs[kw.arg] = kw_value
        return args, kwargs

    def _invoke_callable(self, func_obj: Any, args: list[Any], kwargs: dict[str, Any]) -> Any:
        try:
            return func_obj(*args, **kwargs)
        except Exception:  # noqa: BLE001 - constexpr/builtin evaluation should be side-effect free
            return None

    def _is_builtin_constructor(self, func_obj: Any) -> bool:
        return func_obj in (int, float, bool, str)

    def _is_math_function(self, func_obj: Any) -> bool:
        return callable(func_obj) and getattr(func_obj, "__module__", "") == math.__name__

    def _apply_comparison(self, op: ast.cmpop, left: Any, right: Any) -> Optional[bool]:
        comparisons = {
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
            ast.Is: operator.is_,
            ast.IsNot: operator.is_not,
        }
        func = comparisons.get(type(op))
        if func is not None:
            try:
                return bool(func(left, right))
            except Exception:  # noqa: BLE001 - defensive guard for unsupported comparisons
                return None
        try:
            if isinstance(op, ast.In):
                return operator.contains(right, left)
            if isinstance(op, ast.NotIn):
                return not operator.contains(right, left)
        except Exception:  # noqa: BLE001 - contains may fail for unhashable values
            return None
        return None

    def _apply_unary(self, op: ast.unaryop, operand: Any) -> Any:
        if operand is None:
            return None
        try:
            if isinstance(op, ast.UAdd):
                return +operand
            if isinstance(op, ast.USub):
                return -operand
            if isinstance(op, ast.Not):
                return not operand
            if isinstance(op, ast.Invert):
                return ~operand
        except Exception:  # noqa: BLE001 - defensive
            return None
        return None

    def _apply_binary(self, op: ast.operator, left: Any, right: Any) -> Any:
        if left is None or right is None:
            return None
        operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.MatMult: operator.matmul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.LShift: operator.lshift,
            ast.RShift: operator.rshift,
            ast.BitOr: operator.or_,
            ast.BitXor: operator.xor,
            ast.BitAnd: operator.and_,
        }
        func = operations.get(type(op))
        if func is None:
            return None
        try:
            return func(left, right)
        except Exception:  # noqa: BLE001 - arithmetic fallback
            return None

    def _classify_symbol(self, value: Any) -> SymbolType:
        if isinstance(value, triton_jit.JITCallable):
            return SymbolType.JIT_FUNCTION
        if isinstance(value, tlc.constexpr):
            return SymbolType.CONSTEXPR
        if isinstance(value, tlc.dtype):
            return SymbolType.DTYPE
        if isinstance(
            value,
            (
                types.BuiltinFunctionType,
                types.BuiltinMethodType,
                types.ModuleType,
            ),
        ) or getattr(value, "__module__", None) == builtins.__name__:
            return SymbolType.BUILTIN
        if callable(value) and getattr(value, "__module__", "").startswith("triton.language"):
            return SymbolType.BUILTIN
        return SymbolType.VARIABLE
