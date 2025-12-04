"""Shared fixtures and Triton/Gluon stubs for codegen tests."""

from __future__ import annotations

import ast
import sys
import textwrap
from dataclasses import field, make_dataclass
from types import ModuleType
from typing import Any, Callable, Dict, Mapping, MutableMapping, Sequence

import pytest

from ..test_frontend import _install_triton_stubs


def _install_gluon_stubs() -> None:
    """Provide lightweight ``triton.experimental.gluon`` modules for tests."""

    if "triton.experimental" in sys.modules:
        return

    triton_mod = sys.modules.get("triton")
    if triton_mod is None:
        raise RuntimeError("Triton stubs must be installed before Gluon stubs")
    if not hasattr(triton_mod, "__path__"):
        triton_mod.__path__ = []  # type: ignore[attr-defined]

    experimental_mod = ModuleType("triton.experimental")
    experimental_mod.__path__ = []  # type: ignore[attr-defined]

    gluon_mod = ModuleType("triton.experimental.gluon")
    gluon_mod.__path__ = []  # type: ignore[attr-defined]

    def jit(func: Callable[..., Any] | None = None, **_: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]] | Callable[..., Any]:
        def decorator(target: Callable[..., Any]) -> Callable[..., Any]:
            return target

        if func is None:
            return decorator
        return decorator(func)

    gluon_mod.jit = jit

    def _passthrough(name: str) -> Callable[..., Any]:
        def _impl(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...], Dict[str, Any]]:
            return (name, args, kwargs)

        return _impl

    gluon_mod.arange = _passthrough("arange")
    gluon_mod.program_id = _passthrough("program_id")

    def num_warps() -> int:
        return 4

    gluon_mod.num_warps = num_warps

    def convert_layout(tensor: Any, layout: Any) -> tuple[str, Any, Any]:
        return ("convert_layout", tensor, layout)

    gluon_mod.convert_layout = convert_layout

    language_mod = ModuleType("triton.experimental.gluon.language")
    language_mod.__path__ = []  # type: ignore[attr-defined]

    layouts_mod = ModuleType("triton.experimental.gluon.language._layouts")

    layout_fields: dict[str, tuple[str, ...]] = {
        "AutoLayout": ("value",),
        "BlockedLayout": ("shape", "order"),
        "CoalescedLayout": ("axes",),
        "DistributedLayout": ("axes", "blocks"),
        "DistributedLinearLayout": ("axes",),
        "DotOperandLayout": ("operands",),
        "NVMMADistributedLayout": ("axes",),
        "NVMMASharedLayout": ("axes",),
        "PaddedSharedLayout": ("axes",),
        "SharedLayout": ("axes",),
        "SharedLinearLayout": ("axes",),
        "SliceLayout": ("dim", "parent"),
        "SwizzledSharedLayout": ("axes",),
    }

    layout_classes: dict[str, type] = {}
    for name, fields_spec in layout_fields.items():
        dataclass_fields = [(field_name, Any, field(default=None)) for field_name in fields_spec]
        layout_classes[name] = make_dataclass(name, dataclass_fields, slots=True)

    for name, cls in layout_classes.items():
        setattr(layouts_mod, name, cls)
        setattr(gluon_mod, name, cls)

    language_mod._layouts = layouts_mod  # type: ignore[attr-defined]

    tools_pkg = ModuleType("triton.tools")
    tools_pkg.__path__ = []  # type: ignore[attr-defined]
    helpers_pkg = ModuleType("triton.tools.triton_to_gluon_translater")
    helpers_pkg.__path__ = []  # type: ignore[attr-defined]
    helpers_mod = ModuleType("triton.tools.triton_to_gluon_translater.translator_helpers")

    def reset_to_default_layout(call: Any) -> tuple[str, Any]:
        return ("reset_to_default_layout", call)

    def set_split_src_layout(value: Any) -> tuple[str, Any]:
        return ("set_split_src_layout", value)

    def default_blocked_layout(shape: Any, warps: Any) -> tuple[str, Any, Any]:
        return ("default_blocked_layout", shape, warps)

    def tl_dot(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...], Dict[str, Any]]:
        return ("tl_dot", args, kwargs)

    def tl_dot_decomposed_scale_to_16(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...], Dict[str, Any]]:
        return ("tl_dot_scaled", args, kwargs)

    helpers_mod.reset_to_default_layout = reset_to_default_layout
    helpers_mod.set_split_src_layout = set_split_src_layout
    helpers_mod.default_blocked_layout = default_blocked_layout
    helpers_mod.tl_dot = tl_dot
    helpers_mod.tl_dot_decomposed_scale_to_16 = tl_dot_decomposed_scale_to_16

    triton_mod.experimental = experimental_mod
    triton_mod.tools = tools_pkg
    experimental_mod.gluon = gluon_mod
    gluon_mod.language = language_mod
    tools_pkg.triton_to_gluon_translater = helpers_pkg
    helpers_pkg.translator_helpers = helpers_mod

    sys.modules["triton.experimental"] = experimental_mod
    sys.modules["triton.experimental.gluon"] = gluon_mod
    sys.modules["triton.experimental.gluon.language"] = language_mod
    sys.modules["triton.experimental.gluon.language._layouts"] = layouts_mod
    sys.modules["triton.tools"] = tools_pkg
    sys.modules["triton.tools.triton_to_gluon_translater"] = helpers_pkg
    sys.modules["triton.tools.triton_to_gluon_translater.translator_helpers"] = helpers_mod


def _ensure_tensor_stubs() -> None:
    """Augment the Triton stubs with tensor sentinels used by the transformer."""

    tl_module = sys.modules["triton.language"]
    core_module = sys.modules["triton.language.core"]
    if not hasattr(core_module, "tensor"):
        class _TensorType:  # pragma: no cover - simple sentinel
            pass

        core_module.tensor = _TensorType
        tl_module.tensor = _TensorType
    if not hasattr(tl_module, "float32"):
        tl_module.float32 = core_module.dtype("float32")
    if not hasattr(tl_module, "tensor_descriptor"):
        tl_module.tensor_descriptor = object()


_install_triton_stubs()
_install_gluon_stubs()
_ensure_tensor_stubs()

from src.codegen.generator import CodeGenerator, _call_graph_key
from src.frontend.parser import CallDependency, ConstexprMetadata, ParsedKernel
from src.mapping.function_registry import MappingFunctionRegistry, registry as builtin_registry
from src.ttgir import NodeLocation, NodeMetadata, TTGIROutput
from triton.runtime import jit as triton_jit


class FakeKernelParser:
    """Deterministic parser stub that returns pre-registered kernels."""

    def __init__(self) -> None:
        self._kernels: Dict[object, ParsedKernel] = {}

    def register(self, parsed_kernel: ParsedKernel) -> ParsedKernel:
        self._kernels[parsed_kernel.kernel] = parsed_kernel
        return parsed_kernel

    def add_mapping(self, kernel_obj: object, parsed_kernel: ParsedKernel) -> None:
        self._kernels[kernel_obj] = parsed_kernel

    def parse_kernel(self, kernel_obj: object) -> ParsedKernel:
        if kernel_obj not in self._kernels:
            raise KeyError(f"Kernel {kernel_obj!r} was not registered in FakeKernelParser")
        return self._kernels[kernel_obj]


@pytest.fixture
def fake_kernel_parser() -> FakeKernelParser:
    """Provide a fake parser so tests can inject custom kernel ASTs."""

    return FakeKernelParser()


@pytest.fixture
def mapping_registry_factory() -> Callable[[], MappingFunctionRegistry]:
    """Return a factory that yields isolated mapping registries."""

    def _factory() -> MappingFunctionRegistry:
        builtin_registry.load_builtin_functions()
        return builtin_registry

    return _factory


@pytest.fixture
def code_generator(fake_kernel_parser: FakeKernelParser, mapping_registry_factory: Callable[[], MappingFunctionRegistry]) -> CodeGenerator:
    """Default ``CodeGenerator`` that leverages the fake parser."""

    return CodeGenerator(registry=mapping_registry_factory(), kernel_parser=fake_kernel_parser)


@pytest.fixture
def code_generator_factory(
    fake_kernel_parser: FakeKernelParser,
    mapping_registry_factory: Callable[[], MappingFunctionRegistry],
) -> Callable[[], CodeGenerator]:
    """Allow tests to instantiate multiple ``CodeGenerator`` instances on demand."""

    def _factory() -> CodeGenerator:
        return CodeGenerator(registry=mapping_registry_factory(), kernel_parser=fake_kernel_parser)

    return _factory


@pytest.fixture
def parsed_kernel_factory() -> Callable[..., ParsedKernel]:
    """Build ``ParsedKernel`` objects from source snippets for tests."""

    def _factory(
        source: str,
        *,
        func_name: str = "root_kernel",
        module_name: str = "tests.codegen",
        is_jit: bool = False,
        call_graph: Mapping[str, Sequence[CallDependency]] | None = None,
        globals_extra: MutableMapping[str, Any] | None = None,
        constexpr_metadata: Sequence[ConstexprMetadata] | None = None,
    ) -> ParsedKernel:
        dedented = textwrap.dedent(source)
        tree = ast.parse(dedented)
        ast.fix_missing_locations(tree)
        env: dict[str, Any] = {"__name__": module_name, "tl": sys.modules["triton.language"]}
        if globals_extra:
            env.update(globals_extra)
        exec(dedented, env)
        fn_obj = env[func_name]
        kernel_obj = triton_jit.jit(fn_obj) if is_jit else fn_obj
        kernel_obj.__module__ = module_name
        key = _call_graph_key(kernel_obj)
        graph = dict(call_graph or {key: []})
        metadata = tuple(constexpr_metadata or ())
        return ParsedKernel(
            kernel=kernel_obj,
            tree=tree,
            source=dedented,
            filename=f"/{module_name.replace('.', '/')}/{func_name}.py",
            globals_map=env,
            call_graph=graph,
            constexpr_metadata=metadata,
        )

    return _factory


@pytest.fixture
def ttgir_output_factory() -> Callable[..., TTGIROutput]:
    """Factory that builds ``TTGIROutput`` containers for tests."""

    def _factory(
        *,
        metadata: Mapping[NodeLocation, Any] | None = None,
    ) -> TTGIROutput:
        normalized: dict[NodeLocation, NodeMetadata] = {}
        for location, entry in (metadata or {}).items():
            if isinstance(entry, NodeMetadata):
                normalized[location] = entry
            elif isinstance(entry, Mapping):
                normalized[location] = NodeMetadata(layouts=dict(entry))
            else:
                normalized[location] = NodeMetadata(layouts={"layout": entry})
        return TTGIROutput(metadata=normalized)

    return _factory


@pytest.fixture
def call_dependency_factory() -> Callable[[object], CallDependency]:
    """Helper that constructs ``CallDependency`` entries for kernels."""

    def _factory(kernel_obj: object) -> CallDependency:
        name = getattr(kernel_obj, "__name__", getattr(kernel_obj, "__qualname__", "kernel"))
        qualified_name = _call_graph_key(kernel_obj)
        module_name = getattr(kernel_obj, "__module__", None)
        return CallDependency(name=name, qualified_name=qualified_name, module=module_name, obj=kernel_obj)

    return _factory


@pytest.fixture
def layouts_module() -> ModuleType:
    """Expose the stubbed layout module used for metadata serialization."""

    return sys.modules["triton.experimental.gluon.language._layouts"]
