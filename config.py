"""Configuration for Gluon Ops to TTGIR Mapper Agent."""

import os


# Triton source path
TRITON_PATH = os.environ.get("TRITON_PATH", "/app/triton")

# Files to scan for builtins
BUILTIN_FILES = [
    f"{TRITON_PATH}/python/triton/experimental/gluon/language/_core.py",
    f"{TRITON_PATH}/python/triton/experimental/gluon/language/_math.py",
]
# _standard.py only contains @jit helpers, so skip it when scanning builtins.

# AMD-specific directories and files
AMD_BUILTIN_DIRS = [
    f"{TRITON_PATH}/python/triton/experimental/gluon/language/amd/cdna3",
    f"{TRITON_PATH}/python/triton/experimental/gluon/language/amd/cdna4",
    f"{TRITON_PATH}/python/triton/experimental/gluon/language/amd/gfx1250",
    f"{TRITON_PATH}/python/triton/experimental/gluon/language/amd/rdna3",
    f"{TRITON_PATH}/python/triton/experimental/gluon/language/amd/rdna4",
]

AMD_PRE_SEMANTIC_FILES = {
    os.path.abspath(f"{TRITON_PATH}/python/triton/experimental/gluon/language/amd/_ops.py"),
}

# Semantic and C++ files
SEMANTIC_FILES = [
    f"{TRITON_PATH}/python/triton/experimental/gluon/language/_semantic.py",
    f"{TRITON_PATH}/python/triton/language/semantic.py",
]

CPP_FILES = [
    f"{TRITON_PATH}/python/src/gluon_ir.cc",
    f"{TRITON_PATH}/python/src/ir.cc",
]

# Output paths
OUTPUT_DIR = "/app/TritonToGluon/outputs"

# Agent configuration
MAX_PARALLEL_SUBAGENTS = 10
SUBAGENT_BATCH_SIZE = 5

# Gluon Kernel Example Files
# Tutorials & In-tree Examples
GLUON_TUTORIAL_FILES = [
    f"{TRITON_PATH}/python/examples/gluon/01-attention-forward.py",
    f"{TRITON_PATH}/python/tutorials/gluon/01-intro.py",
    f"{TRITON_PATH}/python/tutorials/gluon/02-layouts.py",
    f"{TRITON_PATH}/python/tutorials/gluon/03-async-copy.py",
    f"{TRITON_PATH}/python/tutorials/gluon/04-tma.py",
    f"{TRITON_PATH}/python/tutorials/gluon/05-wgmma.py",
    f"{TRITON_PATH}/python/tutorials/gluon/06-tcgen05.py",
    f"{TRITON_PATH}/python/tutorials/gluon/07-persistence.py",
    f"{TRITON_PATH}/python/tutorials/gluon/08-warp-specialization.py",
]

# Tests & QA Assets
GLUON_TEST_FILES = [
    f"{TRITON_PATH}/python/test/gluon/test_core.py",
    f"{TRITON_PATH}/python/test/gluon/test_consan.py",
    f"{TRITON_PATH}/python/test/gluon/test_frontend.py",
    f"{TRITON_PATH}/python/test/gluon/test_lowerings.py",
    f"{TRITON_PATH}/python/test/unit/tools/test_triton_to_gluon.py",
    f"{TRITON_PATH}/third_party/amd/python/test/test_gluon_gfx1250.py",
]

# All Gluon kernel example files combined
ALL_GLUON_EXAMPLE_FILES = (
    GLUON_TUTORIAL_FILES +
    GLUON_TEST_FILES
)
