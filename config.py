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
