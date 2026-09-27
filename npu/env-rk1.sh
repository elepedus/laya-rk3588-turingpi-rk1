#!/usr/bin/env bash
# Source on an ARM64 graph-build host; all Python caches and temps stay here.
export NPU_ROOT="${NPU_ROOT:-/mnt/warm/laya-service/npu-backend}"
export HOME="$NPU_ROOT/home"
export XDG_CACHE_HOME="$NPU_ROOT/cache/xdg"
export UV_CACHE_DIR="$NPU_ROOT/cache/uv"
export PIP_CACHE_DIR="$NPU_ROOT/cache/pip"
export TMPDIR="$NPU_ROOT/tmp"
export PYTHONPYCACHEPREFIX="$NPU_ROOT/cache/pycache"
export TORCH_HOME="$NPU_ROOT/cache/torch"
export HF_HOME="${HF_HOME:-$NPU_ROOT/cache/huggingface}"
export LD_LIBRARY_PATH="$NPU_ROOT/vendor:${LD_LIBRARY_PATH:-}"
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" \
    "$TMPDIR" "$PYTHONPYCACHEPREFIX" "$TORCH_HOME" "$HF_HOME"
