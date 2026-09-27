#!/usr/bin/env bash
set -euo pipefail
root="${LAYA_NPU_ROOT:-/mnt/warm/laya-service/npu-backend}"
if [[ -f "$root/runtime.env" ]]; then
    source "$root/runtime.env"
fi
: "${LAYA_MODEL_DIR:?set LAYA_MODEL_DIR to the Laya checkpoint tree}"
export LAYA_MODEL_DIR
export LAYA_RKNNRT="${LAYA_RKNNRT:-$root/vendor/librknnrt.so}"
export LAYA_RKNN_GRAPH_ROOT="${LAYA_RKNN_GRAPH_ROOT:-$root/models}"
export LAYA_NPU_PRELOAD="${LAYA_NPU_PRELOAD:-english,multilingual,typed-decisions}"
export LAYA_NPU_FULL_BIND="${LAYA_NPU_FULL_BIND:-0.0.0.0:8003}"
export LAYA_NPU_TIMING="${LAYA_NPU_TIMING:-1}"
export LAYA_NPU_CACHE_BUCKETS="${LAYA_NPU_CACHE_BUCKETS:-1}"
export LAYA_NPU_FP16_CHAIN="${LAYA_NPU_FP16_CHAIN:-1}"
export LAYA_NPU_FUSED_BLOCKS="${LAYA_NPU_FUSED_BLOCKS:-1}"
export LAYA_NPU_FUSED_GROUP="${LAYA_NPU_FUSED_GROUP:-14}"
export LAYA_NPU_WINDOW_QUERY="${LAYA_NPU_WINDOW_QUERY:-256}"
export LAYA_NPU_KEEP_CACHES="${LAYA_NPU_KEEP_CACHES:-1}"
binary="${LAYA_NPU_BINARY:-$root/bin/laya-rknpu}"
test -x "$binary"
mkdir -p "$root/logs"
cd "$root"
exec "$binary" serve-full >> "$root/logs/full-server.log" 2>&1
