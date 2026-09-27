#!/usr/bin/env bash
set -euo pipefail
source "${LAYA_NPU_ENV:-/mnt/warm/laya-service/npu-backend/env.sh}"
mkdir -p "$NPU_ROOT/tmp" "$NPU_ROOT/logs"
cd "$NPU_ROOT/tmp"
snapshot="${LAYA_MODEL_SNAPSHOT:?set LAYA_MODEL_SNAPSHOT to the pinned Laya checkpoint tree}"
cpu="${LAYA_CPU_PYTHON:-/mnt/warm/laya-service/venv/bin/python}"
compiler="${LAYA_RKNN_PYTHON:-$NPU_ROOT/venv/bin/python}"
case "${1:-}" in
    english)
        checkpoint="$snapshot"
        alias=english
        lengths=(64 128 256 512)
        ;;
    multilingual)
        checkpoint="$snapshot/multilingual"
        alias=multilingual
        lengths=(64 128 256 512 1024)
        ;;
    typed-decisions)
        checkpoint="$snapshot/typed-decisions"
        alias=typed
        lengths=(64 128 256 512 1024)
        ;;
    *) echo 'usage: build_onehot_heads.sh english|multilingual|typed-decisions' >&2; exit 2 ;;
esac

for length in "${lengths[@]}"; do
    graphs="$NPU_ROOT/models/$alias$length"
    oracle="$NPU_ROOT/models/oracle-$alias-$length"
    test -s "$oracle/embedding.f32"
    test -s "$graphs/embedding_norm.rknn"
    if [[ -s "$graphs/head_onehot.rknn" ]]; then
        echo "ready $alias $length one-hot head (existing)"
        continue
    fi
    echo "export $alias $length one-hot head $(date -u +%FT%TZ)"
    USE_TF=0 "$cpu" \
        "$NPU_ROOT/scripts/export_head.py" "$checkpoint" "$oracle" "$graphs" \
        --length "$length" --onehot \
        > "$NPU_ROOT/logs/onehot-$alias-$length-export.log" 2>&1
    echo "convert $alias $length one-hot head $(date -u +%FT%TZ)"
    "$compiler" "$NPU_ROOT/scripts/convert_onnx.py" \
        "$graphs/head_onehot.onnx" "$graphs/head_onehot.rknn" \
        > "$NPU_ROOT/logs/onehot-$alias-$length-convert.log" 2>&1
    rm -f "$graphs/head_onehot.onnx"
    echo "ready $alias $length one-hot head $(date -u +%FT%TZ)"
done
