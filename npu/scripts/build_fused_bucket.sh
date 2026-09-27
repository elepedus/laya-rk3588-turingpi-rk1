#!/usr/bin/env bash
set -euo pipefail
source "${LAYA_NPU_ENV:-/mnt/warm/laya-service/npu-backend/env.sh}"
mkdir -p "$NPU_ROOT/tmp"
cd "$NPU_ROOT/tmp"
snapshot="${LAYA_MODEL_SNAPSHOT:?set LAYA_MODEL_SNAPSHOT to the pinned Laya checkpoint tree}"
cpu="${LAYA_CPU_PYTHON:-/mnt/warm/laya-service/venv/bin/python}"
compiler="${LAYA_RKNN_PYTHON:-$NPU_ROOT/venv/bin/python}"
name="${1:-}"
length="${2:-}"
case "$name" in
    multilingual) checkpoint="$snapshot/multilingual"; alias=multilingual ;;
    typed-decisions) checkpoint="$snapshot/typed-decisions"; alias=typed ;;
    *) echo 'usage: build_fused_bucket.sh multilingual|typed-decisions LENGTH' >&2; exit 2 ;;
esac
if (( length <= 512 || length >= 1024 )); then
    echo "intermediate bucket length must be between 512 and 1024: $length" >&2
    exit 2
fi
oracle="$NPU_ROOT/models/oracle-$alias-$length"
graphs="$NPU_ROOT/models/$alias$length"
logs="$NPU_ROOT/logs/fused-$alias-$length"
mkdir -p "$oracle" "$graphs" "$logs"
if [[ ! -s "$oracle/embedding.f32" ]]; then
    USE_TF=0 "$cpu" \
        "$NPU_ROOT/scripts/oracle_length.py" "$checkpoint" "$oracle" \
        --tokens "$length" > "$logs/oracle.log" 2>&1
fi
for part in embedding_norm head_onehot; do
    if [[ -s "$graphs/$part.rknn" ]]; then
        echo "ready $alias $length $part (existing)"
        continue
    fi
    if [[ "$part" == embedding_norm ]]; then
        exporter=export_embedding_norm.py
        extra=()
    else
        exporter=export_head.py
        extra=(--onehot)
    fi
    echo "export $alias $length $part $(date -u +%FT%TZ)"
    USE_TF=0 "$cpu" \
        "$NPU_ROOT/scripts/$exporter" "$checkpoint" "$oracle" "$graphs" \
        --length "$length" "${extra[@]}" > "$logs/$part-export.log" 2>&1
    echo "convert $alias $length $part $(date -u +%FT%TZ)"
    "$compiler" "$NPU_ROOT/scripts/convert_onnx.py" \
        "$graphs/$part.onnx" "$graphs/$part.rknn" \
        > "$logs/$part-convert.log" 2>&1
    rm -f "$graphs/$part.onnx"
    echo "ready $alias $length $part $(date -u +%FT%TZ)"
done
"$NPU_ROOT/scripts/build_encoder_blocks.sh" "$name" "$length" 14
