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
group="${3:-7}"
window_query="${4:-0}"
case "$name" in
    english) checkpoint="$snapshot"; alias=english; layers=28; max_len=512 ;;
    multilingual) checkpoint="$snapshot/multilingual"; alias=multilingual; layers=22; max_len=1024 ;;
    typed-decisions) checkpoint="$snapshot/typed-decisions"; alias=typed; layers=28; max_len=1024 ;;
    *) echo 'usage: build_encoder_blocks.sh CHECKPOINT LENGTH [GROUP_SIZE] [WINDOW_QUERY]' >&2; exit 2 ;;
esac
if (( length < 64 || length > max_len || group < 2 \
      || (window_query != 0 && window_query < 64) )); then
    echo "invalid length, group, or query block: $length $group $window_query" >&2
    exit 2
fi
oracle="$NPU_ROOT/models/oracle-$alias-$length"
test -s "$oracle/embedding.f32"
suffix=""
extra=()
if (( window_query != 0 )); then
    suffix="-window$window_query"
    extra=(--window-query "$window_query")
fi
destination="$NPU_ROOT/models/$alias$length/blocks$group$suffix"
logs="$NPU_ROOT/logs/fused-$alias-$length$suffix"
mkdir -p "$destination" "$logs"
start=0
while (( start < layers )); do
    remaining=$((layers - start))
    if (( remaining <= group + 1 )); then
        count=$remaining
    else
        count=$group
    fi
    end=$((start + count - 1))
    printf -v first '%02d' "$start"
    printf -v last '%02d' "$end"
    path="$destination/encoder_${first}_${last}.rknn"
    if [[ -s "$path" ]]; then
        echo "ready $alias $length block $first-$last (existing)"
        start=$((end + 1))
        continue
    fi
    work="$destination/block_${first}_${last}"
    mkdir -p "$work"
    echo "export $alias $length block $first-$last $(date -u +%FT%TZ)"
    USE_TF=0 "$cpu" \
        "$NPU_ROOT/scripts/export_encoder_block.py" \
        "$checkpoint" "$oracle" "$work" \
        --start "$start" --count "$count" --length "$length" "${extra[@]}" \
        > "$logs/export_${first}_${last}.log" 2>&1
    echo "convert $alias $length block $first-$last $(date -u +%FT%TZ)"
    "$compiler" "$NPU_ROOT/scripts/convert_onnx.py" \
        "$work/encoder_${first}_${last}.onnx" "$path" \
        > "$logs/convert_${first}_${last}.log" 2>&1
    rm -f "$work/encoder_${first}_${last}.onnx" "$work/hidden.f32" \
        "$work/mask_full.f32" "$work/mask_local.f32" "$work/expected.f32"
    echo "ready $alias $length block $first-$last $(date -u +%FT%TZ)"
    start=$((end + 1))
done
