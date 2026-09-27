#!/usr/bin/env bash
set -euo pipefail
source "${LAYA_NPU_ENV:-/mnt/warm/laya-service/npu-backend/env.sh}"
mkdir -p "$NPU_ROOT/tmp"
cd "$NPU_ROOT/tmp"
snapshot="${LAYA_MODEL_SNAPSHOT:?set LAYA_MODEL_SNAPSHOT to the pinned Laya checkpoint tree}"

case "${1:-}" in
    english)
        checkpoint="$snapshot"
        alias=english
        count=28
        lengths=(64 128 256 512)
        ;;
    multilingual)
        checkpoint="$snapshot/multilingual"
        alias=multilingual
        count=22
        lengths=(64 128 256 512 1024)
        ;;
    typed-decisions)
        checkpoint="$snapshot/typed-decisions"
        alias=typed
        count=28
        lengths=(64 128 256 512 1024)
        ;;
    *)
        echo 'usage: build_full_npu.sh english|multilingual|typed-decisions' >&2
        exit 2
        ;;
esac

cpu="${LAYA_CPU_PYTHON:-/mnt/warm/laya-service/venv/bin/python}"
compiler="${LAYA_RKNN_PYTHON:-$NPU_ROOT/venv/bin/python}"
scripts="$NPU_ROOT/scripts"
decision="$NPU_ROOT/models/$alias-decision-ops"
mkdir -p "$decision" "$NPU_ROOT/logs/full-$alias"
oracle64="$NPU_ROOT/models/oracle-$alias-64"
if [[ ! -s "$oracle64/encoder_$(printf '%02d' "$((count-1))").f32" ]]; then
    mkdir -p "$NPU_ROOT/logs/full-$alias/64"
    USE_TF=0 "$cpu" "$scripts/oracle_length.py" \
        "$checkpoint" "$oracle64" --tokens 64 \
        > "$NPU_ROOT/logs/full-$alias/64/oracle.log" 2>&1
fi

if [[ ! -s "$decision/scorer.rknn" || ! -s "$decision/action.rknn" ]]; then
    echo "export $alias decision operators $(date -u +%FT%TZ)"
    USE_TF=0 "$cpu" "$scripts/export_decision_ops.py" \
        "$checkpoint" "$oracle64" "$decision" \
        > "$NPU_ROOT/logs/full-$alias/decision-export.log" 2>&1
    for op in scorer action; do
        if [[ ! -s "$decision/$op.rknn" ]]; then
            "$compiler" "$scripts/convert_onnx.py" \
                "$decision/$op.onnx" "$decision/$op.rknn" \
                > "$NPU_ROOT/logs/full-$alias/decision-$op-convert.log" 2>&1
        fi
        rm -f "$decision/$op.onnx"
    done
fi

for length in "${lengths[@]}"; do
    graphs="$NPU_ROOT/models/$alias$length"
    logs="$NPU_ROOT/logs/full-$alias/$length"
    mkdir -p "$graphs" "$logs"
    oracle="$NPU_ROOT/models/oracle-$alias-$length"
    if [[ ! -s "$oracle/encoder_$(printf '%02d' "$((count-1))").f32" ]]; then
        echo "oracle $alias $length $(date -u +%FT%TZ)"
        USE_TF=0 "$cpu" "$scripts/oracle_length.py" \
            "$checkpoint" "$oracle" --tokens "$length" \
            > "$logs/oracle.log" 2>&1
    fi

    for part in embedding_norm head; do
        if [[ "$part" == head && -s "$graphs/head_onehot.rknn" ]]; then
            continue
        fi
        graph="$graphs/$part.rknn"
        if [[ -s "$graph" ]]; then
            continue
        fi
        echo "export $alias $length $part $(date -u +%FT%TZ)"
        USE_TF=0 "$cpu" "$scripts/export_$part.py" \
            "$checkpoint" "$oracle" "$graphs" --length "$length" \
            > "$logs/$part-export.log" 2>&1
        echo "convert $alias $length $part $(date -u +%FT%TZ)"
        "$compiler" "$scripts/convert_onnx.py" \
            "$graphs/$part.onnx" "$graph" \
            > "$logs/$part-convert.log" 2>&1
        rm -f "$graphs/$part.onnx"
    done

    for index in $(seq 0 $((count - 1))); do
        printf -v number '%02d' "$index"
        layer="$graphs/layer$number"
        graph="$layer/encoder_$number.rknn"
        if [[ -s "$graph" ]]; then
            continue
        fi
        mkdir -p "$layer"
        echo "export $alias $length layer $number $(date -u +%FT%TZ)"
        USE_TF=0 "$cpu" "$scripts/export_encoder_layer.py" \
            "$checkpoint" "$oracle" "$layer" --layer "$index" --length "$length" \
            > "$logs/layer$number-export.log" 2>&1
        echo "convert $alias $length layer $number $(date -u +%FT%TZ)"
        "$compiler" "$scripts/convert_onnx.py" \
            "$layer/encoder_$number.onnx" "$graph" \
            > "$logs/layer$number-convert.log" 2>&1
        rm -f "$layer/encoder_$number.onnx" "$layer/input_nhwc.f32" \
            "$layer/expected.f32" "$layer/mask.f32"
        echo "ready $alias $length layer $number $(date -u +%FT%TZ)"
    done
    echo "ready $alias bucket $length $(date -u +%FT%TZ)"
done
