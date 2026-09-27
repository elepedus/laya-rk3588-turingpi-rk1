#!/usr/bin/env bash
set -euo pipefail
root="${LAYA_MALI_ROOT:-/mnt/warm/laya-service/mali-backend}"
if [[ -f "$root/runtime.env" ]]; then
    source "$root/runtime.env"
fi
: "${LAYA_MODEL_DIR:?set LAYA_MODEL_DIR to the Laya checkpoint tree}"
export LAYA_MODEL_DIR
export HOME="$root/home"
export XDG_CACHE_HOME="$root/cache"
export TMPDIR="$root/tmp"
export LD_LIBRARY_PATH="$root/sysroot/usr/lib/aarch64-linux-gnu:$root/sysroot/usr/lib/aarch64-linux-gnu/libmali-x11:${LD_LIBRARY_PATH:-}"
export LAYA_MALI_OPENCL="${LAYA_MALI_OPENCL:-$root/sysroot/usr/lib/aarch64-linux-gnu/libmali-x11/libOpenCL.so.1}"
export LAYA_MALI_PRELOAD="${LAYA_MALI_PRELOAD:-english,multilingual,typed-decisions}"
export LAYA_MALI_LINEAR_KERNEL="${LAYA_MALI_LINEAR_KERNEL:-linear_h_transposed_rows4_cols8}"
export LAYA_MALI_SCORE_KERNEL="${LAYA_MALI_SCORE_KERNEL:-attention_scores_vec8}"
export LAYA_MALI_NORM_KERNEL="${LAYA_MALI_NORM_KERNEL:-norm_rows_vec8}"
export LAYA_MALI_WARMUP="${LAYA_MALI_WARMUP:-1}"
export LAYA_MALI_BIND="${LAYA_MALI_BIND:-0.0.0.0:8002}"
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$root/logs"
if [[ "${LAYA_MALI_ALWAYS_ON:-1}" == 1 ]]; then
    sudo -n sh -c 'printf %s always_on > /sys/devices/platform/fb000000.gpu/power_policy'
fi
cd "$root"
binary="${LAYA_MALI_BINARY:-$root/bin/laya-mali}"
test -x "$binary"
exec "$binary" serve >> "$root/logs/server.log" 2>&1
