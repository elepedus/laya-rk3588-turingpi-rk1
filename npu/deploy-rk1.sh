#!/usr/bin/env bash
# Run on a workstation after extracting a published ARM64 release archive.
set -euo pipefail
cd "$(dirname "$0")/.."
node="${LAYA_RK1_NODE:?set LAYA_RK1_NODE, for example ubuntu@rk1.local}"
remote="${LAYA_NPU_ROOT:-/mnt/warm/laya-service/npu-backend}"
binary="${1:?usage: LAYA_RK1_NODE=ubuntu@rk1.local ./npu/deploy-rk1.sh path/to/laya-rknpu}"
health_url="${LAYA_RK1_HEALTH_URL:-http://127.0.0.1:8003/health}"
test -x "$binary"
if [[ ! "$remote" =~ ^/[A-Za-z0-9_./-]+$ ]]; then
    echo 'LAYA_NPU_ROOT must be an absolute path without spaces or shell characters' >&2
    exit 2
fi
if [[ ! "$health_url" =~ ^https?://[A-Za-z0-9_.:/-]+$ ]]; then
    echo 'LAYA_RK1_HEALTH_URL contains unsupported characters' >&2
    exit 2
fi
ssh "$node" "mkdir -p '$remote/bin' '$remote/logs'"
ssh "$node" "test -f '$remote/runtime.env'" || {
    echo "create $remote/runtime.env from npu/runtime.env.example before deployment" >&2
    exit 2
}
scp "$binary" "$node:$remote/bin/laya-rknpu"
scp npu/serve-full-rk1.sh npu/start-full-rk1.sh "$node:$remote/"
ssh "$node" bash -s -- "$remote" "$health_url" <<'REMOTE'
set -euo pipefail
root=$1
health_url=$2
chmod +x "$root/bin/laya-rknpu" "$root/serve-full-rk1.sh" "$root/start-full-rk1.sh"
if systemctl is-active --quiet laya-rknpu-full; then
    sudo -n systemctl stop laya-rknpu-full
fi
LAYA_NPU_ROOT="$root" "$root/start-full-rk1.sh"
for attempt in $(seq 1 25); do
    if curl -fsS --max-time 2 "$health_url"; then
        exit 0
    fi
    sleep 1
done
echo 'NPU service did not become healthy' >&2
exit 1
REMOTE
