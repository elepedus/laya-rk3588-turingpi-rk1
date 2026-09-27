#!/usr/bin/env bash
set -euo pipefail
root="${LAYA_NPU_ROOT:-/mnt/warm/laya-service/npu-backend}"
runtime_user="${LAYA_RK1_USER:-$(id -un)}"
if systemctl is-active --quiet laya-rknpu-full; then
    echo 'laya-rknpu-full is already running'
    exit 0
fi
sudo -n systemctl reset-failed laya-rknpu-full 2>/dev/null || true
if [[ "$(systemctl show laya-rknpu-full -p LoadState --value 2>/dev/null || true)" == loaded ]]; then
    sudo -n systemctl start laya-rknpu-full
    exit 0
fi
sudo -n systemd-run --unit=laya-rknpu-full --property=User="$runtime_user" \
    --property=WorkingDirectory="$root" \
    --property=Environment="LAYA_NPU_ROOT=$root" \
    --property=Restart=on-failure --property=RestartSec=5s \
    --property=RequiresMountsFor="$root" \
    "$root/serve-full-rk1.sh"
