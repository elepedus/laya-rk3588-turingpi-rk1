#!/usr/bin/env bash
set -euo pipefail
root="${LAYA_MALI_ROOT:-/mnt/warm/laya-service/mali-backend}"
runtime_user="${LAYA_RK1_USER:-$(id -un)}"
firmware="${LAYA_MALI_FIRMWARE:-$root/firmware/mali_csffw_latest_jeffy.bin}"
test -f "$firmware"
if [[ -n "${LAYA_MALI_FIRMWARE_SHA256:-}" ]]; then
    printf '%s  %s\n' "$LAYA_MALI_FIRMWARE_SHA256" "$firmware" | sha256sum -c -
fi
ln -sfn "$firmware" "$root/firmware/mali_csffw.bin"
if [[ "$(cat /sys/module/firmware_class/parameters/path)" != "$root/firmware" ]]; then
    printf '%s' "$root/firmware" | sudo -n tee /sys/module/firmware_class/parameters/path > /dev/null
fi
if systemctl is-active --quiet laya-mali; then
    echo 'laya-mali is already running'
    exit 0
fi
sudo -n systemctl reset-failed laya-mali 2>/dev/null || true
if [[ "$(systemctl show laya-mali -p LoadState --value 2>/dev/null || true)" == loaded ]]; then
    sudo -n systemctl start laya-mali
    exit 0
fi
sudo -n systemd-run --unit=laya-mali --property=User="$runtime_user" \
    --property=WorkingDirectory="$root" \
    --property=Environment="LAYA_MALI_ROOT=$root" \
    --property=Restart=on-failure --property=RestartSec=5s \
    --property=RequiresMountsFor="$root" \
    --property=ExecStopPost="$root/restore-rk1.sh" \
    "$root/serve-rk1.sh"
