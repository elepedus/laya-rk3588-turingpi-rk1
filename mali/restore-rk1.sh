#!/usr/bin/env bash
set -euo pipefail
sudo -n sh -c 'printf %s coarse_demand > /sys/devices/platform/fb000000.gpu/power_policy'
