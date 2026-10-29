#!/usr/bin/env bash
# 启动或停止 asknjupt 后端（8787）和 NextChat 前端（3000）。
# 进程由 systemd 守护：asknjupt.service、asknjupt-frontend.service。
set -euo pipefail
if [[ "$(id -u)" -ne 0 ]]; then
  exec sudo "$0" "$@"
fi

case "${1:-start}" in
  start)
    systemctl start asknjupt.service asknjupt-frontend.service
    systemctl --no-pager --full status asknjupt.service asknjupt-frontend.service
    ;;
  stop)
    systemctl stop asknjupt-frontend.service asknjupt.service
    ;;
  *)
    echo "usage: $0 start|stop" >&2
    exit 1
    ;;
esac
