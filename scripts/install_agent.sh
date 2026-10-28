#!/bin/bash
# 装成登录自启的常驻服务（launchd user agent）。
#
# 为什么用 launchd 而不是 crontab：crontab 的最小粒度是分钟、不保证登录时启动、
# 崩溃不重启。这三种情况在这个场景里都会遇到（合盖、重启、模型 OOM）。
#
# 用法：
#   VP_PY=/path/to/venv/bin/python bash scripts/install_agent.sh
#   bash scripts/install_agent.sh --uninstall
set -euo pipefail

LABEL="com.local.voiceprint"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${VP_PY:-$(command -v python3)}"
LOG="$HOME/.voiceprint/logs"

if [ "${1:-}" = "--uninstall" ]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "已卸载 $LABEL"
  exit 0
fi

mkdir -p "$(dirname "$PLIST")" "$LOG"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PY</string>
    <string>$ROOT/src/app.py</string>
  </array>
  <key>WorkingDirectory</key><string>$ROOT/src</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>$LOG/agent.log</string>
  <key>StandardErrorPath</key><string>$LOG/agent.err.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "已装载 $LABEL"
echo "日志：$LOG/agent.log"
sleep 2
grep -o "http://127.0.0.1:[0-9]*" "$LOG/agent.log" | tail -1 || echo "启动中，稍后再看日志"
