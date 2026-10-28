#!/usr/bin/env python3
"""macOS 通知。有 terminal-notifier 就用它（能点开链接），否则退回 osascript。

一条录音最多发一条通知，这是硬约束：推送不是免费的，每次都在消耗耐心，
而「打扰次数」是这套系统唯一真正稀缺的资源。
"""
from __future__ import annotations

import shutil
import subprocess


def _escape(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def send(title: str, body: str, url: str | None = None, group: str = "voiceprint") -> bool:
    if shutil.which("terminal-notifier"):
        args = ["terminal-notifier", "-title", title, "-message", body, "-group", group]
        if url:
            args += ["-open", url]
        try:
            subprocess.run(args, check=False, capture_output=True, timeout=10)
            return True
        except Exception:
            pass
    script = (f'display notification "{_escape(body)}" '
              f'with title "{_escape(title)}"')
    try:
        subprocess.run(["osascript", "-e", script], check=False,
                       capture_output=True, timeout=10)
        return True
    except Exception:
        return False
