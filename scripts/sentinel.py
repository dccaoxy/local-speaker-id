#!/usr/bin/env python3
"""菜单栏哨兵：只显示「还有几个待确认」，点一下打开确认页。

它不是界面——真正的确认界面是浏览器里的那张页面。这里只解决一件事：
窗口关着的时候，你扫一眼菜单栏就知道有没有事要做。

依赖 rumps（`pip install rumps`）。没装也能跑，退化成终端里每 30 秒打一行。
端口从 ~/.voiceprint/port 读（agent 启动时会写），读不到就用 8765。
"""
import json
import os
import sys
import time
import urllib.request

PORT_FILE = os.path.expanduser("~/.voiceprint/port")


def base_url():
    try:
        port = open(PORT_FILE).read().strip()
    except Exception:
        port = os.environ.get("VP_PORT", "8765")
    return f"http://127.0.0.1:{port}"


def stats():
    try:
        with urllib.request.urlopen(f"{base_url()}/health", timeout=3) as r:
            return json.load(r)["stats"]
    except Exception:
        return None


def run_cli():
    last = None
    while True:
        s = stats()
        if s is None:
            line = "服务没在跑"
        else:
            line = f"待确认 {s['awaiting']} 段 · 已定稿 {s['finalized']} 段"
        if line != last:
            print(f"[{time.strftime('%H:%M')}] {line}", flush=True)
            last = line
        time.sleep(30)


def run_bar():
    import rumps
    import webbrowser

    class Bar(rumps.App):
        def __init__(self):
            super().__init__("🎙", quit_button="退出")
            self.timer = rumps.Timer(self.tick, 20)
            self.timer.start()

        def tick(self, _):
            s = stats()
            if not s:
                self.title = "🎙 ?"
                return
            self.title = "🎙" if not s["awaiting"] else f"🎙 {s['awaiting']}"
            self.menu.clear()
            self.menu = [
                rumps.MenuItem(f"待确认 {s['awaiting']} 段",
                               callback=lambda _: webbrowser.open(base_url())),
                rumps.MenuItem(f"已定稿 {s['finalized']} 段"),
                None,
                rumps.MenuItem("打开页面", callback=lambda _: webbrowser.open(base_url())),
            ]

        @rumps.clicked("打开页面")
        def open_page(self, _):
            webbrowser.open(base_url())

    Bar().run()


if __name__ == "__main__":
    try:
        run_bar()
    except ImportError:
        print("没装 rumps，退化成终端输出（pip install rumps 可启用菜单栏）")
        run_cli()
