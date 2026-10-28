#!/usr/bin/env python3
"""从 Magic 录音笔的本地 MCP 取逐字稿（讯飞 ASR，带毫秒级分段）。

Magic 的 MCP 只做了三个只读工具：magic_search / magic_list_recent / magic_get_document。
这里用标准 MCP stdio 协议直接调，不依赖客户端是否已授权。

用法:
  python magic_bridge.py list [N]                  # 列出最近的录音
  python magic_bridge.py dump <documentId> <out>   # 导出逐字稿为 json
  python magic_bridge.py match "标题关键词" <out>   # 按标题关键词导出

产物 json 结构: {"title":..., "durationMs":..., "segments":[{"start":秒,"end":秒,"text":...}]}
"""
import json
import os
import subprocess
import sys
import time

APP = "/Applications/Magic.app/Contents/MacOS/Magic"
MCP = "/Applications/Magic.app/Contents/Resources/mcp/magic-local-mcp.cjs"


class Magic:
    def __init__(self):
        env = dict(os.environ)
        env["ELECTRON_RUN_AS_NODE"] = "1"
        env["MAGIC_MCP_CLIENT_ID"] = "workbuddy"   # 必须是应用白名单里的客户端 id
        self.p = subprocess.Popen([APP, MCP], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, env=env, text=True)
        self._id = 0
        self._req("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                 "clientInfo": {"name": "workbuddy", "version": "1.0"}}, 8)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        time.sleep(0.8)

    def _send(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()

    def _req(self, method, params, wait=30):
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        t0 = time.time()
        while time.time() - t0 < wait:
            line = self.p.stdout.readline()
            if not line:
                return None
            try:
                j = json.loads(line)
            except Exception:
                continue
            if j.get("id") == rid:
                return j
        return None

    def tool(self, name, args, wait=40):
        r = self._req("tools/call", {"name": name, "arguments": args}, wait)
        if not r or "result" not in r:
            raise RuntimeError(f"{name} 调用失败: {str(r)[:200]}")
        return r["result"]["content"][0]["text"]

    def close(self):
        try:
            self.p.terminate()
        except Exception:
            pass


def to_transcript(doc):
    segs = []
    for it in doc.get("transcript", []):
        segs.append({"start": round(it["startMs"] / 1000, 2),
                     "end": round(it["endMs"] / 1000, 2),
                     "text": it.get("text", "").strip()})
    return {"title": doc.get("title"), "documentId": doc.get("documentId"),
            "durationSec": round(doc.get("durationMs", 0) / 1000, 1),
            "segments": segs}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    m = Magic()
    try:
        if cmd == "list":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            items = json.loads(m.tool("magic_list_recent", {"limit": n}))["items"]
            for it in items:
                print(f"{it['documentId']}  {it['durationMs']/60000:6.1f}分  {it['title']}")
        elif cmd in ("dump", "match"):
            if cmd == "dump":
                doc_id, out = sys.argv[2], sys.argv[3]
            else:
                kw, out = sys.argv[2], sys.argv[3]
                items = json.loads(m.tool("magic_search", {"query": kw, "limit": 5}))["items"]
                if not items:
                    sys.exit("没搜到匹配的录音")
                doc_id = items[0]["documentId"]
                print("命中:", items[0]["title"])
            doc = json.loads(m.tool("magic_get_document", {"documentId": doc_id}))
            t = to_transcript(doc)
            json.dump(t, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"{t['title']}：{len(t['segments'])} 段 → {out}")
        else:
            print(__doc__)
    finally:
        m.close()


if __name__ == "__main__":
    main()
