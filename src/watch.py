#!/usr/bin/env python3
"""扫目录、判断文件是否「已经同步完」、入队。

为什么不用 fswatch / FSEvents：本来就要等 60 秒稳定闸，事件驱动的唯一收益是
「早 60 秒知道」，却要多维护一个常驻子进程和一套事件去重。轮询够用。

三个坑，缺一个都会出事：
  1) 半成品：OneDrive 边下边写。用「size+mtime 连续 60 秒不变」当闸，
     而不是「文件大小 > 0」——后者对 100MB 的录音等于没闸。
  2) 重命名：身份用内容指纹，不用路径。改完名不会重跑，也不会变成两份。
  3) 自触发：我们自己会往同一目录写 .说话人.md / 转写稿，扫描必须排除非音频。
"""
from __future__ import annotations

import os
import time

import config
import store

AUDIO_EXT = {".mp3", ".m4a", ".opus", ".wav", ".aac", ".flac", ".amr", ".wma"}

# 我们自己转码时的中间文件名。正常情况下它们只落在私有工作目录，但万一有人用
# CLI 跑过、把 bl8/bl16 留在录音目录，扫描器必须认得出这不是新录音——
# 否则会「处理一个文件 → 产出两个文件 → 各再处理一遍」，无限派生。
DERIVED = {"a.wav", "a16.wav", "bl8.wav", "bl16.wav"}


def _is_audio(p):
    name = os.path.basename(p)
    if name.startswith(".") or name in DERIVED:
        return False
    return os.path.splitext(p)[1].lower() in AUDIO_EXT


def pick_candidates(dir_path=None):
    d = dir_path or config.RECORDING_DIR
    if not os.path.isdir(d):
        return []
    out = []
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if not os.path.isfile(p) or not _is_audio(p):
            continue
        try:
            st = os.stat(p)
        except OSError:
            continue
        if st.st_size == 0:
            continue
        out.append((p, st.st_size, st.st_mtime))
    return out


def scan(dir_path=None, now=None):
    """→ 本轮新入队的 rid 列表。稳定不足 STABLE_SEC 的留在 waiting_stable。"""
    now = time.time() if now is None else now
    state = store.load_state()
    recs = state.setdefault("recordings", {})
    stability = state.setdefault("stability", {})
    by_path = {r.get("path"): rid for rid, r in recs.items() if r.get("path")}
    queued = []
    seen = set()

    for path, size, mtime in pick_candidates(dir_path):
        seen.add(path)
        key = f"{size}:{int(mtime)}"
        prev = stability.get(path) or {}
        if prev.get("key") != key:
            stability[path] = {"key": key, "since": now}
            continue                                    # 还在变，下一轮再看
        if now - prev.get("since", now) < config.STABLE_SEC:
            continue                                    # 稳定时间不够

        rid = by_path.get(path)
        if rid:
            # 同一路径已知并且文件没变：无论什么状态都跳过。
            # **失败也跳过**——自动重试会让坏文件每 5 秒进一次队列，
            # 永远收敛不了（实测过：一段 24 分钟的录音因此被反复重跑）。
            # 重试是用户的动作（页面上的「重试」），不是扫描器的动作。
            if recs[rid].get("size") == size and recs[rid].get("mtime") == mtime:
                continue
        fp = store.fingerprint(path, size, mtime)
        if fp in recs:
            recs[fp]["path"] = path                  # 重命名：身份不变，只更路径
            continue

        rec = recs.setdefault(fp, {"rid": fp, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        if rec.get("status") in (store.ST_QUEUED, store.ST_PROCESSING):
            continue
        rec.update({"path": path, "size": size, "mtime": mtime,
                    "status": store.ST_QUEUED, "stage": "queued", "progress": 0,
                    "error": None,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        recs[fp] = rec
        queued.append(fp)

    for p in list(stability):
        if p not in seen:
            stability.pop(p, None)
    state["stability"] = stability
    store.save_state(state)
    return queued


def recover():
    """进程重启：把卡在 processing 的放回队列。产物按 rid 命名，重跑是幂等的。"""
    state = store.load_state()
    n = 0
    for rec in state.get("recordings", {}).values():
        if rec.get("status") == store.ST_PROCESSING:
            rec["status"] = store.ST_QUEUED
            rec["stage"] = "requeued"
            n += 1
    if n:
        store.save_state(state)
    return n


def next_queued():
    state = store.load_state()
    for rid, rec in sorted(state.get("recordings", {}).items(),
                           key=lambda kv: kv[1].get("created_at", "")):
        if rec.get("status") == store.ST_QUEUED:
            return rid
    return None
