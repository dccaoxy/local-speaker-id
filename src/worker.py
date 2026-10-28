#!/usr/bin/env python3
"""常驻 worker：单线程、串行、可恢复。

串行不是偷懒：60 分钟音频解码成 float32 约 230MB，两个并发就是 460MB，
而实测 4 线程已经是本机最优（8 线程反而慢 20%）。并发只会让两件事都变慢。
"""
from __future__ import annotations

import os
import queue
import threading
import time
import traceback

import config
import notify
import pipeline
import store
import watch

_Q = queue.Queue()
_STOP = threading.Event()


def _progress_maker(rid):
    def cb(stage, pct):
        store.update(rid, stage=stage, progress=pct, status=store.ST_PROCESSING)
    return cb


def run_one(rid):
    rec = store.get(rid)
    if not rec:
        return
    path = rec["path"]
    if not os.path.exists(path):
        store.update(rid, status=store.ST_FAILED, error="文件不在了")
        return
    store.update(rid, status=store.ST_PROCESSING, stage="start", progress=1, error=None)
    try:
        out = pipeline.process(path, rid, progress=_progress_maker(rid))
    except Exception as e:
        attempts = int(rec.get("attempts", 0)) + 1
        store.update(rid, status=store.ST_FAILED, error=f"{type(e).__name__}: {e}",
                     attempts=attempts,
                     trace=traceback.format_exc()[-2000:])
        if attempts >= 3:
            notify.send("录音处理失败", f"{os.path.basename(path)}：{e}")
        return

    # 先落 state，再出稿：崩溃在两者之间只是「没稿」，不是「稿和台账不一致」
    store.update(rid, status=store.ST_REVIEW, stage="render", progress=99,
                 duration=out["duration"], n_clusters=out["n_clusters"],
                 clusters=out["clusters"], segs=out["segs"],
                 warnings=out["warnings"], overlap=out["overlap"],
                 md_path=out["md_path"], title=out["title"],
                 elapsed=out["elapsed"], size=rec.get("size"),
                 mtime=rec.get("mtime"))
    rec2 = store.get(rid)
    pipeline.render_rec(rec2)
    pend = store.pending_clusters(rid)
    if pend:
        names = "、".join(sorted({c.get("top1") or "陌生人" for c in pend}))
        notify.send(f"{len(pend)} 个说话人待确认",
                    f"{rec2.get('title', '')} · 最像 {names}")
    else:
        store.update(rid, status=store.ST_FINAL)


def loop():
    while not _STOP.is_set():
        rid = watch.next_queued()
        if not rid:
            time.sleep(3)
            continue
        try:
            run_one(rid)
        except Exception:
            traceback.print_exc()
        finally:
            time.sleep(0.2)


def start():
    t = threading.Thread(target=loop, name="vp-worker", daemon=True)
    t.start()
    return t


def stop():
    _STOP.set()


def kick():
    """用户点「立即处理」时叫醒 worker。"""
    _Q.put(1)
