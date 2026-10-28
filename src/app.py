#!/usr/bin/env python3
"""本地网页服务：127.0.0.1:8765。

只监听回环地址，不是可配置项——录音里是同事和外部候选人，这套东西一步都不该
出本机。要远程用得自己搭隧道，那是另一件事。

进程模型：**一个进程**。uvicorn 起在主线，watcher 与 worker 是它的两个线程。
不做 daemon + 客户端两段式：确认回灌需要「写台账 → 重渲染 → 可能入库」三件事
在一次请求里原子完成，拆进程只会多出一个不一致窗口。
"""
from __future__ import annotations

import json
import os
import threading
import time

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

import config
import enroll
import pipeline
import protolib
import store
import watch
import worker

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.abspath(os.path.join(HERE, "..", "web"))

_state = {"watcher": None, "last_scan": 0.0}


def _watcher_loop():
    fails, last_msg, interval = 0, "", 5.0
    while True:
        try:
            new = watch.scan()
            for rid in new:
                worker.kick()
            _state["last_scan"] = time.time()
            fails, interval, last_msg = 0, 5.0, ""
        except Exception as e:
            fails += 1
            msg = f"{type(e).__name__}: {e}"
            # 同一个错误只打一次，并且指数退避：环境坏掉时不能每 5 秒刷一条日志、
            # 也不能每 5 秒重试一次（那正是上回堆出 3 万个 tmp 的原因）。
            if msg != last_msg:
                print(f"[watch] 第 {fails} 次失败 {msg}", flush=True)
                last_msg = msg
            interval = min(60.0, 5.0 * (2 ** min(fails, 4)))
        time.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    n = watch.recover()
    if n:
        print(f"[startup] 恢复 {n} 个中断任务", flush=True)
    worker.start()
    t = threading.Thread(target=_watcher_loop, name="vp-watch", daemon=True)
    t.start()
    _state["watcher"] = t
    yield


app = FastAPI(title="本地说话人识别", docs_url=None, redoc_url=None, lifespan=lifespan)


# ── 页面 ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    return _html("index.html")


@app.get("/r/{rid}", response_class=HTMLResponse)
def confirm_page(rid: str):
    return _html("index.html")


def _html(name):
    p = os.path.join(WEB, name)
    if not os.path.exists(p):
        raise HTTPException(500, f"缺少页面 {p}")
    return FileResponse(p, media_type="text/html; charset=utf-8")


@app.get("/static/{name}")
def static(name: str):
    p = os.path.join(WEB, name)
    if not os.path.exists(p) or "/" in name:
        raise HTTPException(404)
    return FileResponse(p)


# ── 接口 ────────────────────────────────────────────────────────────────
def _public(rec):
    """列表页只看得到「有没有事要我做」，看不到任何分数。"""
    pend = store.pending_clusters(rec["rid"]) if rec.get("clusters") else []
    return {
        "rid": rec["rid"],
        "title": rec.get("title") or os.path.basename(rec.get("path", "")),
        "duration": rec.get("duration", 0),
        "status": rec.get("status"),
        "stage": rec.get("stage"),
        "progress": rec.get("progress", 0),
        "error": rec.get("error"),
        "n_clusters": rec.get("n_clusters", 0),
        "pending": len(pend),
        "updated_at": rec.get("updated_at"),
        "md_path": rec.get("md_path"),
    }


@app.get("/api/recordings")
def list_recordings():
    recs = [r for r in store.all_recordings().values() if r.get("path")]
    recs.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
    return {"items": [_public(r) for r in recs], "stats": store.stats()}


@app.get("/api/recordings/{rid}")
def get_recording(rid: str):
    rec = store.get(rid)
    if not rec:
        raise HTTPException(404)
    names = store.known_names()
    verdicts = store.verdicts(rid)
    pipeline._fill_self_excluded(rec)
    cards = []
    for c in rec.get("clusters", []):
        c = dict(c)
        c.pop("_s", None)
        c["decided"] = verdicts.get(c["cluster_id"])
        c["candidates"] = names
        c["clip_url"] = (f"/clip/{rid}/{c['cluster_id']}.wav" if c.get("clip") else None)
        cards.append(c)
    # 打开即算「看过」，这是 B 类指标的分母
    if not rec.get("opened_at"):
        store.update(rid, opened_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return {"rid": rid, "title": rec.get("title"), "duration": rec.get("duration"),
            "warnings": rec.get("warnings", []), "clusters": cards,
            "names": names, "thresholds": config.health()}


class Decide(BaseModel):
    rid: str
    cluster_id: int
    verdict: str            # name | skip | new | none
    name: str | None = None
    persist: bool = False   # 新人入库才有意义；不设默认，必须显式二选一
    new_name_only: bool = False   # 「记住他，以后自动认」


@app.post("/api/decisions")
def decide(d: Decide):
    rec = store.get(d.rid)
    if not rec:
        raise HTTPException(404)
    cluster = next((c for c in rec.get("clusters", [])
                    if c["cluster_id"] == d.cluster_id), None)
    if cluster is None:
        raise HTTPException(404, "没有这个簇")
    if d.verdict not in ("name", "skip", "new", "none"):
        raise HTTPException(400, "verdict 非法")

    src_id = None
    enroll_report = None
    if d.verdict in ("name", "new") and not d.name:
        raise HTTPException(400, "给了名字才能落名")
    if d.verdict == "skip":
        d.name = None

    # 新人入库：先从这段音频把该簇的窗提出来入库，再写台账。
    # 顺序很重要——反过来会出现「台账已记为某人、库里却没有他的样本」。
    if d.verdict == "new" and d.persist and d.name:
        src_id, enroll_report = _enroll_from_cluster(rec, d.cluster_id, d.name)
        if not src_id:
            raise HTTPException(400, enroll_report.get("error", "入库失败"))

    ev = store.append_decision({
        "rid": d.rid, "cluster_id": d.cluster_id,
        "verdict": d.verdict, "name": d.name,
        "persist": bool(d.persist), "source_id": src_id,
        "shown": bool(rec.get("opened_at")),
        "reason": cluster.get("reason"),
        "evidence": cluster.get("_s") or {},
        "scorer": config.SCORER, "threshold": config.T_AUTO,
        "proto_version": protolib.current_version(),
    })
    pipeline.render_rec(store.get(d.rid))
    _maybe_finalize(d.rid)
    return {"ok": True, "decision": ev, "enroll": enroll_report}


def _enroll_from_cluster(rec, cluster_id, name):
    """把该簇的窗入库。用处理时留下的 16k 音频，避免重新转码。"""
    wav = os.path.join(config.WORK_DIR, rec["rid"], "a16.wav")
    if not os.path.exists(wav):
        raise HTTPException(400, "找不到原始 16k 音频，无法入库（可点重新处理）")
    spans = [(s["start"], s["end"]) for s in rec.get("segs", [])
             if int(s["speaker"]) == int(cluster_id)]
    if not spans:
        return None, {"error": "该簇没有可用的分段"}
    try:
        import sherpa_onnx
        import soundfile as sf
        import winmatch
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        ext = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=config.EMB_MODEL, num_threads=config.THREADS))
        vecs = enroll.embed_spans(audio, sr, spans, ext=ext)
        rep = {"n_raw": int(len(vecs))}
        if len(vecs) < config.N_MIN_ASK:
            return None, {**rep, "error": "有效语音太少，先不入库（下次再问一次即可）"}
        pur = enroll.purity(vecs)
        rep["purity"] = pur
        if not pur["ok"]:
            # 混合簇不入库：入了等于往库里加吸引子，污染面比不注册更大。
            # 但用户的判断照记（他可能认得那 3 个人里的一个），只是不写样本。
            rep["error"] = (f"这段前后像两个人（一致度 {pur['time_split']}），"
                            "已记下你的判断，但没有写进声纹库——混着入库会把库带偏")
            return None, rep
        kept = enroll.filter_windows(vecs)
        sid = protolib.save_source(name, kept, wav=wav, segs="", cluster=cluster_id,
                                   gate_db=0.0, recording=rec["rid"])
        rep["source_id"] = sid
        rep["n_kept"] = int(len(kept))
        rep["version"] = protolib.build_version()
        return sid, rep
    except HTTPException:
        raise
    except Exception as e:
        return None, {"error": f"{type(e).__name__}: {e}"}


class Undo(BaseModel):
    rid: str
    cluster_id: int
    purge_source: bool = True     # 撤销入库时连库里的样本一起删（那是隐私，不是体验）


@app.post("/api/undo")
def undo(u: Undo):
    vs = store.verdicts(u.rid)
    prev = vs.get(u.cluster_id)
    if not prev:
        raise HTTPException(404, "没有可撤销的判定")
    purged = False
    if u.purge_source and prev.get("source_id"):
        purged = protolib.delete_source(prev["source_id"])
        if purged:
            protolib.build_version()
    store.append_decision({"rid": u.rid, "cluster_id": u.cluster_id,
                           "kind": "revoke", "verdict": None,
                           "name": None, "shown": True})
    pipeline.render_rec(store.get(u.rid))
    store.update(u.rid, status=store.ST_REVIEW)
    return {"ok": True, "purged": purged}


class Skip(BaseModel):
    rid: str
    clusters: int | None = None      # 「人数不对」时指定，仅本次重跑生效


@app.post("/api/finalize")
def finalize(s: Skip):
    rec = store.get(s.rid)
    if not rec:
        raise HTTPException(404)
    if not rec.get("clusters"):
        raise HTTPException(400, "还没处理完")
    pipeline.render_rec(rec)
    store.update(s.rid, status=store.ST_FINAL)
    return {"ok": True}


@app.post("/api/retry")
def retry(s: Skip):
    rec = store.get(s.rid)
    if not rec:
        raise HTTPException(404)
    fields = {"status": store.ST_QUEUED, "stage": "queued", "progress": 0, "error": None}
    if s.clusters:
        fields["clusters_hint"] = int(s.clusters)
    store.update(s.rid, **fields)
    worker.kick()
    return {"ok": True, "clusters": s.clusters}


@app.post("/api/scan")
def scan_now():
    new = watch.scan()
    worker.kick()
    return {"queued": new}


@app.get("/clip/{rid}/{cid}.wav")
def clip(rid: str, cid: int):
    rec = store.get(rid)
    if not rec:
        raise HTTPException(404)
    c = next((x for x in rec.get("clusters", []) if x["cluster_id"] == cid), None)
    p = (c or {}).get("clip") or os.path.join(config.CLIP_CACHE_DIR, rid, f"c{cid}.wav")
    if not os.path.exists(p):
        raise HTTPException(404, "这段没有试听音频")
    return FileResponse(p, media_type="audio/wav")


@app.get("/health")
def health():
    return {**config.health(), "stats": store.stats(),
            "clusters_default": config.CLUSTERS_DEFAULT,
            "protos": protolib.status(),
            "recording_dir": config.RECORDING_DIR,
            "last_scan": _state["last_scan"]}


def _maybe_finalize(rid):
    """没有待确认的簇就自动定稿——「不用你管」是默认路径，不是特例。"""
    if not store.pending_clusters(rid):
        rec = store.get(rid) or {}
        if rec.get("clusters"):
            store.update(rid, status=store.ST_FINAL)


def main():
    import socket
    import uvicorn
    base = int(os.environ.get("VP_PORT", "8765"))
    port = base
    for _ in range(10):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                break                       # 没人听，就用它
        port += 1
    if port != base:
        print(f"[app] 端口 {base} 被占用，改用 {port}", flush=True)
    try:
        os.makedirs(os.path.dirname(config.PORT_FILE), exist_ok=True)
        with open(config.PORT_FILE, "w") as f:
            f.write(str(port))
    except Exception:
        pass
    print(f"[app] http://127.0.0.1:{port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
