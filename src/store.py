#!/usr/bin/env python3
"""状态与判定台账。

两个载体，职责不同，别混：
  state.json       **可重建**的运行状态（扫描到哪、处理到哪、卡在哪）。
                   丢了最多重跑一遍，所以进程崩了直接重建即可。
  decisions.jsonl  **不可重建**的人工判定，append-only。
                   这是唯一记录「谁在哪一刻把哪个簇判给了谁」的地方，
                   删了就再也做不出阈值标定。所以只追加，撤销也是追加一条。

为什么不用 SQLite：单用户单机、写入全部串行（只有一个 worker + 一个 HTTP 进程），
两张表能表达的东西不值得引入事务层。但原子写必须做——半截 JSON 会让整个库读不出来。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

import config

_LOCK = threading.RLock()

ST_DISCOVERED = "discovered"
ST_WAITING = "waiting_stable"
ST_QUEUED = "queued"
ST_PROCESSING = "processing"
ST_REVIEW = "awaiting_review"
ST_FINAL = "finalized"
ST_FAILED = "failed"

OPEN_STATES = (ST_DISCOVERED, ST_WAITING, ST_QUEUED, ST_PROCESSING, ST_REVIEW)


# ── 基础 IO ─────────────────────────────────────────────────────────────
def _atomic_json(path, obj):
    """tmp + rename。**不用 tempfile**：TMPDIR 不可用时 mkstemp 会退到当前目录里
    随机建目录，守护进程跑久了就是一堆垃圾；而且我们本来就需要「同目录 tmp」才能
    保证 rename 是原子的（跨文件系统 rename 会失败）。"""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError:
        # 有些文件系统不允许 rename 覆盖（SMB、部分同步盘、受限环境）。
        # 这时退化成「直接写」：牺牲「原子替换」的保证，换「能用」。
        # 状态文件本身可重建，被截断的代价是重跑一遍，不是丢数据。
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        try:
            os.remove(tmp)
        except Exception:
            pass
    except Exception:
        # 写失败必须自己收尸。否则每失败一次就落一个 .tmp，扫描器每 5 秒跑一次，
        # 一天下来就是几万个垃圾文件（真实发生过：~/.voiceprint 里 32,177 个）。
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise


def fingerprint(path, size, mtime):
    """内容指纹：大小 + 首尾各 256KB。用来在重命名后仍认出同一个文件。

    不用整文件哈希——OneDrive 上的大文件可能是占位符，读全量等于触发下载。
    """
    h = hashlib.sha1()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(262144))
        if size > 524288:
            f.seek(-262144, os.SEEK_END)
            h.update(f.read(262144))
    return h.hexdigest()[:16]


# ── state ───────────────────────────────────────────────────────────────
def load_state():
    if not os.path.exists(config.STATE):
        return {"version": 1, "recordings": {}}
    try:
        with open(config.STATE, encoding="utf-8") as f:
            s = json.load(f)
        s.setdefault("recordings", {})
        return s
    except Exception:
        # 半截文件：不抛，重建。重建的代价只是重跑，不是丢数据（判定在 jsonl 里）
        return {"version": 1, "recordings": {}, "recovered": True}


def save_state(state):
    _atomic_json(config.STATE, state)


def update(rid, **fields):
    with _LOCK:
        s = load_state()
        rec = s["recordings"].setdefault(rid, {"rid": rid})
        rec.update(fields)
        rec["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save_state(s)
        return rec


def get(rid):
    return load_state()["recordings"].get(rid)


def all_recordings():
    return load_state()["recordings"]


# ── decisions ───────────────────────────────────────────────────────────
def append_decision(rec: dict):
    """只追加。撤销也是一条新记录（kind=revoke），不改写历史。"""
    rec = dict(rec)
    rec.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    rec.setdefault("kind", "decide")
    with _LOCK:
        os.makedirs(os.path.dirname(os.path.abspath(config.DECISIONS)), exist_ok=True)
        with open(config.DECISIONS, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    return rec


def all_decisions():
    if not os.path.exists(config.DECISIONS):
        return []
    out = []
    with open(config.DECISIONS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def verdicts(rid: str) -> dict:
    """{cluster_id: 最新一条未被撤销的判定}。撤销 = 后来有一条同簇的 revoke。"""
    latest = {}
    for d in all_decisions():
        if d.get("rid") != rid:
            continue
        cid = d.get("cluster_id")
        if cid is None:
            continue
        if d.get("kind") == "revoke":
            latest.pop(int(cid), None)
        else:
            latest[int(cid)] = d
    return latest


def pending_clusters(rid: str) -> list:
    """还没被人工处置、且需要问的簇（按 state 里记录的 ask 列表算）。"""
    rec = get(rid) or {}
    done = set(verdicts(rid))
    return [c for c in rec.get("clusters", []) if c.get("ask") and c["cluster_id"] not in done]


def rename_person(old: str, new: str, *, purge=False) -> dict:
    """改一个人的名字（或把名字并入另一个人）。

    purge=True 时把旧名字的来源标成待删（用于「判错了，库里的也别要了」）。
    名字分裂是这套系统里最难修的错——两条记录各自只有一个来源，双双拿不到
    静默资格，还都认不出同一个人，所以改名要和库一起动。
    """
    res = {"renamed": 0, "purged": 0}
    for sid, m in list(_src_items()):
        if m.get("name") != old:
            continue
        if purge:
            res["purged"] += 1
            continue
        m2 = dict(m)
        m2["name"] = new
        _write_source_meta(sid, m2)
        res["renamed"] += 1
    return res


def _src_items():
    import protolib
    return list(protolib.list_sources().items())


def _write_source_meta(sid, meta):
    import protolib
    os.makedirs(protolib.EMB_DIR, exist_ok=True)
    with open(os.path.join(protolib.EMB_DIR, f"{sid}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def known_names() -> list:
    """人工候选列表用：库里出现过的人名（含不够格的，他要能选到才能补来源）。"""
    names = set()
    for d in all_decisions():
        if d.get("name"):
            names.add(d["name"])
    try:
        import protolib
        names |= set(protolib.list_sources_names())
    except Exception:
        pass
    return sorted(names)


def stats() -> dict:
    s = load_state()
    recs = s.get("recordings", {})
    shown = [d for d in all_decisions() if d.get("shown")]
    b = [d for d in shown if d.get("verdict") in ("skip", "none")]
    return {
        "recordings": len(recs),
        "awaiting": sum(1 for r in recs.values() if r.get("status") == ST_REVIEW),
        "finalized": sum(1 for r in recs.values() if r.get("status") == ST_FINAL),
        "cards_shown": len(shown),
        "b_class": len(b),
        "b_ratio": round(len(b) / len(shown), 3) if shown else None,
    }
