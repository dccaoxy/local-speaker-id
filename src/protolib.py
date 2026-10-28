#!/usr/bin/env python3
"""声纹原型的来源级存储与版本管理。

为什么不用单个 protos.npz：
  实测发现库里的「张彤」只有 42 个窗、且全部来自同一次录音，他的原型编码的是
  **信道**而不是人——同一段录音里三个互斥簇全部指向他（0.568/0.614/0.581，ρ≈1.0），
  而换到另一段录音（山西）立刻掉到 0.180/0.345。四条静默条件全部通过，会静默署错名。
  单个 npz 无法回答「这个人由哪几次录音组成」，也就无法做跨来源自检，
  所以必须按来源分文件存。

目录结构：
  db/emb/<source_id>.npy / .json      一次入库动作 = 一个来源，不可变、不可复用
  db/protos/<version>/<name>.npy      人级合并矩阵
  db/protos/<version>/manifest.json   人 ← 来源 的映射、每人窗数、verified 标记
  db/protos/CURRENT                   单行版本号，tmp+rename 原子切换

两个硬门槛（缺一不可）：
  N_PROTO_MIN   原型窗数下限。张彤 42 窗 → 不得参与静默署名。
  N_SOURCE_MIN  独立来源数下限。单来源原型无法与信道解耦。
  verified      用来源 A 的原型去匹配来源 B 的窗，中位数 ≥ T_AUTO 才算通过。
                单来源时**无法自检**，一律 unverified——这是「没有证据」，
                不是「证据表明没问题」，两者不能混。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid

import numpy as np

import config

EMB_DIR = config.EMB_CACHE_DIR
PROTOS_DIR = config.PROTOS_DIR
CURRENT = config.PROTOS_CURRENT

WIN, HOP = config.WIN, config.HOP


# ── 基础 ────────────────────────────────────────────────────────────────
def _norm(v: np.ndarray) -> np.ndarray:
    if len(v) == 0:
        return v.reshape(0, v.shape[-1] if v.ndim == 2 else 192)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def _atomic_write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def new_source_id(name: str) -> str:
    """来源 id 必须唯一对应一次入库动作，永不复用。

    撤销是按 source_id 精确删除，复用 id 会删错来源。
    """
    stamp = time.strftime("%Y%m%d%H%M%S")
    return f"{name}-{stamp}-{uuid.uuid4().hex[:6]}"


# ── 来源层（L1）────────────────────────────────────────────────────────
def save_source(name: str, vecs: np.ndarray, *, wav: str = "", segs: str = "",
                cluster: int = 0, gate_db: float | None = None,
                model_mtime: float | None = None, recording: str = "") -> str:
    """把一次入库动作的窗向量落盘，返回 source_id。

    recording：这些窗来自哪一段录音。**必填**（不知道就传空），因为它决定了
    打分时能不能把这个来源剔除掉——「自己给自己作证」是面试录音那次静默错名的
    唯一成因，剔除靠的就是这个字段。
    """
    vecs = _norm(np.asarray(vecs, dtype=np.float32))
    if len(vecs) == 0:
        raise ValueError("空向量，拒绝入库")
    sid = new_source_id(name)
    os.makedirs(EMB_DIR, exist_ok=True)
    np.save(os.path.join(EMB_DIR, f"{sid}.npy"), vecs)
    meta = {
        "source_id": sid, "name": name, "n_win": int(len(vecs)),
        "wav": os.path.abspath(wav) if wav else "", "segs": segs, "cluster": cluster,
        "recording": recording or (os.path.basename(wav) if wav else ""),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "win": WIN, "hop": HOP, "gate_db": gate_db,
        "emb_model": os.path.basename(config.EMB_MODEL),
        "emb_model_mtime": model_mtime,
        # 一次入库动作同时产出的可删除单元：撤销时按它精确回滚
        "deletable": True,
    }
    _atomic_write(os.path.join(EMB_DIR, f"{sid}.json"),
                  json.dumps(meta, ensure_ascii=False, indent=2))
    return sid


def list_sources() -> dict[str, dict]:
    out = {}
    if not os.path.isdir(EMB_DIR):
        return out
    for fn in sorted(os.listdir(EMB_DIR)):
        if fn.endswith(".json"):
            try:
                m = json.load(open(os.path.join(EMB_DIR, fn), encoding="utf-8"))
                out[m["source_id"]] = m
            except Exception:
                continue
    return out


def delete_source(sid: str) -> bool:
    """撤销一次入库动作。两个文件一起删，不留孤儿。"""
    ok = False
    for ext in (".npy", ".json"):
        p = os.path.join(EMB_DIR, f"{sid}{ext}")
        if os.path.exists(p):
            os.remove(p)
            ok = True
    return ok


def list_sources_names() -> list[str]:
    """库里出现过的人名，含不够静默资格的——候选列表必须能选到他们。"""
    return sorted({m.get("name", "") for m in list_sources().values() if m.get("name")})


def _load_emb(sid: str) -> np.ndarray:
    return _norm(np.load(os.path.join(EMB_DIR, f"{sid}.npy")).astype(np.float32))


# ── 跨来源自检 ─────────────────────────────────────────────────────────
def verify_lo_source(vecs_by_rec: dict[str, np.ndarray],
                     t_auto: float | None = None) -> dict:
    """leave-one-recording-out：用其余录音的原型去匹配留出的那次录音。

    **按录音分组，不按来源 id。** 同一段录音存两次是两个 source，但那不是两份
    独立证据——它们共享信道，互相匹配必然高分，会伪造出 verified=True。
    「独立」的单位是录音，不是入库动作。

    单次录音时无法自检 → verified=False。这是「没有证据」，不是「证明没问题」。
    """
    t_auto = config.T_AUTO if t_auto is None else t_auto
    sides = [(k, v) for k, v in vecs_by_rec.items() if len(v)]
    if len(sides) < 2:
        return {"verified": False, "why": "single_source", "per_recording": {}}

    per = {}
    for held_id, held in sides:
        others = [v for k, v in sides if k != held_id]
        proto = np.concatenate(others, axis=0)
        s = (held @ proto.T).max(axis=1)
        per[held_id] = round(float(np.median(s)), 3)

    worst = min(per.values())
    return {"verified": worst >= t_auto, "why": "ok" if worst >= t_auto else "below_t_auto",
            "per_recording": per, "worst": worst}


# ── 版本层（L2）────────────────────────────────────────────────────────
def build_version(*, t_auto: float | None = None, force: bool = False) -> str:
    """从 emb/ 重建一个人级版本，写 manifest + 原子切换 CURRENT。"""
    srcs = list_sources()
    by_name: dict[str, dict[str, np.ndarray]] = {}
    for sid, m in srcs.items():
        p = os.path.join(EMB_DIR, f"{sid}.npy")
        if not os.path.exists(p):
            continue
        by_name.setdefault(m["name"], {})[sid] = _load_emb(sid)

    if not by_name:
        raise RuntimeError(f"没有任何来源可合并（{EMB_DIR} 为空），先注册一个人")

    version = time.strftime("%Y%m%d-%H%M%S")
    vdir = os.path.join(PROTOS_DIR, version)
    os.makedirs(vdir, exist_ok=True)

    people = {}
    for name, by_src in by_name.items():
        mat = np.concatenate([by_src[k] for k in sorted(by_src)], axis=0)
        np.save(os.path.join(vdir, f"{name}.npy"), mat)
        by_rec: dict[str, list] = {}
        for sid, vecs in by_src.items():
            rec = srcs[sid].get("recording") or sid
            by_rec.setdefault(rec, []).append(vecs)
        by_rec = {k: np.concatenate(v, axis=0) for k, v in by_rec.items()}
        chk = verify_lo_source(by_rec, t_auto)
        n_win = int(len(mat))
        n_src = len(by_rec)          # 独立录音数，不是入库次数
        people[name] = {
            "n_win": n_win, "n_sources": n_src, "sources": sorted(by_src),
            "recordings": sorted(by_rec),
            "verified": bool(chk["verified"]), "verify": chk,
            "enough_windows": n_win >= getattr(config, "N_PROTO_MIN", 100),
            "enough_sources": n_src >= getattr(config, "N_SOURCE_MIN", 2),
        }

    manifest = {
        "version": version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "win": WIN, "hop": HOP,
        "emb_model": os.path.basename(config.EMB_MODEL),
        "scorer_params": f"window@{WIN}/{HOP}|gate={getattr(config, 'RHO_WIN_GATE', 0.36)}"
                         f"|rho={getattr(config, 'RHO_MIN', 0.70)}",
        "t_auto": config.T_AUTO,
        "people": people,
    }
    _atomic_write(os.path.join(vdir, "manifest.json"),
                  json.dumps(manifest, ensure_ascii=False, indent=2))
    _atomic_write(CURRENT, version)
    return version


def read_manifest(version: str | None = None) -> dict:
    version = version or current_version()
    if not version:
        return {}
    p = os.path.join(PROTOS_DIR, version, "manifest.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p, encoding="utf-8"))


def current_version() -> str | None:
    if not os.path.exists(CURRENT):
        return None
    v = open(CURRENT, encoding="utf-8").read().strip()
    return v or None


def load_current() -> tuple[dict[str, np.ndarray], dict]:
    """→ ({人名: (N,192)}, manifest)。没有版本时返回 ({}, {}) 让调用方回落旧路径。"""
    v = current_version()
    if not v:
        return {}, {}
    vdir = os.path.join(PROTOS_DIR, v)
    man = read_manifest(v)
    out = {}
    for name in man.get("people", {}):
        p = os.path.join(vdir, f"{name}.npy")
        if os.path.exists(p):
            out[name] = _norm(np.load(p).astype(np.float32))
    return out, man


def load_with_prov(version: str | None = None):
    """→ ({人: 矩阵}, {人: 每窗所属录音数组}, manifest)。

    prov 让打分层能剔除「来源就是本录音」的窗（口径 §24）。没有它，一段录音
    会用自己的注册窗给自己作证：面试那次 0.569 / ρ 0.994 / 四条静默条件全过。
    """
    v = version or current_version()
    if not v:
        return {}, {}, {}
    protos, man = load_current() if version is None else ({}, read_manifest(v))
    if version is not None:
        vdir = os.path.join(PROTOS_DIR, v)
        for name in man.get("people", {}):
            p = os.path.join(vdir, f"{name}.npy")
            if os.path.exists(p):
                protos[name] = _norm(np.load(p).astype(np.float32))
    srcs = list_sources()
    prov = {}
    for name, info in (man.get("people") or {}).items():
        ids, tags = [], []
        for sid in info.get("sources", []):
            n = int(srcs.get(sid, {}).get("n_win", 0))
            if not n:
                continue
            ids.append(sid)
            tags += [srcs.get(sid, {}).get("recording") or sid] * n
        if ids and name in protos and len(tags) == len(protos[name]):
            prov[name] = np.array(tags, dtype=object)
    return protos, prov, man


def eligible(man: dict, *, for_silent: bool) -> set[str]:
    """谁能参与静默署名 / 谁能参与 top1 评选。

    for_silent=True ：窗数够（N_PROTO_MIN）、独立录音够（N_SOURCE_MIN）、跨录音自检通过。
    for_silent=False：只要窗数不至于少到「排名没有意义」（N_PROTO_EXCLUDE）。
                      低窗原型不参与自动 top1，但**仍留在人工候选列表里**，
                      否则「再确认一次补足第二来源」的闭环就断了。

    两个门槛是不同的数（50 / 100），别合成一个：合并会让「样本量够但不够干净」的人
    既进不了自动判定、也解释不清为什么明明能认出来却不认。
    """
    ok = set()
    floor = getattr(config, "N_PROTO_EXCLUDE", 50)
    need = getattr(config, "N_PROTO_MIN", 100)
    for name, p in (man.get("people") or {}).items():
        if for_silent:
            if p.get("enough_windows") and p.get("n_win", 0) >= need \
                    and p.get("enough_sources") and p.get("verified"):
                ok.add(name)
        elif p.get("n_win", 0) >= floor:
            ok.add(name)
    return ok


def status() -> str:
    """人可读的自检报告。"""
    man = read_manifest()
    if not man:
        return "还没有任何版本（先跑 build_version）"
    lines = [f"版本 {man['version']}  {man['scorer_params']}", "",
             f"{'人':<8}{'窗数':>6}{'来源':>5}{'窗数够':>7}{'来源够':>7}{'跨源自检':>9}   静默资格"]
    for name, p in sorted(man["people"].items(), key=lambda kv: -kv[1]["n_win"]):
        lines.append(f"{name:<8}{p['n_win']:>6}{p['n_sources']:>5}"
                     f"{'✓' if p['enough_windows'] else '✗':>7}"
                     f"{'✓' if p['enough_sources'] else '✗':>7}"
                     f"{'✓' if p['verified'] else '✗':>9}"
                     f"   {'可' if name in eligible(man, for_silent=True) else '否'}")
    return "\n".join(lines)


# ── 迁移 ───────────────────────────────────────────────────────────────
def import_legacy(npz_path: str) -> list[str]:
    """把旧的单文件 protos.npz 拆成来源级存储。

    每个人只能落成**一个**来源（npz 里没有来源信息），所以导入后全部
    n_sources=1、verified=False、不参与静默署名。这是诚实的结果：
    旧库确实无法证明跨信道一致性。要恢复静默资格，只能补第二段录音。
    """
    if not npz_path or not os.path.exists(npz_path):
        return []
    z = np.load(npz_path)
    made = []
    for name in z.files:
        mat = _norm(np.asarray(z[name], dtype=np.float32))
        if len(mat) == 0:
            continue
        made.append(save_source(name, mat, wav=f"[legacy]{npz_path}"))
    return made


# ── 自测（合成向量，不碰模型）──────────────────────────────────────────
def _selftest(tmp: str) -> None:
    global EMB_DIR, PROTOS_DIR, CURRENT
    EMB_DIR = os.path.join(tmp, "emb")
    PROTOS_DIR = os.path.join(tmp, "protos")
    CURRENT = os.path.join(PROTOS_DIR, "CURRENT")

    rng = np.random.default_rng(0)
    dim = 192
    basis = _norm(rng.normal(size=(3, dim)))

    def make(spk, n, noise=0.7, seed=0):
        r = np.random.default_rng(seed)
        return basis[spk] + r.normal(size=(n, dim)) * (noise / np.sqrt(dim))

    # A：两次不同录音，跨录音自检应通过
    sid1 = save_source("A", make(0, 120, seed=1), recording="rec1")
    sid2 = save_source("A", make(0, 120, seed=2), recording="rec2")
    # B：单来源，应 unverified
    save_source("B", make(1, 150, seed=3), recording="rec3")
    # C：窗数太少，应 enough_windows=False
    save_source("C", make(2, 20, seed=4), recording="rec4")
    # D：同一段录音存两次 ≠ 两份独立证据，必须仍然 unverified
    save_source("D", make(0, 100, seed=5), recording="rec5")
    save_source("D", make(0, 100, seed=6), recording="rec5")

    v = build_version()
    man = read_manifest(v)
    pA, pB, pC = man["people"]["A"], man["people"]["B"], man["people"]["C"]
    assert pA["n_win"] == 240 and pA["n_sources"] == 2
    assert pA["verified"], pA["verify"]
    assert not pB["verified"] and pB["verify"]["why"] == "single_source"
    assert not pC["enough_windows"]
    assert man["people"]["D"]["n_sources"] == 1 and not man["people"]["D"]["verified"], \
        "同一段录音存两次不能算两份独立证据"

    silent = eligible(man, for_silent=True)
    top1 = eligible(man, for_silent=False)
    assert "A" in silent, silent
    assert "B" in top1 and "C" not in top1      # 低窗 C 被剔除，但 B 仍可参与 top1

    # 撤销一个来源后，A 掉回单来源 → 失去静默资格
    assert delete_source(sid2)
    v2 = build_version()
    man2 = read_manifest(v2)
    assert man2["people"]["A"]["n_sources"] == 1
    assert not man2["people"]["A"]["verified"]
    assert "A" not in eligible(man2, for_silent=True)

    protos, _ = load_current()
    assert set(protos) == {"A", "B", "C", "D"}
    assert os.path.exists(os.path.join(PROTOS_DIR, v, "manifest.json")), "旧版本必须保留"
    print("protolib self-test ok")
    print(status())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        # 不用 TemporaryDirectory：TMPDIR 不可用时它会退到当前目录建随机目录，
        # 而这里的自测本来就应该在固定位置跑完就留着。
        td = os.path.join(config.VP_ROOT, "_selftest")
        os.makedirs(td, exist_ok=True)
        _selftest(td)
    elif len(sys.argv) > 1 and sys.argv[1] == "--rebuild":
        print("version:", build_version(t_auto=config.T_AUTO))
        print(status())
    else:
        print(status())
