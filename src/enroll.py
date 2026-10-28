#!/usr/bin/env python3
"""把「某段录音里的某个簇」入库成某人的一条声纹来源。

为什么入库前要过纯度检查：
  实测事故——张彤的 42 个原型窗里混了多个人的声音，于是他的原型对同一段录音里
  好几种声音都给 0.5~0.6。评审的结论是「这不是信道问题，是注册片段本身是混合物」。
  每注册一个混合簇，就等于往库里加一个吸引子，污染面比不注册更大。
  所以「入库」这件事必须比「识别」更严格：宁可不入库，也不要入脏样本。

两道门：
  1) 时间切分一致性：把窗按时间切成前后两半，算跨半区余弦中位数。
     前后不像同一个人 → 拒收。这一条专门抓「一段录音里换了人 / 抢话」。
  2) 窗级筛选：去掉离群窗（与簇中心的余弦 < 中位数 - 0.15），
     避免少数撞上的窗把原型带偏。
"""
from __future__ import annotations

import json
import os

import numpy as np
import sherpa_onnx
import soundfile as sf

import config
import protolib
import winmatch


def _norm_rows(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def load_wav16k(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"需要 16k 单声道 wav，得到 {sr}：{path}")
    return a


def purity(vecs: np.ndarray) -> dict:
    """簇内一致性。时间切分比整体均值更能抓「前半是A、后半混进B」。

    门槛用 PURITY_MIN，不是 RHO_MIN——两者是不同的量：
      RHO_MIN    = 有多少窗指向同一个人（查询侧）
      PURITY_MIN = 前后两半互相比的中位余弦（入库侧）
    混用会让所有正常注册被拒（实测三个真人簇 0.584 / 0.694 / 0.641 全在 0.70 以下）。
    """
    if len(vecs) < 4:
        return {"time_split": 0.0, "centroid": 0.0, "ok": False, "why": "too_few"}
    v = _norm_rows(vecs)
    half = len(v) // 2
    a, b = v[:half], v[half:]
    cross = float(np.median((a @ b.T).max(axis=1)))
    c = v.mean(axis=0)
    c /= max(np.linalg.norm(c), 1e-9)
    cent = float(np.median(v @ c))
    ok = cross >= config.PURITY_MIN
    return {"time_split": round(cross, 3), "centroid": round(cent, 3),
            "ok": ok, "why": "ok" if ok else "mixed"}


def filter_windows(vecs: np.ndarray, drop: float = 0.15) -> np.ndarray:
    """去掉离群窗：与簇中心余弦低于 (中位数 - drop) 的都当别人的话。"""
    if len(vecs) < 8:
        return vecs
    v = _norm_rows(vecs)
    c = v.mean(axis=0)
    c /= max(np.linalg.norm(c), 1e-9)
    s = v @ c
    keep = s >= (float(np.median(s)) - drop)
    return vecs[keep] if keep.sum() >= 4 else vecs


def embed_spans(audio, sr, spans, ext=None):
    gate = winmatch.gate_db_of(audio, sr)
    wins = winmatch.windows_of(audio, sr, spans, gate_db=gate)
    if not wins:
        return np.zeros((0, 192), np.float32)
    ext = ext or sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=config.EMB_MODEL,
                                                    num_threads=config.THREADS))
    return winmatch._vecs(ext, wins)


def spans_from_segs(segs_path, cluster):
    segs = json.load(open(segs_path, encoding="utf-8"))["segments"]
    return [(s["start"], s["end"]) for s in segs if int(s["speaker"]) == int(cluster)]


def enroll(name, wav, segs_path, cluster, *, recording="", strict=True,
           rebuild=True, ext=None):
    """入库一个人。返回 (source_id|None, report)。"""
    audio = load_wav16k(wav)
    spans = spans_from_segs(segs_path, cluster)
    rep = {"name": name, "cluster": cluster, "recording": recording, "n_raw": 0}
    if not spans:
        rep["error"] = "该簇没有任何分段"
        return None, rep

    vecs = embed_spans(audio, 16_000, spans, ext=ext)
    rep["n_raw"] = int(len(vecs))
    if len(vecs) < config.N_MIN_ASK:
        rep["error"] = f"有效窗 {len(vecs)} < {config.N_MIN_ASK}，样本不足，拒绝入库"
        return None, rep

    pur = purity(vecs)
    rep["purity"] = pur
    kept = filter_windows(vecs)
    rep["n_kept"] = int(len(kept))
    rep["keep_ratio"] = round(len(kept) / max(1, len(vecs)), 3)

    if strict and not pur["ok"]:
        rep["error"] = (f"注册片段本身像混合物（前后一致度 {pur['time_split']} < "
                        f"{config.RHO_MIN}），拒绝入库——先切干净再注册")
        return None, rep

    sid = protolib.save_source(name, kept, wav=wav, segs=segs_path, cluster=cluster,
                              gate_db=winmatch.gate_db_of(audio, 16_000),
                              recording=recording)
    rep["source_id"] = sid
    if rebuild:
        rep["version"] = protolib.build_version()
    return sid, rep
