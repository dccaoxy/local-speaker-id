#!/usr/bin/env python3
"""滑窗声纹打分器。

核心思路（与「整段取均值」的老做法相对）：
  · 2 秒窗 / 1 秒步长切窗，按能量过滤静音窗
  · 每个人注册成**一组窗口向量**，不做平均——长段里混了别人也不会污染
  · 查询窗与注册窗取最大余弦（最近邻）
  · 一个簇的得分 = 所有查询窗的**中位数**（中位数比 top-K 稳，不会被少数撞上的窗带跑）
  · 判定：中位数 ≥ THRESHOLD，且比次优选手高 MARGIN

实测间隔（同人 vs 最近异人）：
  整段取均值 → +0.019（等于没有）
  滑窗 + 中位数 → +0.111

注册信息来自 speakers.json（参见仓库根的 speakers.example.json）。
"""
import json
import os

import numpy as np
import sherpa_onnx
import soundfile as sf

MODEL_DIR = os.environ.get("VP_MODEL_DIR", os.path.expanduser("~/.voiceprint/models"))
EMB = os.path.join(MODEL_DIR, "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")
DB = os.environ.get("VP_DB", os.path.expanduser("~/.voiceprint/db"))
SPEAKERS = os.environ.get("VP_SPEAKERS", os.path.expanduser("~/.voiceprint/speakers.json"))
CACHE = os.path.join(DB, "protos.npz")

WIN, HOP = 2.0, 1.0
THRESHOLD = 0.40
MARGIN = 0.05


def load_sources(path=None):
    """speakers.json 结构:
    { "张三": [ {"wav": "某人.wav", "segs": "某人.segs.json", "cluster": 0} ] }
    一个人可以有多条来源（不同录音/不同设备），会自动合并成一组原型窗。
    """
    p = path or SPEAKERS
    if not os.path.exists(p):
        return {}
    return json.load(open(p, encoding="utf-8"))


def _em(ext, chunk):
    s = ext.create_stream()
    s.accept_waveform(16000, chunk)
    s.input_finished()
    if not ext.is_ready(s):
        return None
    v = np.array(ext.compute(s), dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def windows_of(audio, sr, spans, cap=1200, min_rms_pct=25):
    chunks = []
    for st, en in spans:
        t = st
        while t + WIN <= en:
            c = audio[int(t * sr): int((t + WIN) * sr)]
            if len(c) == int(WIN * sr):
                chunks.append(c)
            t += HOP
    if not chunks:
        return []
    rms = np.array([float(np.sqrt(np.mean(c ** 2))) for c in chunks])
    thr = np.percentile(rms, min_rms_pct)
    return [c for c, r in zip(chunks, rms) if r >= thr][:cap]


def _vecs(ext, chunks):
    vs = [v for v in (_em(ext, c) for c in chunks) if v is not None]
    return np.stack(vs) if vs else np.zeros((0, 192), np.float32)


def build_protos(force=False):
    """构建/加载各人的滑窗原型（npz 缓存）。"""
    if os.path.exists(CACHE) and not force:
        try:
            z = np.load(CACHE)
            return {k: z[k] for k in z.files}
        except Exception:
            pass

    src = load_sources()
    if not src:
        return {}

    ext = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB, num_threads=4))
    out = {}
    for name, parts in src.items():
        vecs = []
        for part in parts:
            wav, segs_path = part["wav"], part["segs"]
            cluster = part.get("cluster", 0)
            if not (os.path.exists(wav) and os.path.exists(segs_path)):
                continue
            a, sr = sf.read(wav, dtype="float32")
            if a.ndim > 1:
                a = a.mean(axis=1)
            segs = json.load(open(segs_path, encoding="utf-8"))["segments"]
            spans = [(s["start"], s["end"]) for s in segs if s["speaker"] == cluster]
            v = _vecs(ext, windows_of(a, sr, spans))
            if len(v):
                vecs.append(v)
        if vecs:
            out[name] = np.concatenate(vecs, axis=0)

    os.makedirs(DB, exist_ok=True)
    np.savez_compressed(CACHE, **out)
    return out


def score_spans(audio, sr, spans):
    """一个簇（若干时间区间）对每个人的中位数得分。"""
    ext = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB, num_threads=4))
    q = _vecs(ext, windows_of(audio, sr, spans))
    if len(q) == 0:
        return {}, 0
    protos = build_protos()
    scores = {}
    for name, p in protos.items():
        if len(p) == 0:
            continue
        best = (q @ p.T).max(axis=1)
        scores[name] = round(float(np.median(best)), 3)
    return scores, len(q)


def decide(scores, threshold=THRESHOLD, margin=MARGIN):
    if not scores:
        return "未知"
    order = sorted(scores.items(), key=lambda kv: -kv[1])
    best, s1 = order[0]
    s2 = order[1][1] if len(order) > 1 else 0.0
    return best if (s1 >= threshold and s1 - s2 >= margin) else "未知"


if __name__ == "__main__":
    protos = build_protos(force=True)
    print("原型窗数:", {k: len(v) for k, v in protos.items()})
    print("缓存:", CACHE)
