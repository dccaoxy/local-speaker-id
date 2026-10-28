#!/usr/bin/env python3
"""一段录音的完整流水线：转码 → 分离 → 合并 → 打分 → 切片 → 出稿。

分两路音频是刻意的（实测结论，别改成一路）：
  分离跑 8k 带限音频——带限抹掉设备差异，聚类明显更稳
  匹配跑原始 16k 音频——带限会把「同一个人」和「不同人」的分数一起抬高，区分度没了
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import sherpa_onnx
import soundfile as sf

import asr as asr_mod
import clips
import config
import identify
import render
import store
import winmatch


def log(*a):
    print("[pipeline]", *a, flush=True)


# ── 分离 ────────────────────────────────────────────────────────────────
def diarize(audio, sr, clusters=-1):
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=config.SEG_MODEL),
            num_threads=config.THREADS),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=config.EMB_MODEL, num_threads=config.THREADS),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=clusters, threshold=0.5),
        min_duration_on=0.3, min_duration_off=0.5)
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    return [{"start": r.start, "end": r.end, "speaker": int(r.speaker)}
            for r in sd.process(audio).sort_by_start_time()]


def merge_clusters(audio, sr, segs, ext, cos=None):
    """把质心余弦过高的簇并成一个。返回 (segs, 合并了多少对)。

    自动聚类（num_clusters=-1）经常把同一个人切成好几个簇：换语气、走动、
    换麦克风都会。不合并的话同一个人会出 2~3 张卡，用户会认为「它连这是同一个人
    都看不出来」。合并阈值故意取高（0.85），宁可不并也别把两个人并成一个。
    """
    cos = config.CLUSTER_MERGE_COS if cos is None else cos
    by = {}
    for s in segs:
        by.setdefault(s["speaker"], []).append((s["start"], s["end"]))
    if len(by) < 2:
        return segs, 0
    gate = winmatch.gate_db_of(audio, sr)
    cents, ids = [], []
    for cid, spans in by.items():
        v = winmatch._vecs(ext, winmatch.windows_of(audio, sr, spans, gate_db=gate))
        if len(v) == 0:
            continue
        c = v.mean(axis=0)
        cents.append(c / max(np.linalg.norm(c), 1e-9))
        ids.append(cid)
    if len(ids) < 2:
        return segs, 0
    M = np.stack(cents)
    parent = {c: c for c in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merged = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if float(M[i] @ M[j]) >= cos:
                a, b = find(ids[i]), find(ids[j])
                if a != b:
                    parent[b] = a
                    merged += 1
    if not merged:
        return segs, 0
    out = []
    for s in segs:
        s2 = dict(s)
        s2["speaker"] = find(s["speaker"])
        out.append(s2)
    # 重编号成 0..n-1，保持稳定顺序（按首次出现）
    order = {}
    for s in out:
        order.setdefault(s["speaker"], len(order))
    for s in out:
        s["speaker"] = order[s["speaker"]]
    return out, merged


# ── 主流程 ──────────────────────────────────────────────────────────────
def process(path, rid, *, clusters=-1, progress=None, with_asr=True):
    """返回 rec dict（不含 status）。调用方负责落 state 与出稿。"""
    t0 = time.time()

    def step(name, pct):
        if progress:
            progress(name, pct)
        log(f"{rid} {name} {pct}%")

    step("convert", 5)
    keep = os.path.join(config.WORK_DIR, rid)
    os.makedirs(keep, exist_ok=True)
    wav_raw = identify.to_wav16k(path, out_dir=keep)
    # 留一份 16k 单声道在本地工作区：用户确认「新人」时要拿这段音频重新入库，
    # 没有它就得多转一次码（100 分钟录音十几秒），而且入库窗口立刻变长。
    try:
        a16, sr16 = sf.read(wav_raw, dtype="float32")
        if a16.ndim > 1:
            a16 = a16.mean(axis=1)
        sf.write(os.path.join(keep, "a16.wav"), a16, sr16)
    except Exception as e:
        log("保留 16k 音频失败（不影响识别）：", e)
    wav_diar = identify.bandlimit(wav_raw, out_dir=keep)
    audio, sr = sf.read(wav_diar, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio_m, _ = sf.read(wav_raw, dtype="float32")
    if audio_m.ndim > 1:
        audio_m = audio_m.mean(axis=1)
    dur = len(audio) / sr

    step("diarize", 15)
    segs = diarize(audio, sr, clusters)
    ext = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=config.EMB_MODEL,
                                                    num_threads=config.THREADS))
    segs, merged = merge_clusters(audio, sr, segs, ext)

    step("score", 55)
    spans_by = {}
    for s in segs:
        if s["end"] - s["start"] >= config.MIN_DUR:
            spans_by.setdefault(s["speaker"], []).append((s["start"], s["end"]))

    gate = winmatch.gate_db_of(audio_m, sr)
    protos, prov, man = ({}, {}, {})
    try:
        import protolib
        protos, prov, man = protolib.load_with_prov()
    except Exception as e:
        log("protolib 不可用，回落旧原型：", e)

    evs = []
    for cid, spans in sorted(spans_by.items()):
        ev = winmatch.score_cluster(audio_m, sr, spans, cluster_id=cid,
                                    dur_total=dur, protos=protos, prov=prov, man=man,
                                    gate_db=gate, ext=ext, recording=rid)
        ev.share = round(ev.duration_s / dur, 3) if dur else 0.0
        sp = max(spans, key=lambda x: x[1] - x[0])
        ev.longest_span = sp
        evs.append(ev)

    # 互斥：一个人只能占一个簇。强的留名，弱的降级为问。
    taken = set()
    for ev in sorted(evs, key=lambda e: -e.s1):
        if ev.name and ev.name in taken:
            ev.name, ev.ask, ev.reason = None, True, "low_margin"
        elif ev.name:
            taken.add(ev.name)

    n_clusters = len(evs)
    max_share = max((e.share for e in evs), default=0.0)
    for ev in evs:
        winmatch.decide_from_evidence(ev, dur, n_clusters, max_share)

    overlap = winmatch.is_overlap_of(segs, dur)
    warnings = []
    if n_clusters >= config.COLLAPSE_MIN_CLUSTERS and max_share > config.COLLAPSE_SHARE:
        warnings.append(f"这段录音有 {max_share * dur / 60:.0f} 分钟被归成了同一个人，"
                        "分离可能不完整，下面的判定都别太当真。")
    if overlap > config.OVERLAP_ALERT:
        warnings.append(f"有 {overlap:.0%} 的时长被多个说话人同时覆盖，重叠部分请回听。")

    step("clips", 75)
    clip_paths = clips.build_for_recording(audio_m, sr, segs, rid)

    step("asr", 85)
    asr_rows = []
    if with_asr:
        try:
            asr_rows = asr_mod.load_asr(asr_mod.find_asr_for(path))
        except Exception as e:
            log("转写稿解析失败：", e)
    samples = asr_mod.samples_by_cluster(segs, asr_rows) if asr_rows else []

    clusters_out = []
    for ev in sorted(evs, key=lambda e: -e.duration_s):
        api = winmatch.to_api(ev)
        api.update({
            "duration_s": ev.duration_s, "share": round(ev.share, 3),
            "eff_sec": ev.eff_sec, "ask": bool(ev.ask),
            "auto_name": ev.name, "top1": ev.top1 if ev.reason not in
            {"mixed", "noisy", "mixed_weak", "few_match", "low_rho"} else None,
            "samples": next((x["samples"] for x in samples
                             if x["cluster"] == ev.cluster_id), []),
            "clip": clip_paths.get(ev.cluster_id),
            # 台账字段：不进接口，但要能给标定用
            "_s": {"s1": ev.s1, "s2": ev.s2, "m": ev.m, "rho": ev.rho, "rho2": ev.rho2,
                   "n": ev.n_total, "win_pass": ev.win_pass_rate,
                   "proto_n": ev.proto_n, "source_n": ev.source_n,
                   "verified": ev.verified, "silent_ok": ev.silent_ok},
        })
        clusters_out.append(api)

    rec = {
        "rid": rid, "path": path,
        "title": os.path.splitext(os.path.basename(path))[0],
        "duration": round(dur, 2), "n_clusters": n_clusters,
        "merged": merged, "segs": segs, "clusters": clusters_out,
        "warnings": warnings, "overlap": overlap,
        "elapsed": round(time.time() - t0, 1),
        "md_path": os.path.splitext(path)[0] + ".说话人.md",
    }
    step("done", 95)
    return rec


def render_rec(rec):
    """按当前台账重渲染 md（改名 / 撤销后调用，不重跑分离）。"""
    verdicts = store.verdicts(rec["rid"])
    asr_rows = []
    try:
        asr_rows = asr_mod.load_asr(asr_mod.find_asr_for(rec["path"]))
    except Exception:
        pass
    samples = asr_mod.samples_by_cluster(rec["segs"], asr_rows) if asr_rows else []
    text = render.render(rec, verdicts, asr_rows=asr_rows, samples=samples,
                         warnings=rec.get("warnings"))
    return render.write_md(rec["md_path"], text)
