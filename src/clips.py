#!/usr/bin/env python3
"""给每个簇预切一段 15 秒试听音频。

为什么要「跨段拼接」而不是「切最长的一段」：一段 147 秒的簇里常常混着抢话，
单段播出来会让人认错人。从 3 个分散段落各取 5 秒拼接，听到的是「这个声音在
不同时候的样子」，认人速度快得多。

为什么要预切而不是点播时现切：确认页上「点一下就出声」是唯一判据（界面禁数字），
现切要 200ms 以上，还会因为浏览器 Range 请求重复解码，体验上就是「点了没反应」。
"""
from __future__ import annotations

import os

import numpy as np
import soundfile as sf

import config


def _energy(audio, sr, st, en):
    seg = audio[int(st * sr):int(en * sr)]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0


def pick_spans(audio, sr, spans, k=None, sec=None, gap=None):
    """挑出最多 k 段、每段截到 sec 秒的片段。返回 [(st, en)]，按时间排序。

    两种情形都要照顾：
      · 一个簇有很多分散的短段 → 挑能量最高的几段
      · 一个簇就是一整段 145 秒（一对一访谈里最常见）→ 从这一段里等间隔切 3 块
    只认「分散的多段」会让最需要试听的那类录音恰好没有试听音频。
    """
    k = config.CLIP_N if k is None else k
    sec = config.CLIP_SEC if sec is None else sec
    gap = config.CLIP_GAP if gap is None else gap
    cand = [( _energy(audio, sr, st, en), st, en) for st, en in spans if en - st >= 1.0]
    if not cand:
        return []
    cand.sort(key=lambda x: -x[0])
    picked = list(cand[:k])

    if len(picked) < k:                       # 段数不够：从最长的那几段里再切几块
        for _, st, en in sorted(cand, key=lambda x: -(x[2] - x[1])):
            need = k - len(picked)
            if need <= 0 or en - st < sec * 2:
                continue
            for i in range(need):
                mid = st + (en - st) * (i + 1) / (need + 1)
                picked.append((0.0, max(st, mid - sec / 2), min(en, mid + sec / 2)))

    out = []
    for _, st, en in sorted(picked[-k:], key=lambda x: x[1]):
        if en - st > sec:
            mid = (st + en) / 2
            st, en = max(st, mid - sec / 2), min(en, mid + sec / 2)
        out.append((st, en))
    # 去掉重叠（长段切出来的块可能与已选段重叠）
    out.sort()
    dedup = []
    for st, en in out:
        if dedup and st < dedup[-1][1] - 0.2:
            continue
        dedup.append((st, en))
    return dedup


def build_clip(audio, sr, spans, out_path, *, k=None, sec=None, gap=None):
    gap = config.CLIP_GAP if gap is None else gap
    picked = pick_spans(audio, sr, spans, k=k, sec=sec, gap=gap)
    if not picked:
        return None
    total = sum(en - st for st, en in picked)
    if total < 4.0 or len(picked) < 2:
        # 太短的簇不生成音频：卡上写「这段只有 N 秒，听不出人」比给个坏音频好
        return None
    parts = []
    silence = np.zeros(int(gap * sr), dtype=np.float32)
    for i, (st, en) in enumerate(picked):
        if i:
            parts.append(silence)
        parts.append(audio[int(st * sr):int(en * sr)])
    clip = np.concatenate(parts)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, clip, sr)
    return out_path


def build_for_recording(audio, sr, segs, rid):
    """→ {cluster_id: 相对 URL 或 None}。文件落 CLIP_CACHE_DIR，不进 OneDrive。"""
    out = {}
    by = {}
    for s in segs:
        by.setdefault(int(s["speaker"]), []).append((s["start"], s["end"]))
    d = os.path.join(config.CLIP_CACHE_DIR, rid)
    for cid, spans in by.items():
        p = os.path.join(d, f"c{cid}.wav")
        out[cid] = p if build_clip(audio, sr, spans, p) else None
    return out
