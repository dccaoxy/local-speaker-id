#!/usr/bin/env python3
"""解析伴侣转写稿，给每个簇抽几句样例。

这是 MVP 必做项，不是增强项：确认卡片如果只有音频，认一个人要 15 秒以上，
有文字对照 5 秒。4 簇的录音，20 秒注意力会变成 3 分钟，直接冲破
「每 10 小时人工点击数 ≤4」的主指标。

转写稿是**外部产物**（Magic 录音笔 / AudioTranscribe），本模块只读不产。
找不到就在返回空列表，卡片降级为纯音频——不能报错，更不能阻塞流水线。
"""
import json
import os
import re

import config

# 三种格式（与 identify.py 原有实现保持一致，不要重写一套）
FMT_MS = re.compile(r"^-\s*\[(\d+):(\d+\.\d+),(\d+):(\d+\.\d+),(\d+)\]\s*(.*)$")
FMT_HM = re.compile(r"^-\s*(\d+):(\d{2})\s*讲话人(\d+)\s*(.*)$")


def load_asr(path):
    """→ [(start, end|None, text), ...]，按时间升序。解析不了返回 []。"""
    if not path or not os.path.exists(path):
        return []
    rows = []
    if path.lower().endswith(".json"):
        try:
            d = json.load(open(path, encoding="utf-8"))
            for s in d.get("segments", []):
                rows.append((float(s["start"]), float(s["end"]), str(s["text"])))
        except (ValueError, KeyError, TypeError):
            return []
        return sorted(rows, key=lambda r: r[0])

    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    for line in lines:
        s = line.strip()
        m = FMT_MS.match(s)
        if m:
            rows.append((int(m.group(1)) * 60 + float(m.group(2)),
                         int(m.group(3)) * 60 + float(m.group(4)),
                         m.group(6).strip()))
            continue
        m = FMT_HM.match(s)
        if m:
            rows.append((int(m.group(1)) * 60 + int(m.group(2)), None, m.group(4).strip()))
    return sorted(rows, key=lambda r: r[0])


def find_asr_for(audio_path):
    """同名伴生稿：优先 .md（Magic 导出），其次 .json（毫秒分段，粒度最细）。"""
    if not audio_path:
        return None
    base, _ = os.path.splitext(audio_path)
    for ext in (".md", ".json"):
        p = base + ext
        if os.path.exists(p):
            return p
    return None


def samples_for(spans, rows, k=None, max_chars=None):
    """挑 k 句：落在该簇时间范围内的句子 → 去重 → 按字数降序 → 截断。

    按字数降序是有意的：短句（"嗯"、"对"）对认人没有价值，
    长句里才有称谓、口播习惯、语气词这些识别线索。
    """
    k = config.SAMPLE_K if k is None else k
    max_chars = config.SAMPLE_MAX_CHARS if max_chars is None else max_chars
    if not spans or not rows:
        return []
    seen, picked = set(), []
    for st, en, txt in rows:
        if not txt:
            continue
        hit = any(en0 - 0.3 <= st <= en1 + 0.3 for en0, en1 in spans)
        if not hit:
            continue
        t = " ".join(txt.split())
        if len(t) < 2 or t in seen:
            continue
        seen.add(t)
        picked.append(t)
    picked.sort(key=len, reverse=True)
    out = [t if len(t) <= max_chars else t[:max_chars] + "…" for t in picked[:k]]
    return out


def assign_rows(segs, rows):
    """把每句转写文本归给**唯一**一个簇：覆盖该句时长最多的那个。

    早期版本按「起点落在任一簇区间内」判，于是重叠段里同一句会同时进两个簇的
    例句列表——两条卡片的例句一模一样，用户一眼就看出这是糊的。
    """
    # 会议 App 导出的格式只有起始时间（end=None）。归属必须用「时长」，
    # 所以先补一个估计值：到下一句开始为止（最多 15 秒）。
    rows = list(rows or [])
    norm = []
    for i, (st, en, txt) in enumerate(rows):
        if en is None:
            nxt = rows[i + 1][0] if i + 1 < len(rows) else st + 3.0
            en = min(max(nxt, st + 0.5), st + 15.0)
        norm.append((st, en, txt))
    out = {}
    for st, en, txt in norm:
        best, best_cov = None, 0.0
        for s in segs or []:
            cov = min(en, s["end"]) - max(st, s["start"])
            if cov > best_cov:
                best, best_cov = int(s["speaker"]), cov
        if best is None:
            # 没有覆盖：退化成「离哪个簇最近」
            near = min(segs or [{"speaker": 0, "start": 0, "end": 0}],
                       key=lambda s: min(abs(st - s["start"]), abs(st - s["end"])))
            if min(abs(st - near["start"]), abs(st - near["end"])) > 1.5:
                continue
            best = int(near["speaker"])
        out.setdefault(best, []).append((st, en, txt))
    return out


def samples_by_cluster(segs, rows, k=None):
    """→ [{cluster, samples:[...]}]，按簇号升序。给渲染层用，不进 decisions。"""
    by = assign_rows(segs, rows)
    by_spans = {}
    for s in segs or []:
        by_spans.setdefault(int(s["speaker"]), []).append((s["start"], s["end"]))
    return [{"cluster": spk, "samples": samples_for(spans, by.get(spk, []), k)}
            for spk, spans in sorted(by_spans.items())]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("用法: python asr.py <转写稿.md|json>")
    r = load_asr(sys.argv[1])
    print(f"{len(r)} 句")
    for st, en, txt in r[:5]:
        print(f"  {st:8.2f} {en if en is not None else '-':>8} {txt[:40]}")
