#!/usr/bin/env python3
"""上传一段录音 → 自动分离说话人 → 比对声纹库 → 输出带人名的逐句稿。

用法:
  python identify.py <音频> [--clusters N] [--asr 转写.md] [--out 输出.md]

--clusters 不给时按自动判断（人多时建议显式指定）。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
import sherpa_onnx
import soundfile as sf

MODEL_DIR = os.environ.get("VP_MODEL_DIR", os.path.expanduser("~/.voiceprint/models"))
SEG_MODEL = f"{MODEL_DIR}/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMB_MODEL = f"{MODEL_DIR}/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
DB_DIR = os.environ.get("VP_DB", os.environ.get("VP_DB", os.path.expanduser("~/.voiceprint/db")))

# 匹配策略：绝对相似度 + 与次优候选的差距，两个条件都满足才敢落名字
THRESHOLD = 0.70
MARGIN = 0.08
MIN_DUR = 1.5          # 短于此的片段不提取声纹
UNKNOWN = "未知"

LINE = re.compile(r"^-\s*\[(\d+):(\d+\.\d+),(\d+):(\d+\.\d+),(\d+)\]\s*(.*)$")
LINE2 = re.compile(r"^-\s*(\d+):(\d{2})\s*讲话人(\d+)\s*(.*)$")


def to_wav16k(src):
    """任意音频 → 16k 单声道 wav（用 macOS 自带 afconvert，无需 ffmpeg）。"""
    out = os.path.join(tempfile.mkdtemp(prefix="vp_"), "a.wav")
    if src.lower().endswith(".wav"):
        d, sr = sf.read(src, dtype="float32")
        if sr == 16000 and d.ndim == 1:
            return src
    r = subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", src, out],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        sys.exit(f"音频转换失败：{r.stderr.strip()}")
    return out


def bandlimit(wav16k):
    """把音频压到 8k 电话带宽再升回 16k，消除不同录音设备/码率造成的信道差异。

    实测：同一人跨设备相似度从 0.156 提升到 0.572。不带宽归一时声纹基本不可用。
    """
    d = os.path.dirname(wav16k)
    b8 = os.path.join(d, "bl8.wav")
    b16 = os.path.join(d, "bl16.wav")
    for args in (["-f", "WAVE", "-d", "LEI16@8000", "-c", "1", wav16k, b8],
                 ["-f", "WAVE", "-d", "LEI16@16000", "-c", "1", b8, b16]):
        if subprocess.run(["afconvert"] + args, capture_output=True).returncode != 0:
            return wav16k
    return b16


def load_db():
    idx_path = os.path.join(DB_DIR, "人物索引.json")
    if not os.path.exists(idx_path):
        sys.exit(f"声纹库不存在：{idx_path}")
    idx = json.load(open(idx_path, encoding="utf-8"))
    people = {}
    for name, info in idx["people"].items():
        fp = os.path.join(DB_DIR, info["file"])
        v = np.array(json.load(open(fp, encoding="utf-8"))["centroid"], dtype=np.float32)
        people[name] = v / np.linalg.norm(v)
    return people


def load_asr(path):
    """兼容三种转写稿：
      1) Magic 导出的 json（millisecond 分段，粒度最细，推荐）
      2) - [0:6.160,0:10.935,0] 文字            （本机 AudioTranscribe 输出）
      3) - 11:10 讲话人1 文字                    （会议 App 自动转写）
    """
    if path.lower().endswith(".json"):
        d = json.load(open(path, encoding="utf-8"))
        if "segments" in d:
            return [(s["start"], s["end"], s["text"]) for s in d["segments"]]
    rows = []
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        m = LINE.match(s)
        if m:
            rows.append((int(m.group(1)) * 60 + float(m.group(2)),
                         int(m.group(3)) * 60 + float(m.group(4)),
                         m.group(6).strip()))
            continue
        m = LINE2.match(s)
        if m:
            rows.append((int(m.group(1)) * 60 + int(m.group(2)), None, m.group(4).strip()))
    return rows


def embed(extractor, samples):
    stream = extractor.create_stream()
    stream.accept_waveform(16000, samples)
    stream.input_finished()
    if not extractor.is_ready(stream):
        return None
    v = np.array(extractor.compute(stream), dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def assign(cents, names, dbmat, threshold, margin, segs=None):
    """把聚簇一对一分配给库里的人。

    两条约束：
      1) 绝对阈值 threshold（低于就判未知）
      2) 一次录音里一个人只能占一个簇（互斥）。实测发现两个簇会同时算出同一个人，
         其中分低的那次是误判，所以按分数从高到低贪心分配，被占用的候选人不再考虑。
    """
    cands = []
    for spk, c in cents.items():
        sims = dbmat @ c
        order = np.argsort(-sims)
        for rank, idx in enumerate(order[:2]):
            if rank == 1 and float(sims[idx]) < threshold:
                continue
            cands.append((float(sims[idx]), spk, names[idx], rank,
                          float(sims[order[0]]), float(sims[order[1]]) if len(names) > 1 else 0.0))
    cands.sort(reverse=True)

    mapping, taken, info = {}, set(), {}
    for sim, spk, person, rank, best_sim, second in cands:
        if spk in mapping:
            continue
        if person in taken or sim < threshold or best_sim - second < margin:
            continue
        mapping[spk] = person
        taken.add(person)
        info[spk] = (person, best_sim, second)

    for spk in cents:
        mapping.setdefault(spk, UNKNOWN)

    report = []
    for spk, c in cents.items():
        sims = dbmat @ c
        order = np.argsort(-sims)
        who = mapping[spk]
        report.append({"cluster": spk, "assigned": who,
                       "best": names[order[0]], "sim": round(float(sims[order[0]]), 3),
                       "second": round(float(sims[order[1]]) if len(names) > 1 else 0.0, 3),
                       "seconds": round(sum(s["end"] - s["start"] for s in segs
                                            if s["speaker"] == spk), 1) if segs else 0.0})
    return mapping, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--clusters", type=int, default=-1)
    ap.add_argument("--asr", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--bandlimit", action="store_true",
                    help="匹配阶段也用 8k 带宽归一（实测会抹掉区分度，默认关闭）")
    ap.add_argument("--scorer", choices=["window", "centroid"], default="window",
                    help="window=滑窗最近邻+中位数（实测间隔 0.111）；centroid=整段均值（旧法，间隔 0.019）")
    ap.add_argument("--names", default=None,
                    help='手动点名，跳过声纹匹配。格式 "0=金雪竹,1=曹晓雨,2=候选人"')
    ap.add_argument("--no-diar-bandlimit", action="store_true",
                    help="分离阶段不做带宽归一（默认做：带限能显著改善聚类，实测簇数从 1 大 1 小变成 2 均衡簇）")
    a = ap.parse_args()

    # 分离与匹配用不同预处理：分离跑带限音频（聚类更稳），匹配用原始音频（区分度更高）
    wav_raw = to_wav16k(a.audio)
    wav_diar = wav_raw if a.no_diar_bandlimit else bandlimit(wav_raw)
    wav_match = bandlimit(wav_raw) if a.bandlimit else wav_raw

    audio, sr = sf.read(wav_diar, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio_m, _ = sf.read(wav_match, dtype="float32")
    if audio_m.ndim > 1:
        audio_m = audio_m.mean(axis=1)
    print(f"音频 {len(audio)/sr:.0f}s，开始分离…", file=sys.stderr)

    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=SEG_MODEL),
            num_threads=4),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB_MODEL, num_threads=4),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=a.clusters, threshold=0.5),
        min_duration_on=0.3, min_duration_off=0.5)
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    segs = [{"start": r.start, "end": r.end, "speaker": int(r.speaker)}
            for r in sd.process(audio).sort_by_start_time()]
    segs_out = (a.out or os.path.join(os.path.dirname(os.path.abspath(a.audio)),
                                      os.path.splitext(os.path.basename(a.audio))[0] + ".说话人.md"))
    segs_json = os.path.splitext(segs_out)[0] + ".segs.json"
    json.dump({"segments": segs}, open(segs_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("分段结果 →", segs_json, file=sys.stderr)

    ext = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB_MODEL, num_threads=4))
    per = defaultdict(list)
    for s in segs:
        if s["end"] - s["start"] < MIN_DUR:
            continue
        v = embed(ext, audio_m[int(s["start"] * 16000): int(s["end"] * 16000)])
        if v is not None:
            per[s["speaker"]].append(v)

    cents = {}
    for spk, vecs in per.items():
        c = np.mean(vecs, axis=0)
        cents[spk] = c / np.linalg.norm(c)

    if a.names:
        # 手动点名模式：不做声纹匹配，完全按给定的簇→姓名映射
        manual = {}
        for pair in a.names.replace("，", ",").split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                manual[int(k.strip())] = v.strip()
        mapping = {spk: manual.get(spk, UNKNOWN) for spk in cents}
        report = [{"cluster": spk, "assigned": mapping[spk], "best": "-", "sim": 0.0, "second": 0.0,
                   "seconds": round(sum(x["end"] - x["start"] for x in segs if x["speaker"] == spk), 1)}
                  for spk in cents]
        names, note = [], "手动点名"
    elif a.scorer == "window":
        import winmatch
        protos = winmatch.build_protos()
        mapping, report, taken = {}, [], set()
        per_scores = {}
        for spk in cents:
            spans = [(x["start"], x["end"]) for x in segs if x["speaker"] == spk]
            sc, nwin = winmatch.score_spans(audio_m, 16000, spans)
            per_scores[spk] = sc
        # 互斥：按最高分从大到小贪心分配，一人只占一簇
        cand = sorted(((sc.get(nm, 0), spk, nm) for spk, sc in per_scores.items()
                       for nm in sc), reverse=True)
        for sim, spk, nm in cand:
            if spk in mapping or nm in taken:
                continue
            sc = per_scores[spk]
            second = max([v for k, v in sc.items() if k != nm], default=0.0)
            if sim >= winmatch.THRESHOLD and sim - second >= winmatch.MARGIN:
                mapping[spk] = nm
                taken.add(nm)
        for spk in cents:
            mapping.setdefault(spk, UNKNOWN)
            sc = per_scores[spk]
            best = max(sc.items(), key=lambda kv: kv[1]) if sc else ("-", 0.0)
            second = max([v for k, v in sc.items() if k != best[0]], default=0.0)
            report.append({"cluster": spk, "assigned": mapping[spk], "best": best[0],
                           "sim": best[1], "second": round(second, 3),
                           "seconds": round(sum(x["end"] - x["start"] for x in segs if x["speaker"] == spk), 1)})
        note = (f"滑窗打分（{len(protos)} 人，窗 {winmatch.WIN}s/步 {winmatch.HOP}s）｜"
                f"阈值 {winmatch.THRESHOLD}｜margin {winmatch.MARGIN}")
    else:
        db = load_db()
        names = list(db)
        dbmat = np.stack([db[n] for n in names]) if names else np.zeros((0, 192), np.float32)
        mapping, report = assign(cents, names, dbmat, a.threshold, MARGIN, segs)
        note = f"声纹库 {DB_DIR}（{len(names)} 人）｜阈值 {a.threshold}｜margin {MARGIN}"

    asr = load_asr(a.asr) if a.asr else []

    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.audio)),
                                os.path.splitext(os.path.basename(a.audio))[0] + ".说话人.md")
    lines = [f"# 说话人识别结果：{os.path.basename(a.audio)}", "",
             f"> {note}",
             "> 「未知」表示相似度不足或与次优候选差距太小，需人工确认。", "",
             "## 聚簇判定", "", "| 聚簇 | 判定 | 最高相似度 | 次优 | 发言时长 |", "|---|---|---|---|---|"]
    for r in sorted(report, key=lambda x: -x["seconds"]):
        if r["best"] == "-":
            lines.append(f"| {r['cluster']} | {r['assigned']} | 手动 | - | {r['seconds']}s |")
        else:
            lines.append(f"| {r['cluster']} | {r['assigned']} | {r['best']}={r['sim']} | {r['second']:.3f} | {r['seconds']}s |")

    if asr:
        lines += ["", "## 逐句（按声纹命名）", ""]
        prev, prev_t = None, -99.0
        for st, en, txt in asr:
            if not txt:
                continue
            # 同一句被上游转写重复输出时去重
            if txt == prev and st - prev_t < 3:
                continue
            prev, prev_t = txt, st

            # 按「这句话的起始时间」落到哪一段分离结果里来定人
            spk = None
            for s in segs:
                if s["start"] - 0.3 <= st <= s["end"] + 0.3:
                    spk = s["speaker"]
                    break
            if spk is None:  # 落在缝隙里，就近吸附 1.5 秒内的最近段
                near = min(segs, key=lambda s: min(abs(st - s["start"]), abs(st - s["end"])))
                if min(abs(st - near["start"]), abs(st - near["end"])) <= 1.5:
                    spk = near["speaker"]

            who = mapping.get(spk, UNKNOWN) if spk is not None else UNKNOWN
            mm, ss = divmod(int(st), 60)
            lines.append(f"- `{mm:02d}:{ss:02d}` **{who}**：{txt}")

    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("结果 →", out)


if __name__ == "__main__":
    main()
