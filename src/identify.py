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

import asr as asr_mod
import config
import winmatch

SEG_MODEL = config.SEG_MODEL
EMB_MODEL = config.EMB_MODEL
MIN_DUR = config.MIN_DUR
UNKNOWN = config.UNKNOWN


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


def load_asr(path):
    """兼容三种转写稿，实现已收敛到 asr.py（只有一处解析逻辑）。"""
    return asr_mod.load_asr(path)


def embed(extractor, samples):
    stream = extractor.create_stream()
    stream.accept_waveform(16000, samples)
    stream.input_finished()
    if not extractor.is_ready(stream):
        return None
    v = np.array(extractor.compute(stream), dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


_WHY = {
    "isolation_broken": "分离可能不完整，整段录音的判定都不作数",
    "mixed": "像几个人在抢话，分段本身就不准",
    "noisy": "声音太小或太远，没敢认",
    "mixed_weak": "断断续续，拿不准",
    "low_margin": "像两个人，分不出",
    "low_rho": "前后不一致，拿不准",
    "few_match": "库里没有这个人",
    "too_short": "这段太短，没认人",
    "few_evidence": "有效语音太少",
    None: "自动判定",
}


def _why(reason):
    return _WHY.get(reason, "自动判定")


def _atomic_write(path, text):
    """OneDrive 目录下必须 tmp + rename：同步进程会锁文件，用户也可能正开着看。"""
    import tempfile
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".vp_", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _human(sec):
    m, s = divmod(int(round(sec)), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h} 小时 {m} 分"
    return f"{m} 分 {s} 秒"


def _label(ev, idx_by_duration, manual=None):
    """未定的簇渲染为 未知A/未知B（按发言时长降序），否则稿子里分不清是同一人还是两个人。"""
    if manual:
        return manual.get(ev.cluster_id, UNKNOWN)
    if ev.name:
        return ev.name
    return f"{UNKNOWN}{idx_by_duration.get(ev.cluster_id, '')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--clusters", type=int, default=-1)
    ap.add_argument("--asr", default=None, help="转写稿；不给则自动找同名 .md/.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--names", default=None,
                    help='手动点名，跳过声纹匹配。格式 "0=金雪竹,1=曹晓雨,2=候选人"')
    ap.add_argument("--bandlimit", action="store_true",
                    help="匹配阶段也用 8k 带宽归一（实测会抹掉区分度，默认关闭）")
    ap.add_argument("--no-diar-bandlimit", action="store_true",
                    help="分离阶段不做带宽归一（默认做：带限能显著改善聚类）")
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
    dur_total = len(audio) / sr
    print(f"音频 {dur_total:.0f}s，开始分离…", file=sys.stderr)

    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=SEG_MODEL),
            num_threads=config.THREADS),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB_MODEL, num_threads=config.THREADS),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=a.clusters, threshold=0.5),
        min_duration_on=0.3, min_duration_off=0.5)
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    segs = [{"start": r.start, "end": r.end, "speaker": int(r.speaker)}
            for r in sd.process(audio).sort_by_start_time()]

    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.audio)),
                                os.path.splitext(os.path.basename(a.audio))[0] + ".说话人.md")
    segs_json = os.path.splitext(out)[0] + ".segs.json"

    # 录音级：同时间段被 ≥2 个簇覆盖的时长占比。只驱动稿子头部预警，不驱动卡片。
    overlap = winmatch.is_overlap_of(segs, dur_total)
    json.dump({"segments": segs, "duration": round(dur_total, 2),
               "is_overlap": overlap > config.OVERLAP_ALERT,
               "overlap_ratio": overlap},
              open(segs_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("分段结果 →", segs_json, file=sys.stderr)

    asr_rows = asr_mod.load_asr(a.asr or asr_mod.find_asr_for(a.audio))
    spans_by = defaultdict(list)
    for s in segs:
        if s["end"] - s["start"] >= MIN_DUR:
            spans_by[s["speaker"]].append((s["start"], s["end"]))

    # 手动点名：不出卡、不比对，直接按给定映射渲染
    manual = {}
    if a.names:
        for pair in a.names.replace("，", ",").split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                manual[int(k.strip())] = v.strip()

    windows = {}
    if not manual:
        ext = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB_MODEL, num_threads=config.THREADS))
        protos = winmatch.build_protos()
        gate_db = winmatch.gate_db_of(audio_m, 16000)
        evs = [winmatch.score_cluster(audio_m, 16000, spans, cluster_id=spk,
                                      dur_total=dur_total, protos=protos,
                                      gate_db=gate_db, ext=ext)
               for spk, spans in sorted(spans_by.items())]
        for ev in evs:
            ev.share = round(ev.duration_s / dur_total, 3) if dur_total else 0.0
            if not ev.longest_span or ev.longest_span == (0.0, 0.0):
                sp = max(spans_by[ev.cluster_id], key=lambda x: x[1] - x[0])
                ev.longest_span = sp

        # 互斥：一个人只能占一个簇。按分数降序贪心，被更强的簇占走后降级为出卡。
        taken = set()
        for ev in sorted(evs, key=lambda e: -e.s1):
            if ev.name and ev.name in taken:
                ev.name, ev.ask = None, True
                ev.reason = ["same_person"]
            elif ev.name:
                taken.add(ev.name)
        # 塌陷是录音级失效：单簇占比过高时任何簇卡都不该出，只留黄条预警
        n_clusters = len(evs)
        max_share = max((e.duration_s / dur_total for e in evs), default=0.0)
        for ev in evs:
            winmatch.decide_from_evidence(ev, dur_total, n_clusters, max_share)
            windows[ev.cluster_id] = ev
        collapsed = (n_clusters >= config.COLLAPSE_MIN_CLUSTERS
                     and max_share > config.COLLAPSE_SHARE)
        note = (f"{config.SCORER}@{config.WIN}/{config.HOP}｜声纹库 {len(protos)} 人｜"
                f"判定 T_AUTO {config.T_AUTO} / T_ASK {config.T_ASK}")
    else:
        evs = []
        note = "手动点名"
        collapsed = False
        max_share = 0.0

    order = {spk: chr(ord("A") + i) for i, spk in
             enumerate(sorted(spans_by, key=lambda k: -sum(e - s for s, e in spans_by[k])))}
    labels = {spk: _label(windows.get(spk) or _Fake(spk), order, manual)
              for spk in spans_by}

    lines = [f"# 说话人识别结果：{os.path.basename(a.audio)}", "",
             f"> 音频 {_human(dur_total)} · {len(spans_by)} 个说话人",
             f"> {note}",
             "> 「未知」表示相似度不足或与次优候选差距太小，需人工确认。"]
    if collapsed:
        # 带具体数字：预警要可核实，否则用户只会当成噪音
        lines.append(f"> ⚠ 这段录音有 {max_share * dur_total / 60:.0f} 分钟被归成了同一个人，"
                     "分离可能不完整。")
    if overlap > config.OVERLAP_ALERT:
        lines.append(f"> ⚠ 有 {overlap:.0%} 的时长被多个说话人同时覆盖，"
                     "分离可能不完整，重叠部分请回听。")
    lines += ["", "## 聚簇判定", "",
              "| 聚簇 | 判定 | 说明 | 说话时长 | 占比 |", "|---|---|---|---|---|"]
    if manual:
        for spk in sorted(spans_by, key=lambda k: -sum(e - s for s, e in spans_by[k])):
            secs = sum(e - s for s, e in spans_by[spk])
            lines.append(f"| {spk} | {labels.get(spk, UNKNOWN)} | 手动指定 | "
                         f"{_human(secs)} | {secs / dur_total:.0%} |")
    else:
        for ev in sorted(evs, key=lambda e: -e.duration_s):
            tag = labels.get(ev.cluster_id, UNKNOWN)
            star = "" if ev.name else " *"
            lines.append(f"| {ev.cluster_id} | {tag}{star} | {_why(ev.reason)} | "
                         f"{_human(ev.duration_s)} | {ev.share:.0%} |")
    if not manual:
        lines += ["", "带 * 的是没经过核对的判定，或需要人工确认的簇。"]

    samples = asr_mod.samples_by_cluster(segs, asr_rows)
    if samples:
        lines += ["", "## 认人参考（每簇抽样，按字数取长句）", ""]
        for item in samples:
            if not item["samples"]:
                continue
            lines.append(f"- **簇 {item['cluster']}（{labels.get(item['cluster'], UNKNOWN)}）**")
            lines += [f"  > {t}" for t in item["samples"]]

    if asr_rows:
        lines += ["", "## 逐句（按声纹命名）", ""]
        prev, prev_t = None, -99.0
        for st, en, txt in asr_rows:
            if not txt:
                continue
            if txt == prev and st - prev_t < 3:
                continue
            prev, prev_t = txt, st
            spk = None
            for s in segs:
                if s["start"] - 0.3 <= st <= s["end"] + 0.3:
                    spk = s["speaker"]
                    break
            if spk is None:
                near = min(segs, key=lambda s: min(abs(st - s["start"]), abs(st - s["end"])))
                if min(abs(st - near["start"]), abs(st - near["end"])) <= 1.5:
                    spk = near["speaker"]
            who = labels.get(spk, UNKNOWN) if spk is not None else UNKNOWN
            mm, ss = divmod(int(st), 60)
            lines.append(f"- `{mm:02d}:{ss:02d}` **{who}**：{txt}")

    _atomic_write(out, "\n".join(lines) + "\n")
    print(json.dumps([winmatch.to_api(e) for e in evs], ensure_ascii=False, indent=2))
    print("结果 →", out)


class _Fake:
    """手动点名模式下没有 evidence，用最小占位让 _label 走同一路径。"""

    def __init__(self, spk):
        self.cluster_id, self.name = spk, None


if __name__ == "__main__":
    main()
