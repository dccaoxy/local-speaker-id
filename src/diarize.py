#!/usr/bin/env python3
"""说话人分离（diarization）：输出带说话人编号与时间戳的分段。

用法: python diarize.py <wav16k> <out.json> [num_clusters] [threshold]
  num_clusters=-1 表示自动判断人数；threshold 越小越容易把不同人合并。
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import sherpa_onnx
import soundfile as sf

MODEL_DIR = os.environ.get("VP_MODEL_DIR", os.path.expanduser("~/.voiceprint/models"))
SEG_MODEL = f"{MODEL_DIR}/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMB_MODEL = f"{MODEL_DIR}/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"


def main():
    wav = sys.argv[1]
    out = sys.argv[2]
    num_clusters = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    threads = int(sys.argv[5]) if len(sys.argv) > 5 else 4

    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=SEG_MODEL),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB_MODEL, num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_clusters, threshold=threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not cfg.validate():
        print("配置校验失败", file=sys.stderr)
        sys.exit(1)

    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    if sd.sample_rate != 16000:
        print(f"采样率必须是 16000，当前 {sd.sample_rate}", file=sys.stderr)
        sys.exit(1)

    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    print(f"音频时长 {len(audio)/sr:.1f}s，开始分离…", file=sys.stderr)

    t0 = time.time()
    result = sd.process(audio).sort_by_start_time()
    el = time.time()-t0
    print(f"分离耗时 {el:.1f}s，线程 {threads}，实时倍速 {len(audio)/sr/el:.1f}x", file=sys.stderr)

    segs = []
    dur = defaultdict(float)
    cnt = defaultdict(int)
    for r in result:
        segs.append({"start": round(r.start, 2), "end": round(r.end, 2), "speaker": int(r.speaker)})
        dur[int(r.speaker)] += r.end - r.start
        cnt[int(r.speaker)] += 1

    summary = [
        {"speaker": s, "segments": cnt[s], "seconds": round(dur[s], 1)}
        for s in sorted(dur, key=lambda x: -dur[x])
    ]
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"segments": segs, "summary": summary}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
