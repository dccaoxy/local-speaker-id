#!/usr/bin/env python3
"""生成「每一簇抽样句子」清单，供人工确认每个声纹聚簇对应哪个人。

用法: python sample.py <segs.json> <asr.md> <out.md>
"""
import json
import os
import re
import sys

LINE = re.compile(r"^-\s*\[(\d+):(\d+\.\d+),(\d+):(\d+\.\d+),(\d+)\]\s*(.*)$")


def load_asr(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        m = LINE.match(line.strip())
        if not m:
            continue
        s = int(m.group(1)) * 60 + float(m.group(2))
        e = int(m.group(3)) * 60 + float(m.group(4))
        rows.append((s, e, int(m.group(5)), m.group(6).strip()))
    return rows


def main():
    segs_path, asr_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    segs = json.load(open(segs_path, encoding="utf-8"))["segments"]
    asr = load_asr(asr_path)

    by_spk = {}
    for s in segs:
        by_spk.setdefault(s["speaker"], []).append(s)

    lines = ["# 声纹聚簇抽样（待人工认人）", "",
             "每个聚簇取该簇里最长的 8 段，附上同一时间段内的转写文字。",
             "读一遍就能判断哪一簇是谁在说话。", ""]
    for spk in sorted(by_spk, key=lambda k: -sum(x["end"] - x["start"] for x in by_spk[k])):
        segs_s = sorted(by_spk[spk], key=lambda x: -(x["end"] - x["start"]))[:8]
        total = sum(x["end"] - x["start"] for x in by_spk[spk])
        lines.append(f"## 聚簇 {spk}（{len(by_spk[spk])} 段，累计 {total/60:.1f} 分钟）")
        lines.append("")
        for s in sorted(segs_s, key=lambda x: x["start"]):
            txt = " ".join(r[3] for r in asr if r[0] >= s["start"] - 0.5 and r[0] <= s["end"] + 0.5)
            txt = txt[:120] + ("…" if len(txt) > 120 else "")
            mm, ss = divmod(int(s["start"]), 60)
            lines.append(f"- `{mm:02d}:{ss:02d}` {txt or '（该时段无可对齐文字）'}")
        lines.append("")

    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print("已生成", out)


if __name__ == "__main__":
    main()
