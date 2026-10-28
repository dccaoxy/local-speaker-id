#!/usr/bin/env python3
"""从 segs + 判定台账渲染 `.说话人.md`。

一条硬规则：**名字不烧进任何中间数据**。md 只是 (segs, decisions, asr) 的渲染产物，
所以改名 / 撤销 / 补第二来源都不需要重跑分离，改一行台账重渲染即可。

斜体 vs 粗体是这套稿子唯一能表达「谁签的字」的地方：
  *张三*   机器认的，没经过你
  **张三**  你确认过的
不区分这两种，自动判错就会被人当成事实引用，而且错得毫无痕迹。
"""
from __future__ import annotations

import os

import config

_WHY = {
    "mixed": "像几个人在抢话，分段本身就不准",
    "mixed_weak": "断断续续，可能混了别人",
    "noisy": "声音太小或太远，没敢认",
    "low_margin": "像两个人，分不出",
    "low_rho": "前后不一致，拿不准",
    "gray_zone": "像，但把握不够",
    "few_windows": "分数够，但有效语音太短",
    "weak_proto": "像，但这个人的样本还不够",
    "few_match": "库里没有这个人",
    "too_short": "这段太短，没认人",
    "few_evidence": "有效语音太少",
    "isolation_broken": "分离可能不完整",
}


def human(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h} 小时 {m} 分"
    return f"{m} 分 {s} 秒"


def labels_for(rec: dict, verdicts: dict) -> tuple[dict, dict]:
    """→ ({cluster: 显示名}, {cluster: 'auto'|'user'|'skip'|'unknown'})。

    未定簇按发言时长降序编 未知A/B/C，**只在未定的里面编号**：
    如果按「所有簇」的序号编，一段只有一个未定簇的稿子里会出现「未知B」，
    读者会以为漏了一个「未知A」。
    """
    clusters = sorted(rec.get("clusters", []), key=lambda c: -c.get("duration_s", 0))
    labels, kind = {}, {}
    n = 0
    for c in clusters:
        cid = c["cluster_id"]
        v = verdicts.get(cid)
        if v and v.get("name"):
            labels[cid], kind[cid] = v["name"], "user"
        elif v and v.get("verdict") in ("skip", "none"):
            labels[cid], kind[cid] = f"{config.UNKNOWN}{chr(ord('A') + n)}", "skip"
            n += 1
        elif c.get("auto_name"):
            labels[cid], kind[cid] = c["auto_name"], "auto"
        else:
            labels[cid], kind[cid] = f"{config.UNKNOWN}{chr(ord('A') + n)}", "unknown"
            n += 1
    return labels, kind


def _fmt(name, kind_) -> str:
    if kind_ == "auto":
        return f"*{name}*"
    if kind_ == "user":
        return f"**{name}**"
    return name


def render(rec: dict, verdicts: dict, asr_rows=None, samples=None, warnings=None) -> str:
    labels, kinds = labels_for(rec, verdicts)
    dur = rec.get("duration", 0.0)
    clusters = sorted(rec.get("clusters", []), key=lambda c: -c.get("duration_s", 0))
    auto_n = sum(1 for k in kinds.values() if k == "auto")
    user_n = sum(1 for k in kinds.values() if k == "user")

    lines = [f"# 说话人识别结果：{rec.get('title') or rec.get('rid')}", "",
             f"> 音频 {human(dur)} · {len(clusters)} 个说话人 · "
             f"你确认 {user_n} 个 / 自动认 {auto_n} 个",
             f"> **粗体**是你确认过的，*斜体*是我自动认的、没经过你。",
             "> 人名只是参考，署名内容请自行核对。"]
    for w in (warnings or []):
        lines.append(f"> ⚠ {w}")
    lines += ["", "## 聚簇判定", "",
              "| 聚簇 | 判定 | 说明 | 说话时长 | 占比 |", "|---|---|---|---|---|"]
    for c in clusters:
        cid = c["cluster_id"]
        lines.append(f"| {cid} | {_fmt(labels.get(cid), kinds.get(cid))} | "
                     f"{_WHY.get(c.get('reason'), '自动判定')} | "
                     f"{human(c.get('duration_s', 0))} | {c.get('share', 0):.0%} |")

    if samples:
        lines += ["", "## 认人参考（每簇抽样，按字数取长句）", ""]
        for item in samples:
            if not item.get("samples"):
                continue
            cid = item["cluster"]
            lines.append(f"- **簇 {cid}（{labels.get(cid, config.UNKNOWN)}）**")
            lines += [f"  > {t}" for t in item["samples"]]

    if asr_rows:
        segs = rec.get("segs") or []
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
            if spk is None and segs:
                near = min(segs, key=lambda s: min(abs(st - s["start"]), abs(st - s["end"])))
                if min(abs(st - near["start"]), abs(st - near["end"])) <= 1.5:
                    spk = near["speaker"]
            who = labels.get(spk, config.UNKNOWN) if spk is not None else config.UNKNOWN
            mm, ss = divmod(int(st), 60)
            lines.append(f"- `{mm:02d}:{ss:02d}` **{who}**：{txt}")

    return "\n".join(lines) + "\n"


def write_md(path: str, text: str):
    """OneDrive 目录里必须 tmp + rename：同步进程会锁文件，用户也可能正开着看。

    tmp 用固定名字而不是 tempfile：同目录才能保证 rename 原子，且 TMPDIR 坏掉时
    不会在当前目录里乱建随机目录。
    """
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path
