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
import math
import os
from dataclasses import dataclass, field

import numpy as np
import sherpa_onnx
import soundfile as sf

import config

# 路径与阈值统一由 config 提供（见 docs/应用架构方案.md D3：单一配置源）
EMB = config.EMB_MODEL
DB = config.DB_DIR
SPEAKERS = config.SPEAKERS
CACHE = os.path.join(DB, "protos.npz")

WIN, HOP = config.WIN, config.HOP
THRESHOLD = config.T_AUTO
MARGIN = config.MARGIN


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


def median_dbfs(audio, sr, frame=0.4):
    """整段录音的 RMS 电平中位数（dBFS）。能量门限的相对部分由它决定。

    必须对**整段**算一次再传进 windows_of：若在簇内按分位算，会恒定削掉
    每个人固定比例的窗，N 就只是时长的函数，跟内容无关。
    """
    n = max(1, int(frame * sr))
    m = (len(audio) // n) * n
    if m < n:
        return -90.0
    r = audio[:m].reshape(-1, n)
    rms = np.sqrt(np.mean(r.astype(np.float32) ** 2, axis=1))
    rms = np.maximum(rms, 1e-9)
    return float(20.0 * np.log10(np.median(rms)))


def gate_db_of(audio, sr):
    """绝对下限与相对动态范围取大者。手机录音整体 -30dBFS、会议麦远端 -55dBFS，
    裸绝对门限会把整场远场参与者判成「无」。"""
    return max(config.MIN_RMS_ABS_DBFS, median_dbfs(audio, sr) - config.RELEASE_MEDIAN_DB)


def windows_of(audio, sr, spans, cap=None, gate_db=None):
    """切 2 秒窗，按能量门限过滤，超上限时**均匀步进**降采样。

    改过两处（都是 bug 修复，别改回去）：
      1) 门限从「簇内 25 分位」改为「整段中位数 - 20dB，且不低于 -50dBFS」
      2) cap 从「取时间序前缀」改为 stride。前缀截断会让 100 分钟的长簇只
         覆盖前 20 分钟，ρ 与 median 都失效，"前段 A 后段混 B"会被算成很纯。
    """
    cap = config.WIN_CAP if cap is None else cap
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
    if gate_db is None:
        gate_db = gate_db_of(audio, sr)
    gate = 10.0 ** (gate_db / 20.0)
    keep = [c for c, r in zip(chunks, rms) if r >= gate]
    if len(keep) > cap:
        step = max(1, len(keep) // cap)
        keep = keep[::step][:cap]
    return keep


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


@dataclass
class ClusterEvidence:
    """一个簇的全部打分证据。判定逻辑只吃这个结构，不散到 UI 层。

    「窗数」三义见 config.py：n_theory 仅自检，n_total 是 ρ 的唯一分母，
    n_valid 只写台账。用 n_valid 当分母会让「谁都不像」的簇反而得到高 ρ。
    """
    cluster_id: int
    duration: float
    share: float = 0.0
    longest_span: tuple = (0.0, 0.0)
    n_theory: int = 0
    n_total: int = 0
    n_valid: int = 0
    win_pass_rate: float = 0.0
    s1: float = 0.0
    s2: float = 0.0
    m: float = 0.0
    rho: float = 0.0
    rho2: float = 0.0
    top1: str | None = None
    second_name: str | None = None
    tier: str | None = None          # 很像 / 有点像 / 无
    reason: list = field(default_factory=list)   # 见 decide_from_evidence
    name: str | None = None          # 最终落的名字；None = 未定
    ask: bool = False                # 是否出卡（用户要不要看这一张）


def score_cluster(audio, sr, spans, cluster_id=0, dur_total=None,
                  protos=None, gate_db=None, ext=None):
    """一个簇 → ClusterEvidence。ρ/ρ2 在同一个循环里算，矩阵已在手上，零额外推理。"""
    gate_db = gate_db_of(audio, sr) if gate_db is None else gate_db
    wins = windows_of(audio, sr, spans, gate_db=gate_db)
    seconds = sum(max(0.0, en - st) for st, en in spans)
    ev = ClusterEvidence(
        cluster_id=cluster_id, duration=round(seconds, 1),
        n_theory=max(0, int(math.floor(seconds)) - 1), n_total=len(wins))

    if not wins:
        ev.reason = ["few_evidence"]
        return ev

    if ext is None:
        ext = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB, num_threads=config.THREADS))
    q = _vecs(ext, wins)
    if len(q) == 0:
        ev.reason = ["few_evidence"]
        return ev

    protos = build_protos() if protos is None else protos
    names = [n for n, p in protos.items() if len(p)]
    if names:
        # B[i, w] = 第 w 个查询窗与第 i 个人（其全部注册窗）的最大余弦
        B = np.stack([(q @ protos[n].T).max(axis=1) for n in names])
        med = np.array([float(np.median(b)) for b in B])
        order = np.argsort(-med)
        ev.top1 = names[int(order[0])]
        ev.s1 = round(float(med[order[0]]), 3)
        if len(order) > 1:
            ev.second_name = names[int(order[1])]
            ev.s2 = round(float(med[order[1]]), 3)
        ev.m = round(ev.s1 - ev.s2, 3)

        best = B.max(axis=0)
        ev.n_valid = int((best >= config.RHO_WIN_GATE).sum())
        ident = B.argmax(axis=0)
        passed = best >= config.RHO_WIN_GATE
        top1_idx = int(order[0])
        ev.rho = round(float(((ident == top1_idx) & passed).sum()) / len(q), 3)
        others = [(int(((ident == i) & passed).sum()), names[i])
                  for i in range(len(names)) if i != top1_idx]
        if others:
            cnt, nm = max(others)
            ev.rho2 = round(cnt / len(q), 3)
            if cnt:
                ev.second_name = nm if ev.rho2 > 0 else ev.second_name
    ev.win_pass_rate = round(ev.n_total / ev.n_theory, 3) if ev.n_theory else 0.0
    decide_from_evidence(ev, dur_total)
    return ev


def decide_from_evidence(ev, dur_total=None, n_recording=None):
    """严格按序判定，顺序不可调换（改顺序等于改产品定义）。

    1. 窗数低于硬底线            → few_evidence，不出卡（bug 检测，不是业务门槛）
    2. 发言时长不够              → 静默标「说话人N」，不出卡
    3. 四项全过                  → 静默落名
    4. 过出卡线                  → 出卡，reason 说明为什么要问
    5. 其余                      → 未知，不出卡
    """
    d = ev.duration if dur_total is None else dur_total
    # n_recording 必须是**整场**录音的窗数，不能拿本簇的 n_total 顶替——
    # 那样下限恒为 0.6*n，永远判通过，等于没有这个门槛。
    n_min_auto = (config.n_min_auto_dyn(n_recording)
                  if n_recording else config.N_MIN_AUTO)

    if ev.n_total < config.N_MIN_ASK:
        ev.reason, ev.ask, ev.tier, ev.name = ["few_evidence"], False, "无", None
        return ev
    if ev.duration < config.sec_threshold(d):
        ev.reason, ev.ask, ev.tier, ev.name = ["short"], False, "无", None
        return ev
    if (ev.s1 >= config.T_AUTO and ev.m >= config.MARGIN
            and ev.rho >= config.RHO_MIN and ev.n_total >= n_min_auto):
        ev.name, ev.ask, ev.tier, ev.reason = ev.top1, False, "很像", []
        return ev
    if ev.s1 >= config.T_ASK:
        ev.ask, ev.tier, ev.name = True, ("有点像" if ev.s1 < config.T_AUTO else "很像"), None
        if ev.m < config.MARGIN:
            ev.reason = ["low_margin"]
        elif ev.rho < config.RHO_MIN:
            # 簇内混人的两种成因，动作相同（都要用户听），只是文案不同
            ev.reason = ["mixed" if ev.rho2 >= 0.25
                         else ("mixed_weak" if ev.rho2 >= 0.10 else "noisy")]
        elif ev.n_total < n_min_auto:
            ev.reason = ["few_windows"]
        else:
            ev.reason = ["low_rho"]
        return ev
    if config.CARD_STRANGER:
        ev.ask, ev.tier, ev.name, ev.reason = True, "无", None, ["unknown_person"]
        return ev
    ev.ask, ev.tier, ev.name, ev.reason = False, "无", None, ["unknown_person"]
    return ev


def to_api(ev):
    """序列化给前端。数字白名单在这里兜，不指望 UI 自觉。

    只有「像两个人」和「分数不够」两支允许出现数字；few_windows / low_rho /
    noisy / unknown_person 一律 null——0.50 印在卡上，用户会立刻拿去对照
    「0.44 才敢认」，之后所有数字都不可信。
    """
    expose = ("low_margin" in (ev.reason or [])) or ev.tier in ("有点像", "无")
    return {
        "cluster_id": ev.cluster_id,
        "duration_s": ev.duration,
        "share": ev.share,
        "state": "pending",
        "tier": ev.tier,
        "reason": ev.reason or [],
        "longest_span": list(ev.longest_span),
        "top1": ev.top1 if expose or not ev.reason else None,
        "sim": round(ev.s1, 2) if expose else None,
        "second": round(ev.s2, 2) if expose else None,
        "second_name": ev.second_name if expose else None,
    }


def is_overlap_of(segs, dur_total):
    """录音级：同一时刻被 ≥2 个簇覆盖的时长占比。只驱动 md 头部预警。

    与 rho2 分工不同——rho2 是窗层面不一致（簇级），这里是分段层面就撞了（录音级）。
    抢话时段分离本身就不准，值得告诉用户「这张卡问得没道理」。
    """
    if not segs or dur_total <= 0:
        return 0.0
    marks = []
    for s in segs:
        marks.append((s["start"], 1))
        marks.append((s["end"], -1))
    marks.sort()
    cov, prev, acc = 0, marks[0][0], 0.0
    for t, delta in marks:
        if cov >= 2 and t > prev:
            acc += t - prev
        cov += delta
        prev = t
    return round(acc / dur_total, 4)


def score_spans(audio, sr, spans, gate_db=None):
    """兼容旧调用：返回 {name: 中位数得分} 和有效窗数。"""
    ev = score_cluster(audio, sr, spans, gate_db=gate_db)
    if not ev.top1:
        return {}, ev.n_total
    return {ev.top1: ev.s1} | ({ev.second_name: ev.s2} if ev.second_name else {}), ev.n_total


def decide(scores, threshold=None, margin=None):
    if not scores:
        return "未知"
    threshold = config.T_AUTO if threshold is None else threshold
    margin = config.MARGIN if margin is None else margin
    order = sorted(scores.items(), key=lambda kv: -kv[1])
    best, s1 = order[0]
    s2 = order[1][1] if len(order) > 1 else 0.0
    return best if (s1 >= threshold and s1 - s2 >= margin) else "未知"


if __name__ == "__main__":
    protos = build_protos(force=True)
    print("原型窗数:", {k: len(v) for k, v in protos.items()})
    print("缓存:", CACHE)
