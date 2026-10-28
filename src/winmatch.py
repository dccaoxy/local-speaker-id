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
    cap = config.CAP if cap is None else cap
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
    """一个簇的全部打分证据。判定只吃这个结构，逻辑不外溢到 UI。

    时长有两个，别混（产品口径-终版 §3）：
      duration_s 含停顿，只用于卡片统计行「说得最多 · 18 分 20 秒」
      eff_sec    有效语音 = n_total × HOP，门槛判定只看它
    一个时长 200 秒的簇，有效语音可能只有 35 秒。混用会渲染出
    「说了 18 分钟，判成太短」这种判定 bug。
    """
    cluster_id: int
    duration_s: float = 0.0
    eff_sec: float = 0.0
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
    reason: str | None = None        # 单值枚举，见 REASONS
    name: str | None = None
    ask: bool = False
    proto_n: int = 0             # 该 top1 的原型窗数（原型质量分级用）
    source_n: int = 0            # 该 top1 的独立来源数
    verified: bool = False       # 跨来源自检是否通过
    silent_ok: bool = True       # 是否获准静默（protolib.eligible(for_silent=True)）
    is_self_match: bool = False      # 注册音频自身当查询，台账里必须能剔除
    can_one_click: bool = False      # 界面能不能给「是 X」一键按钮（原型质量门控）


# 十支固定枚举，按优先级排列。前端按枚举查表出文案，不许自由文本。
#
# 命名以已提交代码为准，不再改名（改名要同时动代码+文档+接口三处，这轮已经付过代价）。
# isolation_broken / too_short / few_evidence 不进卡片流，只进 md 或录音级黄条。
REASONS = ("isolation_broken", "mixed", "mixed_weak", "noisy",
           "low_margin", "few_windows", "weak_proto", "gray_zone",
           "low_rho", "few_match", "too_short", "few_evidence")
# 文案说「没有可靠的人」时 top1 必须为 null
_NO_RELIABLE = {"mixed", "noisy", "mixed_weak", "few_match", "isolation_broken",
                "few_evidence", "low_rho"}
# 主按钮：文案给了名字就给「是 X」，没给名字就只能「先留空」/「选人」
_PRIMARY = {
    "low_margin": "choose_one",     # 两选一
    "few_windows": "confirm",       # 就按 X 记（我确认）
    "weak_proto": "confirm",
    "gray_zone": "yes_named",       # 是 X（带名字，但系统犹豫）
    "low_rho": "skip",              # 先留空
    "mixed": "skip",
    "mixed_weak": "skip",
    "noisy": "skip",
    "few_match": "new_person",
}



def _usable_matrices(protos, prov, recording, man):
    """取出每个人可用的原型矩阵，并剔除「来源就是本录音」的窗。

    这一步是 §24 的主修法。面试录音那次事故：张彤的原型窗取自**这条录音本身**的
    簇0，于是簇1 里像簇0 的窗全得 1.000，中位数被拉到 0.569，ρ 反而 0.994——
    四条静默条件全部通过，把候选人静默签成了张彤。
    「自己给自己作证」只能靠剔除同源窗来切断，调阈值救不了。
    """
    mats, meta = {}, {}
    for name, p in protos.items():
        if not len(p):
            continue
        m = prov.get(name) if prov else None
        if m is not None and recording:
            keep = np.array([str(x) != str(recording) for x in m])
            if not keep.any():
                continue          # 这个人的样本全部来自本录音 → 对这段不参与打分
            mats[name] = p[keep]
        else:
            mats[name] = p
        pm = (man or {}).get("people", {}).get(name, {})
        meta[name] = {
            "proto_n": int(len(mats[name])),
            "source_n": int(pm.get("n_sources", 0)),
            "verified": bool(pm.get("verified", False)),
        }
    return mats, meta


def score_cluster(audio, sr, spans, cluster_id=0, dur_total=None,
                  protos=None, prov=None, man=None, gate_db=None, ext=None,
                  recording=None):
    """一个簇 → ClusterEvidence。ρ/ρ2 在同一个循环里算，零额外推理。

    recording：本段录音的标识。给了就会剔除「来源＝本录音」的原型窗。
    """
    gate_db = gate_db_of(audio, sr) if gate_db is None else gate_db
    wins = windows_of(audio, sr, spans, gate_db=gate_db)
    dur = sum(max(0.0, en - st) for st, en in spans)
    ev = ClusterEvidence(cluster_id=cluster_id, duration_s=round(dur, 1),
                         n_total=len(wins), n_theory=max(0, int(math.floor(dur)) - 1))
    ev.eff_sec = round(ev.n_total * HOP, 1)

    if not wins:
        ev.reason = "few_evidence"
        return ev

    if ext is None:
        ext = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB, num_threads=config.THREADS))
    q = _vecs(ext, wins)
    if len(q) == 0:
        ev.reason = "few_evidence"
        return ev

    if protos is None:
        try:
            import protolib
            p2, prov2, man2 = protolib.load_with_prov()
            if p2:
                protos, prov, man = p2, prov2, man2
        except Exception:
            pass
    if protos is None:
        protos = build_protos()

    mats, meta = _usable_matrices(protos, prov, recording, man)
    names = list(mats)
    # 原型质量门控：低窗原型不参与自动 top1 排名，但仍留在人工候选列表里
    if man and man.get("people"):
        try:
            import protolib
            pool = protolib.eligible(man, for_silent=False)
            names = [n for n in names if n in pool] or names
        except Exception:
            pass
    if names:
        # B[i, w] = 第 w 个查询窗与第 i 个人（其全部注册窗）的最大余弦
        B = np.stack([(q @ mats[n].T).max(axis=1) for n in names])
        med = np.array([float(np.median(b)) for b in B])
        order = np.argsort(-med)
        top1_idx = int(order[0])
        ev.top1 = names[top1_idx]
        ev.s1 = round(float(med[top1_idx]), 3)
        if len(order) > 1:
            ev.second_name = names[int(order[1])]
            ev.s2 = round(float(med[order[1]]), 3)
        ev.m = round(ev.s1 - ev.s2, 3)

        mt = meta.get(ev.top1, {})
        ev.proto_n, ev.source_n = mt.get("proto_n", 0), mt.get("source_n", 0)
        ev.verified = mt.get("verified", False)
        if man and man.get("people"):
            try:
                import protolib
                ev.silent_ok = ev.top1 in protolib.eligible(man, for_silent=True)
            except Exception:
                ev.silent_ok = False
        else:
            ev.silent_ok = False     # 没有版本信息 → 无法证明可静默，一律不静默
        ev.can_one_click = (ev.proto_n >= config.N_PROTO_MIN
                            and ev.source_n >= config.N_SOURCE_MIN
                            and ev.verified)
        # 查询簇就是注册来源本身时 sim≡1.0，这种记录不能进阈值标定
        ev.is_self_match = ev.s1 >= 0.999

        best = B.max(axis=0)
        passed = best >= config.RHO_WIN_GATE
        ev.n_valid = int(passed.sum())
        ident = B.argmax(axis=0)
        ev.rho = round(float(((ident == top1_idx) & passed).sum()) / len(q), 3)
        others = [(int(((ident == i) & passed).sum()), names[i])
                  for i in range(len(names)) if i != top1_idx]
        if others:
            cnt, nm = max(others)
            ev.rho2 = round(cnt / len(q), 3)
            if cnt:
                ev.second_name = nm

    # 分母必须与 windows_of 共用同一个 CAP：否则 30~100 分钟的簇会被截断稀释，
    # 长簇被系统性推进 noisy 分支，界面写「声音太小」而实际他说了一小时。
    denom = min(ev.n_theory, config.CAP)
    ev.win_pass_rate = round(ev.n_total / denom, 3) if denom > 0 else 0.0
    decide_from_evidence(ev, dur_total)
    return ev


def decide_from_evidence(ev, dur_total=None, n_clusters=None, max_share=None):
    """严格按 产品口径-终版 §1 的序列，命中即返回，不许并行判断。

    并行判断会让同一段录音跑两次显示不同理由——这是最容易被用户抓到的
    「这工具不靠谱」的证据。

    与早期版本的差别（都是为了「每条 reason 都真的会发生」）：
      · 补 gray_zone（0.36~0.44 是主力卡，之前全落进 low_rho，文案说错话）
      · 补 few_windows（分数够、只是听得短）与 weak_proto（原型资格不足）
      · rho2 ≥ 0.25 统一走 mixed，不再按 s1 分叉
    """
    d = dur_total if dur_total is not None else max(ev.duration_s, 1.0)

    # 1. 窗数不够，纯噪声 —— 这是 bug 信号，不是业务判断，不出卡也不给用户文案
    if ev.n_total < config.N_MIN_ASK:
        ev.reason, ev.ask, ev.tier, ev.name = "few_evidence", False, None, None
        return ev

    # 2. 塌陷：整段录音的分离失效，任何簇卡都不该出（挂录音级黄条）
    if (n_clusters is not None and n_clusters >= config.COLLAPSE_MIN_CLUSTERS
            and max_share is not None and max_share > config.COLLAPSE_SHARE):
        ev.reason, ev.ask, ev.tier, ev.name = "isolation_broken", False, None, None
        return ev

    gated = ev.eff_sec >= config.card_gate(d)

    if ev.s1 >= config.T_ASK:
        # 抢话优先：窗口大面积指向第二个人时，任何名字都给不准
        if ev.rho2 >= 0.25:
            ev.reason, ev.ask, ev.tier, ev.name = "mixed", True, "无", None
            return ev
        if ev.s1 >= config.T_AUTO:
            # 分数已经很高，就不再受「出卡时长门槛」约束——门槛是防打扰的，
            # 而这一档恰恰是「听得出像谁、只是他说得少」，问了最有价值。
            # 早期版本在这里也套 gate，导致 few_windows 永远不会发生（N<60 必然不 gated）。
            ev.tier = "很像"
            if ev.rho < config.RHO_MIN:
                ev.reason = "mixed_weak" if ev.rho2 >= 0.10 else "noisy"
                ev.ask, ev.name = True, None
                return ev
            if ev.m < config.MARGIN:
                ev.reason, ev.ask, ev.name = "low_margin", True, None
                return ev
            if ev.n_total < config.N_MIN_AUTO:
                ev.reason, ev.ask, ev.name = "few_windows", True, None
                return ev
            if not ev.silent_ok:
                # 分数够、但这个人只有单来源/窗数不足/跨录音自检未过。
                # 静默署名是「不告诉用户就写名字」，必须要求证据干净。
                ev.reason, ev.ask, ev.name = "weak_proto", True, None
                return ev
            ev.name, ev.ask, ev.reason = ev.top1, False, None
            return ev
        # 0.36 ≤ s1 < 0.44：占比最大的一批，宁可问一句
        if not gated:
            ev.reason, ev.ask, ev.tier, ev.name = "too_short", False, "无", None
            return ev
        ev.tier = "有点像"
        ev.ask, ev.name = True, None
        ev.reason = "low_rho" if ev.rho < config.RHO_MIN else "gray_zone"
        return ev

    # 3. 分数低：先看信号质量，糊了就不问（问了也答不出来）
    if ev.win_pass_rate < config.WIN_PASS_MIN:
        ev.reason, ev.ask, ev.tier, ev.name = "noisy", False, "无", None
        return ev

    # 4. 分低但信号干净且够长 → 库里没这人
    if gated and ev.duration_s >= config.NEW_MIN_SEC:
        ev.reason, ev.ask, ev.tier, ev.name = "few_match", True, "无", None
        return ev

    # 5. 其余：静默标「说话人N」
    ev.reason = "too_short" if not gated else None
    ev.ask, ev.tier, ev.name = False, "无", None
    return ev


def to_api(ev):
    """序列化给前端。**全禁数字**（含调试灰字）——接口层返 null，不靠前端自律。

    唯一的例外是 duration_s（卡片统计行）与 few_windows 的秒数（口径 §19 明确
    允许的孤例：前提是有效语音 < 60 秒，CAP 咬不到，用户能自行核实）。
    """
    top1 = None if ev.reason in _NO_RELIABLE else ev.top1
    primary = _PRIMARY.get(ev.reason, "choose")
    if ev.reason in ("gray_zone", "low_margin") and not ev.can_one_click:
        primary = "choose"        # 原型不达标：不给一键确认，但名字仍置顶
    return {
        "cluster_id": ev.cluster_id,
        "duration_s": ev.duration_s,
        "share": ev.share,
        "state": "pending",
        "tier": ev.tier,
        "reason": ev.reason,
        "top1": top1,
        "choose_between": ([ev.top1, ev.second_name]
                           if primary == "choose_one" and ev.top1 and ev.second_name
                           else None),
        "primary_action": primary,
        "can_one_click": bool(ev.can_one_click),
        "few_seconds": int(ev.eff_sec) if ev.reason == "few_windows" else None,
        "longest_span": list(ev.longest_span),
        "is_self_match": ev.is_self_match,
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
