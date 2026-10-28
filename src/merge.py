#!/usr/bin/env python3
"""跨段说话人合并。

问题：60 分钟以上的录音一次性全局聚类会塌成一个巨簇（实测 100 分钟的会，
4 簇里有一簇吞掉 80 分钟）。原因是全局相似度矩阵在长时程上被增益/房间漂移
拉成近似全连接，归一化割退化。

做法：切段 → 段内独立聚类（信道在 6 分钟内近似平稳）→ 再用本模块把段簇合并成
全局簇。合并只用**窗级**证据，绝不用段级均值——均值会重犯「整段平均向量是
混合物」的错（见 docs/技术结论.md 第四节：均值法间隔 +0.019，滑窗法 +0.111）。

三条约束，优先级从高到低：
  1) 重叠区硬约束：滑段之间有 1 分钟重叠。同一时刻只可能有一个人在说话，
     所以两段在重叠区同时覆盖同一时间点的两个簇，必然是同一个人，直接合并。
     取「互为最大重叠」的一对，避免分割抖动造成的串并。
  2) 反吞并：合并后超过全场 40% 时长、且相似度不到 0.65，拒绝合并。
     两簇都超过 20% 时长时，门槛同样抬到 0.65。这是长会议塌陷的直接对策。
  3) 质量不退化：合并后簇内凝聚度（窗到质心的中位余弦）比合并前差 0.03 以上，
     拒绝合并。防止把「勉强像」的异人并进来后污染整簇的统计量。

相似度定义（对称双中位数，抗少数撞窗）：
    sim(A,B) = min( median_i max_j cos(a_i,b_j) , median_j max_i cos(a_i,b_j) )
取 min 而不是平均，保证两个方向都要成立——单向取最大值会被某一侧几千个窗里
偶然撞上的几帧抬高（这正是文档里「top-K 统计对噪声极度敏感」的教训）。

本模块只依赖 numpy，可用合成向量完整自测；端到端需要真实 embedding。
"""
from __future__ import annotations

import numpy as np

# ── 合并参数（应迁到 config.py，这里给默认值以便独立测试）─────────────────
MERGE_SIM = 0.62          # 软合并的相似度下限（段内同人通常 0.7+，异人 0.35~0.45）
MERGE_SIM_STRICT = 0.65   # 两簇都占比 >20% 时的更高门槛（防吞并）
SWALLOW_SHARE = 0.40      # 单簇超过全场这个比例就算「巨簇」
SWALLOW_SIM = 0.65        # 巨簇场景下仍然允许合并所需的最小相似度
COHESION_DROP = 0.03      # 合并后凝聚度允许的最大退化
MIN_OVERLAP_SEC = 0.50    # 重叠区硬约束所需的最小时长
MERGE_WIN_CAP = 400       # 计算两簇相似度时每簇最多用的窗数（均匀抽样）
EPS = 1e-9


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, EPS)


def _subsample(v: np.ndarray, cap: int = MERGE_WIN_CAP) -> np.ndarray:
    """均匀步进抽样，不用前缀——前缀会让长簇只反映开头几分钟。"""
    if len(v) <= cap:
        return v
    step = max(1, len(v) // cap)
    return v[::step][:cap]


def symmetric_sim(a: np.ndarray, b: np.ndarray, cap: int = MERGE_WIN_CAP) -> float:
    """两簇之间的对称双中位数余弦相似度。a/b 为 (N,D)，内部会归一化。"""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    a = _subsample(_norm(a), cap)
    b = _subsample(_norm(b), cap)
    s = a @ b.T
    return float(min(np.median(s.max(axis=1)), np.median(s.max(axis=0))))


def cohesion(v: np.ndarray) -> float:
    """簇内凝聚度：各窗到质心的余弦中位数。用于「合并是否让簇变糊」的判据。"""
    if len(v) == 0:
        return 0.0
    v = _norm(v)
    c = v.mean(axis=0)
    c = c / max(np.linalg.norm(c), EPS)
    return float(np.median(v @ c))


class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[rb] = ra
        return True


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _overlap_matrix(ca: dict, cb: dict) -> float:
    """两个簇的重叠总秒数（spans 为 [(start,end), ...]）。"""
    total = 0.0
    for sa in ca["spans"]:
        for sb in cb["spans"]:
            total += _overlap(sa, sb)
    return total


def merge(clusters: list[dict], cfg: dict | None = None) -> dict:
    """把段级簇合并成全局簇。

    参数
      clusters: list[dict]，每项至少含
          seg_id   int     该簇来自哪一段（同段的簇不做重叠区硬合并）
          speaker  int     段内局部编号（仅用于诊断）
          spans    [(t0,t1), ...]  绝对时间
          win_vecs np.ndarray (N,D) 该簇的窗向量（未归一化亦可）
        可选
          win_times np.ndarray (N,)  仅用于诊断
      cfg: 覆盖模块级参数（便于标定时扫参）

    返回
      {
        "labels":  [gid, ...]          与输入 clusters 顺序对齐的全局簇编号
        "groups":  {gid: [输入下标]}    每个全局簇包含哪些输入簇
        "forced":  [(i,j), ...]        被重叠区硬约束合并的对
        "soft":    [(i,j,sim), ...]    被软合并的对
        "rejected":[(i,j,sim,why),...] 被拒绝的候选（why ∈ swallow/strict/cohesion）
        "conflicts": int               重叠区出现「一簇对多簇」的次数
      }
    """
    p = dict(MERGE_SIM=MERGE_SIM, MERGE_SIM_STRICT=MERGE_SIM_STRICT,
             SWALLOW_SHARE=SWALLOW_SHARE, SWALLOW_SIM=SWALLOW_SIM,
             COHESION_DROP=COHESION_DROP, MIN_OVERLAP_SEC=MIN_OVERLAP_SEC,
             MERGE_WIN_CAP=MERGE_WIN_CAP)
    p.update(cfg or {})

    n = len(clusters)
    if n == 0:
        return {"labels": [], "groups": {}, "forced": [], "soft": [],
                "rejected": [], "conflicts": 0}

    secs = [sum(e - s for s, e in c["spans"]) for c in clusters]
    total_sec = sum(secs) or 1.0
    share = [s / total_sec for s in secs]

    dsu = _DSU(n)
    forced, soft, rejected = [], [], []
    conflicts = 0

    # ── 1. 重叠区硬约束：互为最大重叠才合并 ────────────────────────────────
    by_seg: dict[int, list[int]] = {}
    for i, c in enumerate(clusters):
        by_seg.setdefault(c["seg_id"], []).append(i)
    seg_ids = sorted(by_seg)

    for k in range(len(seg_ids) - 1):
        left_ids = by_seg[seg_ids[k]]
        # 只和相邻段比（滑段是顺序生成的，跨段不相邻没有重叠区）
        right_ids = by_seg[seg_ids[k + 1]]
        best_l: dict[int, tuple[float, int]] = {}
        best_r: dict[int, tuple[float, int]] = {}
        for i in left_ids:
            for j in right_ids:
                ov = _overlap_matrix(clusters[i], clusters[j])
                if ov < p["MIN_OVERLAP_SEC"]:
                    continue
                if ov > best_l.get(i, (0.0, -1))[0]:
                    best_l[i] = (ov, j)
                if ov > best_r.get(j, (0.0, -1))[0]:
                    best_r[j] = (ov, i)
        for i, (ov, j) in best_l.items():
            if best_r.get(j, (0.0, -1))[1] != i:
                conflicts += 1          # 不是互为最优，说明分割抖动，交给软合并
                continue
            if dsu.union(i, j):
                forced.append((i, j))

    # ── 2. 软合并：按相似度从高到低贪心，带反吞并与质量守卫 ────────────────
    cands = []
    for i in range(n):
        for j in range(i + 1, n):
            if clusters[i]["seg_id"] == clusters[j]["seg_id"]:
                continue                # 同段内已经分开的簇不再合并
            if dsu.find(i) == dsu.find(j):
                continue
            s = symmetric_sim(clusters[i]["win_vecs"], clusters[j]["win_vecs"],
                              p["MERGE_WIN_CAP"])
            cands.append((s, i, j))
    cands.sort(reverse=True)

    def group_of_root(r: int) -> list[int]:
        return [x for x in range(n) if dsu.find(x) == r]

    for s, i, j in cands:
        ri, rj = dsu.find(i), dsu.find(j)
        if ri == rj:
            continue
        gi, gj = group_of_root(ri), group_of_root(rj)
        share_i = sum(share[x] for x in gi)
        share_j = sum(share[x] for x in gj)

        if s < p["MERGE_SIM"]:
            break                       # 已按降序，后面只会更低
        # 两簇都很大时抬门槛（长会议里两个真实的大说话人，不能凭 0.62 就并）
        strict = share_i > 0.20 and share_j > 0.20
        if strict and s < p["MERGE_SIM_STRICT"]:
            rejected.append((i, j, s, "strict"))
            continue
        # 反吞并：并完变巨簇，但相似度不够硬
        if share_i + share_j > p["SWALLOW_SHARE"] and s < p["SWALLOW_SIM"]:
            rejected.append((i, j, s, "swallow"))
            continue
        # 质量不退化
        vi = np.concatenate([clusters[x]["win_vecs"] for x in gi], axis=0)
        vj = np.concatenate([clusters[x]["win_vecs"] for x in gj], axis=0)
        c_before = min(cohesion(vi), cohesion(vj))
        c_after = cohesion(np.concatenate([vi, vj], axis=0))
        if c_after < c_before - p["COHESION_DROP"]:
            rejected.append((i, j, s, "cohesion"))
            continue

        dsu.union(i, j)
        soft.append((i, j, round(s, 4)))

    # ── 3. 编号：按时长降序，便于「说话人0」永远是最主要的那个 ──────────────
    roots = {}
    for i in range(n):
        roots.setdefault(dsu.find(i), []).append(i)
    order = sorted(roots.values(), key=lambda g: -sum(secs[x] for x in g))
    groups = {gid: members for gid, members in enumerate(order)}
    labels = [0] * n
    for gid, members in groups.items():
        for x in members:
            labels[x] = gid

    return {"labels": labels, "groups": groups, "forced": forced,
            "soft": soft, "rejected": rejected, "conflicts": conflicts}


# ── 自测：合成向量，不依赖 sherpa_onnx ────────────────────────────────────
def _synth(n_spk: int = 3, dim: int = 192, seed: int = 0, noise: float = 0.7):
    """每人一个基向量；窗向量 = 基向量 + 各向同性噪声。

    noise 的含义是「噪声向量的范数 / 基向量范数」：
      同人两窗余弦 ≈ 1/(1+noise²)，noise=0.7 → ≈0.67（贴近实测跨录音水平）
      异人两窗余弦 ≈ 0（基向量正交）
    直接给每个维度加 scale=noise 是错的——192 维下噪声范数会被放大 √192 倍。
    """
    rng = np.random.default_rng(seed)
    basis = _norm(rng.normal(size=(n_spk, dim)))
    return basis, rng, noise, dim


def _cluster_of(basis, rng, spk, n, noise, dim, seg_id, t0, t1):
    v = basis[spk] + rng.normal(size=(n, dim)) * (noise / np.sqrt(dim))
    return {"seg_id": seg_id, "speaker": spk, "spans": [(t0, t1)], "win_vecs": v}


def _self_test() -> None:
    basis, rng, noise, dim = _synth(3)
    # 时间线（互不重叠）：A 0-280 与 640-880，B 300-360 与 900-1080，C 360-620
    # 四段：seg_k = [300k, 300k+360)，相邻段重叠 60 秒
    C = [
        # seg0 [0,360)
        _cluster_of(basis, rng, 0, 280, noise, dim, 0, 0, 280),     # 0 A
        _cluster_of(basis, rng, 1, 60, noise, dim, 0, 300, 360),    # 1 B
        # seg1 [300,660)
        _cluster_of(basis, rng, 1, 60, noise, dim, 1, 300, 360),    # 2 B（重叠区，应与 1 硬合并）
        _cluster_of(basis, rng, 2, 260, noise, dim, 1, 360, 620),   # 3 C
        # seg2 [600,960)
        _cluster_of(basis, rng, 2, 20, noise, dim, 2, 600, 620),    # 4 C（重叠区，应与 3 硬合并）
        _cluster_of(basis, rng, 0, 240, noise, dim, 2, 640, 880),   # 5 A（与 0 不相邻，只能软合并）
        # seg3 [900,1260)
        _cluster_of(basis, rng, 1, 180, noise, dim, 3, 900, 1080),  # 6 B（与 1/2 不相邻，软合并）
    ]
    r = merge(C)
    g = r["groups"]
    assert len(g) == 3, f"应合成 3 个全局簇，实际 {len(g)}: {g}"

    def gid_of(idx):
        return [k for k, m in g.items() if idx in m][0]

    assert gid_of(0) == gid_of(5), "A 的两段必须合并（跨非相邻段，靠软合并）"
    assert gid_of(1) == gid_of(2) == gid_of(6), "B 的三段必须合并"
    assert gid_of(3) == gid_of(4), "C 必须合并"
    assert len({gid_of(0), gid_of(1), gid_of(3)}) == 3, "三个人不能混"
    # 重叠区硬约束确实生效过（不是全靠软合并碰巧合上）
    assert r["forced"], "重叠区硬约束没有产生任何合并，逻辑可疑"
    # 编号按时长降序：A 520s 最长
    assert gid_of(0) == 0, "全局编号应按时长降序"

    # 守卫 1：两簇都 >20% 时长，相似度不够就不许并（把 MERGE_SIM 压到 0 逼它走 strict）
    C2 = [_cluster_of(basis, rng, 0, 300, noise, dim, 0, 0, 300),
          _cluster_of(basis, rng, 1, 260, noise, dim, 1, 400, 660)]
    r2 = merge(C2, {"MERGE_SIM": 0.0})
    assert len(r2["groups"]) == 2, "两簇都 >20% 时不应合并"
    assert any(w == "strict" for *_, w in r2["rejected"]), r2["rejected"]

    # 守卫 2：合并后超 40% 变巨簇，相似度不够不许并
    C3 = [_cluster_of(basis, rng, 0, 300, noise, dim, 0, 0, 300),
          _cluster_of(basis, rng, 1, 60, noise, dim, 1, 500, 560)]
    r3 = merge(C3, {"MERGE_SIM": 0.0})
    assert len(r3["groups"]) == 2, "低相似度不应并成巨簇"
    assert any(w == "swallow" for *_, w in r3["rejected"]), r3["rejected"]

    # 同段内的簇永远不合并（段内已经聚过，再并就是重复计票）
    C4 = [_cluster_of(basis, rng, 0, 200, noise, dim, 0, 0, 200),
          _cluster_of(basis, rng, 0, 200, noise, dim, 0, 200, 400)]
    assert len(merge(C4)["groups"]) == 2

    print("self-test ok")
    print(f"  forced={r['forced']}  soft={len(r['soft'])}  rejected={len(r['rejected'])}")
    print(f"  groups={g}")


if __name__ == "__main__":
    _self_test()
