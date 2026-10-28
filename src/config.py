#!/usr/bin/env python3
"""唯一配置源。其他模块一律 `import config`，禁止再写任何字面量阈值。

## 为什么必须集中
判定阈值曾经散落在三处且互相矛盾：identify.py 的 0.70、winmatch.py 的 0.40、
产物里的 0.45。其中 0.70 属于**质心模式**（整段取均值）的量纲，与滑窗模式不可互换
（实测同人都是 0.487，异人一个是 0.468、一个是 0.376）。该模式已整段删除，
现在只剩一套滑窗刻度。

## 「窗数」三义，代码里不要混用
  N_theory = floor(seconds) - 1        # 理论值，仅展示与自检
  N_total  = windows_of() 的返回长度    # 能量过滤 + 均匀步进后；**ρ 的唯一分母**
  N_valid  = best_sim >= T_ASK 的窗数   # 只写台账做诊断，禁止参与判定
用 N_valid 当 ρ 的分母会让「一堆窗谁都不像」的簇反而得到高 ρ，正好和要抓的混合物相反。

## is_overlap 与 rho2 分工（别混用）
  is_overlap：录音级。segs 里同一时间段被 ≥2 个簇覆盖的时长占比 > 15%。
              只驱动 md 头部一行「分离可能不完整」预警。
  rho2      ：簇级。指向「除 top1 外第二频繁身份」的窗占比。
              只驱动卡片文案（mixed / mixed_weak / noisy）。
"""
import os

# ── 路径 ────────────────────────────────────────────────────────────────
MODEL_DIR = os.environ.get("VP_MODEL_DIR", os.path.expanduser("~/.voiceprint/models"))
SEG_MODEL = os.path.join(MODEL_DIR, "sherpa-onnx-pyannote-segmentation-3-0", "model.onnx")
EMB_MODEL = os.path.join(MODEL_DIR, "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")

# 合规红线：声纹向量只写本地，任何路径都不许指向 ~/OneDrive/。
# 生物识别信息躺在同步盘上是合规问题，不是工程问题。
VP_ROOT = os.path.expanduser("~/.voiceprint")
DB_DIR = os.environ.get("VP_DB", os.path.join(VP_ROOT, "db"))
WORK_DIR = os.environ.get("VP_WORK", os.path.join(VP_ROOT, "work"))
STATE = os.path.join(VP_ROOT, "state.json")
DECISIONS = os.path.join(VP_ROOT, "decisions.jsonl")
SPEAKERS = os.environ.get("VP_SPEAKERS", os.path.join(VP_ROOT, "speakers.json"))
PROTOS_DIR = os.path.join(DB_DIR, "protos")
PROTOS_CURRENT = os.path.join(PROTOS_DIR, "CURRENT")
EMB_CACHE_DIR = os.path.join(DB_DIR, "emb")        # L1：按来源的窗向量
VOICES_DIR = os.path.join(DB_DIR, "voices")        # 注册时抽出的簇音频，切断对 OneDrive 原文件名的依赖
CLIP_CACHE_DIR = os.path.expanduser("~/Library/Caches/voiceprint")   # 15 秒试听样本
RECORDING_DIR = os.path.expanduser("~/OneDrive/数据库/录音")          # 只读，永不写入


def assert_not_onedrive(path):
    """写入前调用。防的是以后有人随手把 DB_DIR 指回同步盘。"""
    p = os.path.abspath(os.path.expanduser(path))
    if "OneDrive" in p:
        raise RuntimeError(f"拒绝写入 OneDrive：{p}（声纹与状态只允许落在 {VP_ROOT}）")
    return p


# ── 声纹比对（只有一套刻度）─────────────────────────────────────────────
SCORER = "window"
WIN, HOP = 2.0, 1.0
MIN_DUR = 1.5              # 短于此的片段不提取声纹

T_AUTO = 0.44              # 静默落名
T_ASK = 0.36               # 出卡下限
MARGIN = 0.05
RHO_MIN = 0.70             # 与 RHO_WIN_GATE 成对改，单改任一个是 bug
RHO_WIN_GATE = 0.36        # 统计 ρ/ρ2 时认定"这一窗算数"的门槛；与 T_ASK 同值
N_MIN_ASK = 20             # 硬底线，低于此不出卡 → reason=few_evidence（bug 检测，不是业务门槛）
N_MIN_AUTO = 60            # 静默所需窗数。相邻窗重叠 50%，n_eff≈0.4~0.7n

# 「时长」是唯一的业务门槛，nwin 只是硬底线与诊断。两个门槛并存会让 60 秒边界上的簇随机落档。
SEC_RULE_FACTOR = 0.3
SEC_RULE_CAP = 60

# 能量门限：绝对下限 + 相对整段动态范围。全录音中位数在 score 阶段算一次，
# 不在簇内算——簇内分位会恒定削掉每人固定比例的窗，让 N 只是时长的函数。
MIN_RMS_ABS_DBFS = -50
RELEASE_MEDIAN_DB = 20

# 切窗上限。**必须均匀步进，不能取前缀**：100 分钟的簇约 4800 窗，
# 前缀截断会让 ρ 和 median 只反映前 20 分钟，"前段 A 后段混 B"会被算成很纯。
WIN_CAP = 1200

# ── 流水线 ──────────────────────────────────────────────────────────────
THREADS = 4                # 实测 4 最快，8 慢 20%，10 慢 2.4 倍
STABLE_SEC = 60            # 文件 size+mtime 连续这么多秒不变才认为同步完成
SEG_MIN_SEC = 1500         # 短于 25 分钟且簇数少时单段直跑（分段会拖慢）
SEG_MAX_CLUSTERS = 5       # 簇数 ≥5 也走分段：人多时相似度矩阵更病态，短录音同样会塌
SEG_LEN = 360              # 每段 6 分钟
SEG_HOP = 300              # 步长 5 分钟，1 分钟重叠用于边界去重
OVERLAP_ALERT = 0.15       # 同时间段多簇覆盖占比超过此值 → is_overlap

# 陌生人是否出卡。ux 指出「库里没有这个人」的模板建在出卡上，若 s1<T_ASK 一律不出卡，
# 那个模板永远不会显示、新人也永远没有认领入口。终版判定顺序是「不出卡」，
# 故此处留 False 待 team-lead 拍板；改 True 即放行（不改任何其他逻辑）。
CARD_STRANGER = False

# ── 待确认稿 ────────────────────────────────────────────────────────────
UNKNOWN = "未知"            # 未处置簇渲染为 未知A / 未知B（按发言时长降序）
CLIP_SEC = 5.0             # 每段取 5 秒
CLIP_N = 3                 # 能量最高的 3 段拼成 15 秒
CLIP_GAP = 0.25            # 段间静音
SAMPLE_K = 3               # 每簇抽 3 句转写样例
SAMPLE_MAX_CHARS = 30


def sec_threshold(dur_total):
    """业务门槛：簇发言时长至少要这么长才值得问人 / 才够静默落名。"""
    return min(SEC_RULE_CAP, dur_total * SEC_RULE_FACTOR)


def n_min_auto_dyn(n_recording):
    """短录音放宽：整场窗数少时不应该因为绝对窗数不够而永远落不了名。"""
    return min(N_MIN_AUTO, max(30, int(0.6 * n_recording)))


def env_overrides():
    """给 /health 与报告头用。前端文案里的 0.44 必须从这里读，别写死在 JS 里。"""
    return {
        "scorer": f"{SCORER}@{WIN}/{HOP}",
        "threshold": T_AUTO, "ask": T_ASK, "margin": MARGIN,
        "rho_min": RHO_MIN, "rho_win_gate": RHO_WIN_GATE,
        "n_min_ask": N_MIN_ASK, "n_min_auto": N_MIN_AUTO,
        "sec_rule": [SEC_RULE_FACTOR, SEC_RULE_CAP],
        "db": DB_DIR,
    }
