#!/usr/bin/env python3
"""唯一配置源。其他模块一律 `import config`，禁止再写任何字面量阈值。

口径来源：`docs/产品口径-终版.md`（封版）。与本文件冲突的旧文档、旧消息、
旧产物头信息一律以该文件为准。改任何数值都会改变历史 decisions 的可比性。

## 字段命名（不许自创第四个）
  N_theory   = floor(seconds) − 1      理论窗数，只做展示与自检
  N_total    = windows_of 返回长度      能量过滤 + cap 之后；ρ 的唯一分母
  N_valid    = best_sim ≥ T_ASK 的窗数  只进台账做诊断，不参与任何判定
  eff_sec    = N_total × HOP            HOP=1.0s，故 30/60 窗 ≡ 30/60 秒有效语音
  duration_s = 簇内各段时长之和          含停顿，只用于展示「说得最多 · 18 分 20 秒」
判定只用 N_total。eff_sec ≠ duration_s：时长 200 秒的簇，有效语音可能只有 35 秒。

## ⚠ 必读警示（来自产品口径-终版 §2，原样保留）
即使 80 个窗，±2SE≈0.039 也大于 T_AUTO 与最近正例（0.456）之间的间隙 0.016。
**统计上没有任何簇「够格」静默署名。** 静默档的本质是「用 T_AUTO 把可疑的推给人，
接受残余错误率」，不是「窗够了就保正确」。N_MIN_AUTO 只用于挡住纯噪声。

⚠ win_pass_rate = N_total / min(N_theory, CAP)。分母必须封顶，否则 30~100 分钟的簇
会被 cap 截断污染，判据在该区间失效。簇长 > CAP 时 win_pass_rate ≡ 1.0，无诊断意义。

⚠ win_pass_rate 测不到**分离塌陷**（塌陷时窗能量全够，它接近常态 0.74）。
塌陷只能靠 max_cluster_share 判。
"""
import os

# ── 路径 ────────────────────────────────────────────────────────────────
def _guess_model_dir() -> str:
    """模型目录：环境变量 > ~/.voiceprint/models > WorkBuddy 的模型缓存。

    第二个默认值曾是错的：`~/.voiceprint/models` 在本机不存在，模型实际在
    `~/.workbuddy/models/voiceprint`，新环境跑起来直接 "Load model failed"。
    按顺序探测，找到就用，避免再写死一个不存在的路径。
    """
    cands = [os.path.expanduser("~/.voiceprint/models"),
             os.path.expanduser("~/.workbuddy/models/voiceprint")]
    for c in cands:
        if os.path.exists(os.path.join(c, "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")):
            return c
    return cands[1]


MODEL_DIR = os.environ.get("VP_MODEL_DIR") or _guess_model_dir()
SEG_MODEL = os.path.join(MODEL_DIR, "sherpa-onnx-pyannote-segmentation-3-0", "model.onnx")
EMB_MODEL = os.path.join(MODEL_DIR, "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")

# 合规红线：声纹向量、原型、decisions、state 只许落本地，绝不进 OneDrive。
# VP_ROOT 可覆盖，是为了让测试能把整套状态放进临时目录，不污染真实库。
VP_ROOT = os.path.expanduser(os.environ.get("VP_ROOT", "~/.voiceprint"))
DB_DIR = os.environ.get("VP_DB", os.path.join(VP_ROOT, "db"))
WORK_DIR = os.environ.get("VP_WORK", os.path.join(VP_ROOT, "work"))
STATE = os.path.join(VP_ROOT, "state.json")
DECISIONS = os.path.join(VP_ROOT, "decisions.jsonl")
SPEAKERS = os.environ.get("VP_SPEAKERS", os.path.join(VP_ROOT, "speakers.json"))
PROTOS_DIR = os.path.join(DB_DIR, "protos")
PROTOS_CURRENT = os.path.join(PROTOS_DIR, "CURRENT")
EMB_CACHE_DIR = os.path.join(DB_DIR, "emb")       # L1：按来源的窗向量
VOICES_DIR = os.path.join(DB_DIR, "voices")       # 注册时抽出的簇音频切片
CLIP_CACHE_DIR = os.path.expanduser("~/Library/Caches/voiceprint")
RECORDING_DIR = os.environ.get(
    "VP_RECORDINGS", os.path.expanduser("~/OneDrive/数据库/录音"))   # 只往里写 .说话人.md


def assert_not_onedrive(path):
    """写入前调用。防的是以后有人随手把 DB_DIR 指回同步盘。"""
    p = os.path.abspath(os.path.expanduser(path))
    if "OneDrive" in p:
        raise RuntimeError(f"拒绝写入 OneDrive：{p}（声纹与状态只允许落在 {VP_ROOT}）")
    return p


# ── 判定常数（产品口径-终版 §2，唯一来源）───────────────────────────────
SCORER = "window"          # 质心模式已整段删除，没有第二套刻度
WIN, HOP = 2.0, 1.0
MIN_DUR = 1.5

T_AUTO = 0.44              # 静默落名门槛
T_ASK = 0.36               # 提问下沿
MARGIN = 0.05              # 次优差；库 <5 人时恒真，仅作保险
RHO_MIN = 0.70             # 一致窗比例；只降级不升级
RHO_WIN_GATE = 0.36        # 统计 ρ/ρ2 时认定「这窗算数」的门槛；与 RHO_MIN 成对改
N_MIN_ASK = 30             # 出卡下限（窗）＝ 30 秒有效语音
N_MIN_AUTO = 60            # 静默下限（窗）＝ 60 秒有效语音；只防纯噪声，不保证正确性
WIN_PASS_MIN = 0.40        # 新人卡信号质量下限
CAP = 1200                 # windows_of 的截断上限；win_pass_rate 的分母也必须用它封顶
NEW_PER_REC_MAX = 3        # 新人卡张数上限（超出折叠，仍可展开认领，不丢弃）
NEW_MIN_SEC = 60           # 新人簇最短时长

# ── 原型质量分级 [PROVISIONAL] ──────────────────────────────────────────
# 这三个数没有实测标定，是按「张彤 42 窗编码的是信道不是人」这一个案例外推的。
# N_PROTO_EXCLUDE 只该剔除「样本量根本不够排名」的人；不要写成 sources<2 就剔除——
# 现有三人全是单来源，那样库会瞬间归零。无人能当 top1 时全库都变「库里没这人」。
N_PROTO_EXCLUDE = 50     # [PROVISIONAL] 原型窗数 < 50 → 不参与 top1 排名，但仍留在人工候选
N_PROTO_MIN = 100        # [PROVISIONAL] < 100 → 可当 top1，禁静默
N_SOURCE_MIN = 2         # [PROVISIONAL] < 2 → 可当 top1，禁静默（单来源无法与信道解耦）

# 入库前的纯度门槛。**和 RHO_MIN 不是同一个量，别混用**（混过一次，见下）：
#   RHO_MIN  = 窗「指向同一个人」的比例，判的是查询簇像不像库里某人
#   PURITY_MIN = 把簇的窗按时间切成前后两半，两半互相比的中位余弦，
#                判的是「这一段前后是不是同一个人」
# 实测（本机，同一段 24 分钟录音）：
#   单人簇 A 0.584 / 单人簇 B 0.694 / 人工 50:50 混合 0.448
# 所以 0.52 落在缝里。**只有 2 个纯样本 + 1 个合成混合样本**，属于 [PROVISIONAL]。
# 它是次要把关：真正防「自己给自己作证」的是打分时剔除同录音来源，
# 不是这道门（面试那次误判的簇纯度是 0.641，这道门拦不住）。
PURITY_MIN = 0.52          # [PROVISIONAL]

SEC_RULE_FACTOR = 0.3      # 出卡门槛 = min(SEC_RULE_CAP, 录音总长 × 本值)
SEC_RULE_CAP = 60          # 只对出卡生效，不约束静默

MIN_RMS_ABS_DBFS = -50     # 能量门限的绝对下限
RELEASE_MEDIAN_DB = 20     # 相对整段电平中位数的动态范围

# 塌陷：整段录音层面的失效，与 win_pass_rate 无关（后者测不到它）
COLLAPSE_SHARE = 0.70
COLLAPSE_MIN_CLUSTERS = 3
OVERLAP_ALERT = 0.15       # 同时间段多簇覆盖占比超过此值 → is_overlap（录音级预警）

# 自动聚类后的合并：分离器常把一个人切成 2~3 个簇（换语气、换位置）。
# 质心余弦高于此值就并成一个簇。0.85 只并「几乎肯定同一个人」的，
# 低于这个值宁可留着让上层按 rho2 报「抢话」。
CLUSTER_MERGE_COS = 0.85

# ── 流水线 ──────────────────────────────────────────────────────────────
THREADS = 4                # 实测 4 最快，8 慢 20%，10 慢 2.4 倍
STABLE_SEC = int(os.environ.get("VP_STABLE_SEC", "60"))  # 文件 size+mtime 连续这么多秒不变才认为同步完成
LONG_AUDIO_SEC = 1800      # >30 分钟只打 long_audio_unreliable 标记，MVP 不切段
VP_SEG_MIN_SEC = 1500      # 标定之后再启用的分流阈值（当前不生效）
SEG_LEN, SEG_HOP = 360, 300

# ── 待确认稿 ────────────────────────────────────────────────────────────
UNKNOWN = "未知"
CLIP_SEC, CLIP_N, CLIP_GAP = 5.0, 3, 0.25
SAMPLE_K, SAMPLE_MAX_CHARS = 3, 30


def card_gate(dur_total):
    """出卡门槛：时长下限 = min(60, 录音总长 × 0.3)。只对出卡生效，不约束静默。"""
    return min(SEC_RULE_CAP, dur_total * SEC_RULE_FACTOR)


def health():
    """/health 与前端文案的唯一数值来源。JS 里不许写死 0.44/30 这类数。"""
    return {
        "scorer": f"{SCORER}@{WIN}/{HOP}",
        "T_AUTO": T_AUTO, "T_ASK": T_ASK, "MARGIN": MARGIN,
        "RHO_MIN": RHO_MIN, "RHO_WIN_GATE": RHO_WIN_GATE,
        "ask_min_seconds": float(N_MIN_ASK), "auto_min_windows": float(N_MIN_AUTO),
        "win_pass_min": WIN_PASS_MIN, "cap": CAP,
        "new_per_rec_max": NEW_PER_REC_MAX, "new_min_sec": NEW_MIN_SEC,
        "db": DB_DIR,
    }


def env_overrides():
    """旧名保留，等同 health()。"""
    return health()
