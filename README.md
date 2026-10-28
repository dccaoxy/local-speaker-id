# 本地离线说话人识别工具链

把一段录音变成「**谁、在什么时候、说了什么**」的逐句稿。全程本地运行，不联网、不上传、不花钱。

适合：会议录音、访谈、电话记录的事后整理。
不适合：需要 100% 准确率的场景，以及实时转写。

---

## 现在到底能做什么

| 能力 | 状态 | 说明 |
|---|---|---|
| 说话人分离（谁在什么时候说） | ✅ 可用 | 纯本地，25 分钟音频约 1 分钟处理完 |
| 给每个说话人贴名字 | ⚠️ 部分可用 | 已注册且音质够好时自动识别，否则标「未知」 |
| 跨录音自动认人 | ⚠️ 有条件下可行 | 实测 4 个样本全判对，但标注样本太少，未达交付级 |
| 逐句归属文字 | ✅ 可用 | 需要一个带毫秒时间戳的转写稿（见下） |

**它不做什么**：不自己转写。文字要么来自你已有的转写稿，要么接 macOS 上的 Magic 录音笔（讯飞 ASR）。本仓库不含 ASR 模型。

---

## 自动跑（现在的主用法）

录音丢进目录，剩下的不用管：

```bash
bash scripts/run.sh --check      # 先看环境：用的哪个 Python、模型在不在、库里几个人
bash scripts/run.sh              # 启动，浏览器打开 http://127.0.0.1:8765

# 装成登录自启的常驻服务（launchd）
VP_PY=/path/to/venv/bin/python bash scripts/install_agent.sh
```

**不要直接 `python src/app.py`**：系统 Python 里没有 `sherpa_onnx`，报的却是
`No module named 'fastapi'`，看起来像依赖没装，实际是解释器选错了。`run.sh` 会按
`VP_PY` → 仓库内 `.venv` → 项目 venv → `python3` 的顺序，挑第一个真能 import 的。

它每 5 秒扫一次 `~/OneDrive/数据库/录音/`（可用 `VP_RECORDINGS` 改），
文件「大小和修改时间连续 60 秒不变」才认为同步完成，然后排队处理：

```
转码 → 分离说话人 → 打分 → 切 15 秒试听 → 出稿
```

- 全部认出来的：**不通知你**，直接出 `.说话人.md`
- 有拿不准的：发一条通知，点开是一张卡片 —— 15 秒试听 + 3 句原文 + 几个按钮
- 你点一下，稿子立刻重渲染（不重跑分离）；选「记住他」才写入声纹库

界面里**不显示任何相似度数字**，理由见 `docs/交互方案.md`：用户会对 0.41 和 0.44 的关系作出错误推断，
而一旦他因为一个错误推断吃了亏，整页数字就都不可信了。

数据落在哪（都不进 OneDrive）：

| 东西 | 位置 |
|---|---|
| 声纹向量、原型、版本 | `~/.voiceprint/db/` |
| 状态（可重建） | `~/.voiceprint/state.json` |
| 人工判定（不可重建，只追加） | `~/.voiceprint/decisions.jsonl` |
| 试听切片（缓存在本地） | `~/Library/Caches/voiceprint/` |
| 交付物 | 录音同目录的 `xxx.说话人.md` |

---

## 实测数据（都是真跑出来的，不是论文指标）

核心指标是「**同一个人**的得分」与「**最近的不同人**」之间差多少。差值太小就没法用。

| 取样方式 | 同人得分 | 最近异人 | 间隔 |
|---|---|---|---|
| 整段取均值（旧做法） | 0.487 | 0.468 | **+0.019**（等于没有） |
| 2 秒滑窗 + 最近邻 + 中位数（本项目） | 0.487 | 0.376 | **+0.111** |

跨录音验证（注册样本来自 A 会议，待测样本来自 B 手机录音，互不重叠）：

| 待测样本 | 得分 | 判定 | 是否正确 |
|---|---|---|---|
| 本人 | 0.456 | 命中 | ✅ |
| 陌生同事甲 | 0.345 | 未知 | ✅ |
| 陌生同事乙 | 0.264 | 未知 | ✅ |
| 一次性人物 | 0.261 | 未知 | ✅ |

端到端：一段 25 分钟的多人通话，从未参与过注册，自动判出「本人」所在的那一簇（0.459），并把另一个簇正确地判为未知。

---

## 安装

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

下载两个模型（约 33MB，放在同一个目录）：

```bash
mkdir -p ~/.voiceprint/models && cd ~/.voiceprint/models

# 说话人分离（pyannote segmentation）
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar xjf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

# 声纹嵌入（3D-Speaker CAM++ 中文）
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
```

模型目录、声纹库目录、工作目录都可以用环境变量改：

```bash
export VP_MODEL_DIR=~/.voiceprint/models
export VP_DB=~/.voiceprint/db
export VP_WORK=~/.voiceprint/work
export VP_SPEAKERS=~/.voiceprint/speakers.json
```

---

## 用法

### 第一步：分离，拿到分段结果

```bash
python src/diarize.py 会议.mp3 out/会议.segs.json 3 0.5 4
#                    音频      输出              人数 阈值 线程
```

人数拿不准时可以试 2/3/4 各跑一次，看哪个簇长分布最合理。线程数 **4 最快**（实测 8 线程反而慢 20%，10 线程慢 2.4 倍）。

### 第二步：注册声纹（一次性）

先要有一份 `segs.json`，并且知道「哪个簇是谁」。

```bash
python src/sample.py out/会议.segs.json 会议转写.md 抽样.md   # 按簇抽样句子，人眼认人
python src/register.py --name 张三 --wav 会议.wav --segs out/会议.segs.json --cluster 0
python src/register.py --list
```

同一个人的多条来源（不同录音、不同设备）可以重复登记，原型会自动合并。

### 第三步：识别新录音

```bash
python src/identify.py 新录音.mp3 --clusters 3 --asr 转写.json --out 结果.md
```

- `--asr` 支持三种格式：带 `start/end/text` 的 json（推荐）、`- [0:6.160,0:10.935,0] 文字`、`- 11:10 说话人1 文字`
- 判不出来的簇会标「未知」，不会瞎猜
- 已知谁是谁时，直接点名跳过匹配：`--names "0=张三,1=李四"`

### 可选：从 Magic 录音笔取逐字稿（macOS）

```bash
python src/magic_bridge.py list 10
python src/magic_bridge.py dump <documentId> out/转写.json
```

Magic 录音笔是讯飞出的 Electron 应用，本机装了才有用。它只提供毫秒级逐字稿，**不含声纹能力**，声纹匹配仍由本项目完成。

---

## 三个关键设计（都是踩坑换来的）

**1. 滑窗，不要取均值。**
把 1.5～147 秒的整段音频平均成一个向量，长段里只要有两个人，结果就是「谁都不像」的混合物。改成 2 秒窗后，每个窗只含一个人，坏窗可以被中位数丢掉。

**2. 统计量用中位数，不要用 top-K。**
一开始取「最像的 10 个窗的平均」，结果几个撞上的窗就能把分数抬到 0.885，全是假高分。中位数把这个噪声压到 0.24。

**3. 分离和匹配要用不同预处理。**
分离跑 8k 带宽限制过的音频（聚类更稳，簇数从「1 大 1 小」变成 2 个均衡簇），匹配跑原始 16k（区分度更高）。用同一份预处理会两头不讨好。

另外：一次录音里**一个人只能占一个簇**（互斥分配）。否则两个簇会同时认领同一个人，其中分数低的那个必然是错的。

---

## 目录结构

```
src/
  app.py          本地网页服务（127.0.0.1），确认页面 + 接口
  watch.py        扫目录、判断同步是否完成、入队
  worker.py       单线程串行处理（并发会 OOM，也不会更快）
  pipeline.py     转码 → 分离 → 合并 → 打分 → 切片 → 出稿
  winmatch.py     滑窗声纹打分 + 判定序列（核心）
  protolib.py     来源级原型库：版本管理、跨录音自检、撤销
  enroll.py       入库（带纯度检查）
  store.py        state.json + decisions.jsonl
  render.py       从台账渲染 .说话人.md
  clips.py        预切 15 秒试听音频
  identify.py     命令行版一键流程
  register.py     命令行版注册
  asr.py          三种转写稿格式的统一解析
  diarize.py      说话人分离（独立小工具）
  magic_bridge.py 从 Magic 录音笔取逐字稿（可选，macOS）
web/              确认页面（原生 JS，无构建步骤）
scripts/
  e2e_check.py    端到端自检（真音频真模型）
  install_agent.sh 装/卸 launchd 常驻服务
  sentinel.py     菜单栏待确认数（可选，需 rumps）
docs/
  产品口径-终版.md  唯一数字来源：阈值、判定序列、指标
  应用设计裁决.html 五个角色讨论后的裁决与驳回理由
  交互方案.md      界面形态、文案表、异常态
  应用架构方案.md  进程、IPC、数据模型、状态机
  技术结论.md      完整实验记录，含所有失败的尝试
  踩坑清单.md      同类项目容易踩的坑
```

---

## 已知问题

- **长会议分离会塌**：100 分钟的会，4 个簇里有一簇吞掉 80 分钟。需要改成分段处理 + 跨段合并聚类。
- **阈值样本不足**：0.40 这个阈值只用 4 个标注点标定，最近的一次假阳性距阈值仅 0.055。样本攒多了应该重标。
- **跨设备未充分验证**：注册用宽带会议录音、识别用窄带手机录音时，分数会整体下移。当前靠阈值硬挡。
- **短句不判**：1.5 秒以内的片段不提取声纹。
- **静默落名初期是关着的**：它要求「原型 ≥100 窗 + ≥2 段独立录音 + 跨录音自检通过」。
  三条同时满足前，所有判定都会出卡问你。这是有意的——实测过一个 42 窗、单来源的原型，
  在一段录音里把 3 个互斥的人全认成了同一个人（得分 0.57~0.61，一致率 0.99）。
- **同一段录音里的同一个人，在那段录音里认不出来**：打分时会把「来源 = 这段录音」的原型剔除，
  否则一段录音会用自己的注册窗给自己作证。代价是：注册音频本身不适合用来验证。
- **重叠说话**：两人抢话时分离会出错，这是分割模型的上限。

---

## 隐私

本仓库**不含任何真实录音、声纹向量、人名或转写内容**。示例配置里的「张三/李四」是占位符。

声纹向量本身属于敏感个人信息（生物识别信息），如果你要处理他人语音，注意取得同意——这一点在国内是有法律要求的，别当技术问题处理。

## 许可

MIT
