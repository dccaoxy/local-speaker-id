#!/usr/bin/env python3
"""纯逻辑单测：不依赖 numpy / sherpa-onnx / soundfile。

判定顺序、ρ 语义、白名单、重放、抽样 这几块最容易写错，而且都能在没有模型和
音频的环境里验证，所以把重依赖 stub 掉单独跑：
    python3 tests/test_units.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# ── stub 掉重依赖，只测纯逻辑 ────────────────────────────────────────────
for name in ("numpy", "sherpa_onnx", "soundfile"):
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.__getattr__ = lambda attr, _n=name: (_ for _ in ()).throw(
            AssertionError(f"纯逻辑测试不应触碰 {_n}.{attr}"))
        sys.modules[name] = stub

import config
import winmatch
import asr

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILED.append(name)


def ev(**kw):
    base = dict(cluster_id=0, duration=300.0, n_total=200, n_theory=299,
                s1=0.50, s2=0.30, m=0.20, rho=0.90, rho2=0.05)
    base.update(kw)
    return winmatch.ClusterEvidence(**base)


def test_is_overlap():
    print("is_overlap（录音级）")
    segs = [{"start": 0, "end": 20, "speaker": 0},
            {"start": 0, "end": 20, "speaker": 1}]
    check("全段重叠=1.0", winmatch.is_overlap_of(segs, 20) == 1.0)
    segs = [{"start": 0, "end": 10, "speaker": 0},
            {"start": 0, "end": 10, "speaker": 1},
            {"start": 10, "end": 100, "speaker": 0}]
    check("10/100=0.1 不报警", abs(winmatch.is_overlap_of(segs, 100) - 0.1) < 1e-6)
    segs = [{"start": 0, "end": 30, "speaker": 0},
            {"start": 0, "end": 30, "speaker": 1},
            {"start": 30, "end": 100, "speaker": 0}]
    check("30% 超阈值", winmatch.is_overlap_of(segs, 100) > config.OVERLAP_ALERT)
    check("空输入不炸", winmatch.is_overlap_of([], 0) == 0.0)
    # 三人叠加只算一次，不能重复累计
    segs = [{"start": 0, "end": 10, "speaker": s} for s in (0, 1, 2)]
    check("三重叠不重复计数", abs(winmatch.is_overlap_of(segs, 10) - 1.0) < 1e-6)


def test_decide_order():
    print("判定顺序（顺序错了等于改了产品定义）")
    e = ev(n_total=10)
    winmatch.decide_from_evidence(e, 600)
    check("窗数低于硬底线 → few_evidence", e.reason == ["few_evidence"] and not e.ask)

    e = ev(duration=10.0, n_total=200)
    winmatch.decide_from_evidence(e, 600)
    check("时长不够 → 不出卡", not e.ask and e.name is None and e.reason == ["too_short"])

    e = ev()
    winmatch.decide_from_evidence(e, 600)
    check("四项全过 → 静默落名", e.name == e.top1 and not e.ask and e.tier == "很像")

    e = ev(rho=0.5)
    winmatch.decide_from_evidence(e, 600)
    check("ρ 不够 → 出卡不是落名", e.ask and e.name is None)

    e = ev(m=0.01)
    winmatch.decide_from_evidence(e, 600)
    check("margin 不够 → low_margin", e.reason == ["low_margin"])

    e = ev(rho=0.5, rho2=0.30)
    winmatch.decide_from_evidence(e, 600)
    check("rho2 高 → mixed", e.reason == ["mixed"])

    e = ev(rho=0.5, rho2=0.15)
    winmatch.decide_from_evidence(e, 600)
    check("rho2 中 → mixed_weak", e.reason == ["mixed_weak"])

    e = ev(rho=0.5, rho2=0.02)
    winmatch.decide_from_evidence(e, 600)
    check("rho2 低 → noisy", e.reason == ["noisy"])

    e = ev(rho=0.9, n_total=30)
    winmatch.decide_from_evidence(e, 600)
    check("分数够但窗数不够 → few_windows", e.reason == ["few_windows"] and e.ask)

    e = ev(s1=0.20, s2=0.10)
    winmatch.decide_from_evidence(e, 600)
    check("低于出卡线 → 不出卡（CARD_STRANGER=False）", not e.ask)


def test_whitelist():
    print("数字白名单（接口层兜，不指望 UI 自觉）")
    e = ev(rho=0.5, rho2=0.30)
    e.top1, e.second_name = "甲", "乙"
    winmatch.decide_from_evidence(e, 600)
    api = winmatch.to_api(e)
    check("mixed 不给数字", api["sim"] is None and api["second"] is None)
    check("mixed 不给 second_name", api["second_name"] is None)

    e = ev(m=0.01)
    e.top1, e.second_name = "甲", "乙"
    winmatch.decide_from_evidence(e, 600)
    api = winmatch.to_api(e)
    check("low_margin 给双数字", api["sim"] == 0.5 and api["second"] == 0.3)


def test_asr():
    print("转写稿抽样")
    rows = [(0.0, 2.0, "嗯"), (2.0, 6.0, "我今天下午要去见一个客户聊报价"),
            (6.0, 8.0, "好"), (8.0, 12.0, "这个价格报备的流程是这样的你先填表"),
            (100.0, 104.0, "完全不在这段里的句子啊啊啊")]
    got = asr.samples_for([(0.0, 20.0)], rows, k=3, max_chars=30)
    check("只取范围内的句子", all("不在" not in t for t in got))
    check("按字数降序", got == sorted(got, key=len, reverse=True) and got)
    check("1 字句直接丢（认人无价值）", all(len(t) > 1 for t in got))
    check("超长截断", all(len(t) <= 31 for t in got))
    check("去重", len(asr.samples_for([(0, 20)], [(0, 1, "重复的话"),
                                                  (1, 2, "重复的话")], k=3)) == 1)
    check("空输入返回空表", asr.samples_for([], rows) == [])
    check("缺稿不报错", asr.load_asr("/nope/x.md") == [])


if __name__ == "__main__":
    test_is_overlap()
    test_decide_order()
    test_whitelist()
    test_asr()
    print()
    if FAILED:
        print(f"{len(FAILED)} 项失败：{FAILED}")
        raise SystemExit(1)
    print("全部通过")
