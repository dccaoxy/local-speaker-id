#!/usr/bin/env python3
"""纯逻辑单测：判定序列、reason 优先级、数字禁令、is_overlap、转写稿抽样。

判定这块最容易被「改一句 if」悄悄改坏，而它直接决定产品行为，所以锁死在测试里。
系统 python 也能跑（重依赖会被 stub 掉）：
    python3 tests/test_units.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

for name in ("numpy", "sherpa_onnx", "soundfile"):
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:
            stub = types.ModuleType(name)
            stub.__getattr__ = lambda attr, _n=name: (_ for _ in ()).throw(
                AssertionError(f"纯逻辑测试不应触碰 {_n}.{attr}"))
            sys.modules[name] = stub

import config
import winmatch
import asr

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {'' if cond else detail}")
    if not cond:
        FAILED.append(name)


def ev(**kw):
    """「四项全过」的簇做基线，每个用例只改需要的那一项。"""
    base = dict(cluster_id=0, duration_s=300.0, eff_sec=300.0, n_total=300,
                n_theory=299, win_pass_rate=1.0, s1=0.50, s2=0.30, m=0.20,
                rho=0.90, rho2=0.05)
    base.update(kw)
    e = winmatch.ClusterEvidence(**base)
    e.top1 = kw.get("top1", "甲")
    e.second_name = kw.get("second_name", "乙")
    return e


def d(e, dur=None, n_clusters=None, max_share=None):
    return winmatch.decide_from_evidence(e, dur, n_clusters, max_share)


def test_is_overlap():
    print("is_overlap（录音级）")
    segs = [{"start": 0, "end": 20, "speaker": 0}, {"start": 0, "end": 20, "speaker": 1}]
    check("全段重叠=1.0", winmatch.is_overlap_of(segs, 20) == 1.0)
    segs = [{"start": 0, "end": 10, "speaker": 0}, {"start": 0, "end": 10, "speaker": 1},
            {"start": 10, "end": 100, "speaker": 0}]
    check("10/100=0.1 不报警", abs(winmatch.is_overlap_of(segs, 100) - 0.1) < 1e-6)
    segs = [{"start": 0, "end": 30, "speaker": 0}, {"start": 0, "end": 30, "speaker": 1},
            {"start": 30, "end": 100, "speaker": 0}]
    check("30% 超阈值", winmatch.is_overlap_of(segs, 100) > config.OVERLAP_ALERT)
    check("空输入不炸", winmatch.is_overlap_of([], 0) == 0.0)
    segs = [{"start": 0, "end": 10, "speaker": s} for s in (0, 1, 2)]
    check("三重叠不重复计数", abs(winmatch.is_overlap_of(segs, 10) - 1.0) < 1e-6)


def test_order():
    print("七步判定序列（顺序错了等于改了产品定义）")
    e = ev(n_total=20)
    d(e, 600)
    check("1 窗数<30 → few_evidence", e.reason == "few_evidence" and not e.ask)

    e = ev()
    d(e, 600, n_clusters=4, max_share=0.80)
    check("2 塌陷 → isolation_broken 不出卡",
          e.reason == "isolation_broken" and not e.ask)

    e = ev()
    d(e, 600, n_clusters=4, max_share=0.50)
    check("2' 未塌陷时不受影响", e.name == "甲" and e.reason is None)

    e = ev()
    d(e, 600)
    check("3 四项全过 → 静默落名", e.name == "甲" and not e.ask)

    e = ev(rho=0.5)
    d(e, 600)
    check("3' ρ 不够 → 出卡不落名", e.ask and e.name is None)

    e = ev(rho=0.5, rho2=0.30)
    d(e, 600)
    check("4 rho2 高 → mixed", e.reason == "mixed" and e.ask)

    e = ev(rho=0.5, rho2=0.15)
    d(e, 600)
    check("4 rho2 中 → mixed_weak", e.reason == "mixed_weak")

    e = ev(rho=0.5, rho2=0.02)
    d(e, 600)
    check("4 rho2 低 → noisy", e.reason == "noisy")

    e = ev(m=0.01)
    d(e, 600)
    check("4 margin 不够 → low_margin", e.reason == "low_margin")

    e = ev(s1=0.20, s2=0.10, win_pass_rate=0.30)
    d(e, 600)
    check("5 信号糊 → noisy 不出卡", e.reason == "noisy" and not e.ask)

    e = ev(s1=0.20, s2=0.10, win_pass_rate=0.80)
    d(e, 600)
    check("6 分低但干净 → 新人卡 few_match",
          e.reason == "few_match" and e.ask and e.name is None)

    e = ev(duration_s=30.0, eff_sec=30.0, n_total=40, s1=0.20, win_pass_rate=0.8)
    d(e, 600)
    check("7 短簇不到出卡门槛 → too_short", e.reason == "too_short" and not e.ask)

    e = ev(s1=0.20, win_pass_rate=0.8, duration_s=40.0, eff_sec=40.0)
    d(e, 600)
    check("7' 新人簇 <60s 不出卡", not e.ask)

    e = ev(n_total=45, s1=0.50, m=0.20, rho=0.9)
    d(e, 600)
    check("静默需 N≥60：45 窗不静默", e.name is None and e.ask)


def test_reason_priority():
    print("reason 单值 + 优先级短路")
    e = ev(rho=0.5, rho2=0.30, m=0.01)
    d(e, 600)
    check("抢话优先于灰区", e.reason == "mixed")
    e = ev()
    d(e, 600, n_clusters=5, max_share=0.9)
    check("isolation_broken 排最前", e.reason == "isolation_broken")
    for r in ("mixed", "noisy", "mixed_weak", "low_margin", "low_rho",
              "few_match", "too_short", "few_evidence", "isolation_broken"):
        check(f"枚举含 {r}", r in winmatch.REASONS)


def test_api():
    print("接口全禁数字")
    e = ev(rho=0.5, rho2=0.30)
    d(e, 600)
    api = winmatch.to_api(e)
    check("无 sim/second 字段", "sim" not in api and "second" not in api)
    check("无 second_name 字段", "second_name" not in api)
    check("无 rho/n_total 字段", "rho" not in api and "n_total" not in api)
    check("无 eff_sec 字段", "eff_sec" not in api)
    check("mixed 时 top1 置 null", api["top1"] is None)
    check("保留 duration_s 供统计行", "duration_s" in api)

    e = ev(m=0.01)
    d(e, 600)
    check("low_margin 可给 top1", winmatch.to_api(e)["top1"] == "甲")


def test_asr():
    print("转写稿抽样")
    rows = [(0.0, 2.0, "嗯"), (2.0, 6.0, "我今天下午要去见一个客户聊报价"),
            (6.0, 8.0, "好"), (8.0, 12.0, "这个价格报备的流程是这样的你先填表"),
            (100.0, 104.0, "完全不在这段里的句子啊啊啊")]
    got = asr.samples_for([(0.0, 20.0)], rows, k=3, max_chars=30)
    check("只取范围内的句子", all("不在" not in t for t in got))
    check("按字数降序", got == sorted(got, key=len, reverse=True) and got)
    check("1 字句直接丢", all(len(t) > 1 for t in got))
    check("超长截断", all(len(t) <= 31 for t in got))
    check("去重", len(asr.samples_for([(0, 20)], [(0, 1, "重复的话"), (1, 2, "重复的话")], k=3)) == 1)
    check("空输入返回空表", asr.samples_for([], rows) == [])
    check("缺稿不报错", asr.load_asr("/nope/x.md") == [])


def test_health():
    print("/health 供前端读，不许写死")
    h = config.health()
    for k in ("T_AUTO", "T_ASK", "MARGIN", "RHO_MIN",
              "ask_min_seconds", "auto_min_windows"):
        check(f"包含 {k}", k in h)
    check("无 n_min_* 旧名", not any(k.startswith("n_min") for k in h))
    check("单位是秒", h["ask_min_seconds"] == float(config.N_MIN_ASK))


if __name__ == "__main__":
    test_is_overlap()
    test_order()
    test_reason_priority()
    test_api()
    test_asr()
    test_health()
    print()
    if FAILED:
        print(f"{len(FAILED)} 项失败：{FAILED}")
        raise SystemExit(1)
    print("全部通过")
