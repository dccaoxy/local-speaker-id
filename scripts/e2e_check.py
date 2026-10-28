#!/usr/bin/env python3
"""端到端自检：一段真音频走完 扫描 → 分离 → 打分 → 切片 → 出卡 → 确认 → 重渲染 → 入库 → 撤销。

这是脚本不是 pytest 用例：它需要真音频和真模型（几十秒），不适合放进每次跑的单测。
用法：
    VP_PY=.vpvenv/bin/python scripts/e2e_check.py 某段录音.mp3

跑完会在临时目录里留一份 md，并打印每一步的检查结果。任何一步失败就非零退出。
"""
import json
import os
import shutil
import subprocess
import sys
import time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{'' if cond else '  ' + str(detail)}")
    if not cond:
        FAILED.append(name)
    return cond


def main(audio, seconds=150):
    # 固定根目录 + 时间戳：TMPDIR 不可用时 mkdtemp 会退到当前目录乱建随机目录，
    # 而这里需要的只是一个「不会撞车的空目录」。
    tmp = os.path.join(ROOT, ".e2e", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(tmp, exist_ok=True)
    root = os.path.join(tmp, "root")
    rec = os.path.join(tmp, "rec")
    os.makedirs(rec, exist_ok=True)
    # 环境必须在 import config 之前落定，否则拿到的还是真实库
    os.environ.update({"VP_ROOT": root, "VP_RECORDINGS": rec, "VP_STABLE_SEC": "0",
                       "VP_DB": os.path.join(root, "db"),
                       "VP_WORK": os.path.join(root, "work")})

    clip = os.path.join(rec, "e2e 测试 两人对话.wav")
    # 先整段转成 16k wav，再切前 N 秒：afconvert 没有「切时长」这类参数，
    # 硬塞 -t 会被当成容器类型，结果是「命令成功、文件没生成」。
    full = os.path.join(tmp, "full16.wav")
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                    audio, full], capture_output=True, check=False)
    if os.path.exists(full):
        import soundfile as sf
        a, sr = sf.read(full, dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        sf.write(clip, a[: int(seconds * sr)], sr)
    check("切出测试音频", os.path.exists(clip), clip)

    import config
    import store
    import watch
    import pipeline
    import protolib
    import render as render_mod
    from fastapi.testclient import TestClient
    import app as appmod

    print(f"临时状态目录 {root}")
    check("状态不落在 OneDrive", "OneDrive" not in config.STATE, config.STATE)

    rids = watch.scan(now=1000)
    rids += watch.scan(now=2000)
    check("扫描发现新录音", len(rids) == 1, rids)
    rid = rids[0]

    rec_meta = store.get(rid)
    check("状态为 queued", rec_meta.get("status") == store.ST_QUEUED, rec_meta.get("status"))

    out = pipeline.process(clip, rid)
    check("分离出 ≥1 个簇", len(out["clusters"]) >= 1, len(out["clusters"]))
    check("每簇都有 reason", all(c.get("reason") for c in out["clusters"]),
          [c.get("reason") for c in out["clusters"]])
    check("接口不带分数", all("s1" not in c and "rho" not in c for c in out["clusters"]))
    check("有评分台账 _s", all("_s" in c for c in out["clusters"]))
    clips_ok = [c for c in out["clusters"] if c.get("clip")]
    check("至少一个簇切出了试听音频", bool(clips_ok), len(clips_ok))

    store.update(rid, status=store.ST_REVIEW, clusters=out["clusters"], segs=out["segs"],
                 duration=out["duration"], n_clusters=out["n_clusters"],
                 warnings=out["warnings"], overlap=out["overlap"],
                 md_path=out["md_path"], title=out["title"])
    pipeline.render_rec(store.get(rid))
    md = open(out["md_path"], encoding="utf-8").read()
    check("出稿", "# 说话人识别结果" in md)
    check("未确认时用「未知」", "未知" in md)

    with TestClient(appmod.app) as cli:
        r = cli.get("/health")
        check("/health 200", r.status_code == 200)
        h = r.json()
        check("/health 吐门槛", h["T_AUTO"] == config.T_AUTO and "ask_min_seconds" in h)

        r = cli.get("/api/recordings")
        check("列表含该录音", any(i["rid"] == rid for i in r.json()["items"]))

        r = cli.get(f"/api/recordings/{rid}")
        d = r.json()
        check("详情返回卡片", len(d["clusters"]) == len(out["clusters"]))
        check("详情不带分数", all("s1" not in c for c in d["clusters"]))
        cid = d["clusters"][0]["cluster_id"]

        if clips_ok:
            r = cli.get(f"/clip/{rid}/{cid}.wav")
            check("试听音频可取", r.status_code == 200 and len(r.content) > 1000,
                  r.status_code)

        # 1) 落名（不入库）
        r = cli.post("/api/decisions", json={"rid": rid, "cluster_id": cid,
                                             "verdict": "name", "name": "测试甲"})
        check("落名 200", r.status_code == 200, r.text[:200])
        md = open(out["md_path"], encoding="utf-8").read()
        check("稿里出现粗体人名", "**测试甲**" in md, md[:200])

        # 2) 撤销
        r = cli.post("/api/undo", json={"rid": rid, "cluster_id": cid})
        check("撤销 200", r.status_code == 200, r.text[:200])
        md = open(out["md_path"], encoding="utf-8").read()
        check("撤销后人名消失", "测试甲" not in md.split("## 聚簇判定")[1][:400], md[:400])
        check("撤销是追加不是删除",
              any(dd.get("kind") == "revoke" for dd in store.all_decisions()))

        # 3) 新人入库
        before = len(protolib.list_sources())
        r = cli.post("/api/decisions", json={"rid": rid, "cluster_id": cid,
                                             "verdict": "new", "name": "测试甲",
                                             "persist": True})
        body = r.json() if r.status_code == 200 else {"detail": r.text}
        after = len(protolib.list_sources())
        enrolled = after == before + 1
        print(f"     入库结果：{json.dumps(body.get('enroll'), ensure_ascii=False)[:220]}")
        # 混人簇会被纯度门挡下，那也是正确行为——只要「拒绝时说明了原因」
        check("入库要么成功要么给出原因",
              enrolled or (body.get("enroll") or {}).get("error"), body)
        if enrolled:
            md = open(out["md_path"], encoding="utf-8").read()
            check("入库后稿里仍是对勾人名", "测试甲" in md)
            man = protolib.read_manifest()
            check("manifest 记了来源录音",
                  any(p.get("recordings") for p in man.get("people", {}).values()), man)

    print(f"\n产物：{out['md_path']}")
    print(f"临时目录：{tmp}（可删）")
    if FAILED:
        print(f"\n{len(FAILED)} 项失败：{FAILED}")
        return 1
    print("\n端到端自检全部通过")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("用法：scripts/e2e_check.py <音频> [秒数]")
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 150))
