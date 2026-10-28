#!/usr/bin/env python3
"""处理一段录音（在独立进程里跑）。

**为什么必须是独立进程而不是线程**：分离与打分是 CPU 密集的 C++/numpy 调用，
放在线程里会把 GIL 攥住，HTTP 事件循环整个卡死——实测处理 24 分钟录音的那两分钟里，
浏览器打开确认页是「一直转圈」。用户看到的就是「这工具死了」。
单独进程还顺带解决两件事：单任务内存峰值随进程退出释放；一段录音把内存吃爆不会带走服务。

用法：python run_one.py <rid>
"""
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config           # noqa: E402
import notify           # noqa: E402
import pipeline         # noqa: E402
import store            # noqa: E402


def _progress(rid):
    def cb(stage, pct):
        store.update(rid, status=store.ST_PROCESSING, stage=stage, progress=pct)
    return cb


def main(rid):
    rec = store.get(rid)
    if not rec:
        print(f"没有这个 rid: {rid}", file=sys.stderr)
        return 2
    path = rec.get("path")
    if not path or not os.path.exists(path):
        store.update(rid, status=store.ST_FAILED, error="文件不在了")
        return 3

    store.update(rid, status=store.ST_PROCESSING, stage="start", progress=1, error=None)
    t0 = time.time()
    try:
        out = pipeline.process(path, rid, clusters=rec.get("clusters_hint"),
                               progress=_progress(rid))
    except Exception as e:
        store.update(rid, status=store.ST_FAILED,
                     error=f"{type(e).__name__}: {e}",
                     trace=traceback.format_exc()[-4000:])
        print(traceback.format_exc(), file=sys.stderr)
        return 1

    store.update(rid, status=store.ST_REVIEW, stage="render", progress=99,
                 duration=out["duration"], n_clusters=out["n_clusters"],
                 clusters=out["clusters"], segs=out["segs"],
                 warnings=out["warnings"], overlap=out["overlap"],
                 md_path=out["md_path"], title=out["title"],
                 elapsed=out["elapsed"], size=rec.get("size"), mtime=rec.get("mtime"))
    rec2 = store.get(rid)
    try:
        pipeline.render_rec(rec2)
    except Exception as e:
        # 出稿失败（最常见是 OneDrive 目录被同步进程锁住）不能作废已经算出来的结果：
        # 重新分离一段 25 分钟录音要几分钟，而稿子随时可以从台账重渲染。
        store.update(rid, md_error=f"{type(e).__name__}: {e}")
        print(f"[run_one] 出稿失败，判定已保存，可稍后重渲染：{e}", file=sys.stderr)

    pend = store.pending_clusters(rid)
    if pend:
        names = "、".join(sorted({c.get("top1") or "陌生人" for c in pend}))
        notify.send(f"{len(pend)} 个说话人待确认",
                    f"{rec2.get('title', '')} · 最像 {names}")
    else:
        store.update(rid, status=store.ST_FINAL)

    print(json.dumps({"rid": rid, "elapsed": round(time.time() - t0, 1),
                      "clusters": len(out["clusters"]),
                      "pending": len(pend)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("用法：run_one.py <rid>")
    sys.exit(main(sys.argv[1]))
