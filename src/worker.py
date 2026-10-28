#!/usr/bin/env python3
"""常驻 worker：单任务、串行、子进程执行、可恢复。

两条设计都是被实测逼出来的：
  · **串行**：60 分钟音频解码成 float32 约 230MB，两个并发就是 460MB；实测 4 线程已是最优，
    并发只会让两件事都变慢。
  · **子进程**：分离与打分吃满 CPU，放在线程里会把 GIL 攥住，HTTP 整个卡死——
    处理期间点开确认页只会一直转圈。单独进程还顺便隔离了 OOM 和崩溃。
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import traceback

import config
import store
import watch

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ONE = os.path.join(HERE, "run_one.py")
_STOP = threading.Event()
_WAKE = threading.Event()


def _log_path(rid):
    d = os.path.join(config.WORK_DIR, rid)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "worker.log")


def run_one(rid):
    """同步跑一个任务（内部起子进程），返回是否成功。"""
    rec = store.get(rid)
    if not rec:
        return True
    path = rec.get("path")
    if not path or not os.path.exists(path):
        store.update(rid, status=store.ST_FAILED, error="文件不在了")
        return False

    store.update(rid, status=store.ST_PROCESSING, stage="spawn", progress=1, error=None)
    log = _log_path(rid)
    t0 = time.time()
    with open(log, "ab") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {rid} ===\n".encode())
        f.flush()
        try:
            p = subprocess.run([sys.executable, RUN_ONE, rid],
                               cwd=HERE, stdout=f, stderr=subprocess.STDOUT,
                               timeout=3600, env=dict(os.environ))
            code = p.returncode
        except subprocess.TimeoutExpired:
            code = -9
            f.write("\n[worker] 超时 3600s，已终止\n".encode())

    if code == 0:
        return True

    attempts = int((store.get(rid) or {}).get("attempts", 0)) + 1
    tail = ""
    try:
        with open(log, "rb") as f:
            tail = f.read()[-1500:].decode("utf-8", "ignore")
    except Exception:
        pass
    store.update(rid, status=store.ST_FAILED, attempts=attempts,
                 error=f"子进程退出码 {code}，用时 {time.time()-t0:.0f}s",
                 trace=tail)
    if attempts >= 3:
        import notify
        notify.send("录音处理失败", f"{(rec.get('title') or rid)[:40]}（已重试 {attempts} 次）")
    return False


def loop():
    """主循环。失败必须退避——否则一个坏文件（比如状态文件写不进去）会让
    循环每 0.3 秒抛一次异常，一分钟刷出 20 万字符日志，而队列永远不动。"""
    fails, last_msg = 0, ""
    while not _STOP.is_set():
        rid = watch.next_queued()
        if not rid:
            _WAKE.wait(3)
            _WAKE.clear()
            continue
        try:
            ok = run_one(rid)
            fails, last_msg = 0, ""
            if not ok and fails == 0:
                pass                      # run_one 内部已记录 error，扫描器不会自动重排队
        except Exception as e:
            fails += 1
            msg = f"{type(e).__name__}: {e}"
            if msg != last_msg:
                print(f"[worker] 第 {fails} 次连续失败 {msg}", flush=True)
                traceback.print_exc()
                last_msg = msg
            let = min(60.0, 3.0 * (2 ** min(fails, 5)))
            time.sleep(let)
        finally:
            time.sleep(0.3)


def start():
    t = threading.Thread(target=loop, name="vp-worker", daemon=True)
    t.start()
    return t


def stop():
    _STOP.set()
    _WAKE.set()


def kick():
    """用户点了「重新处理」时叫醒 worker。"""
    _WAKE.set()


if __name__ == "__main__":
    # 手动跑一段：python worker.py <rid>
    if len(sys.argv) > 1:
        sys.exit(0 if run_one(sys.argv[1]) else 1)
    print("用法：worker.py <rid>（常驻模式由 app.py 启动）")
