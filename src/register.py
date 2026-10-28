#!/usr/bin/env python3
"""注册一个人：把「某段录音里的某个簇」登记成某人的声纹来源。

用法:
  # 第一次注册（会自动创建 speakers.json）
  python register.py --name 张三 --wav a.wav --segs a.segs.json --cluster 0

  # 同一个人可以登记多条来源（不同录音、不同设备），原型窗会自动合并
  python register.py --name 张三 --wav b.wav --segs b.segs.json --cluster 2

  # 查看当前已登记的人
  python register.py --list

注册完会清掉 winmatch 的原型缓存，下次打分自动重建。
"""
import argparse
import json
import os

import winmatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--wav")
    ap.add_argument("--segs")
    ap.add_argument("--cluster", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    path = winmatch.SPEAKERS
    db = winmatch.load_sources(path)

    if a.list:
        if not db:
            print(f"还没有任何注册记录：{path}")
        for name, parts in db.items():
            print(f"{name}: {len(parts)} 条来源")
            for p in parts:
                print(f"    {os.path.basename(p['wav'])}  簇 {p.get('cluster', 0)}")
        return

    if not (a.name and a.wav and a.segs):
        ap.error("需要 --name --wav --segs（或用 --list 查看）")
    for f in (a.wav, a.segs):
        if not os.path.exists(f):
            raise SystemExit(f"文件不存在：{f}")

    db.setdefault(a.name, []).append(
        {"wav": os.path.abspath(a.wav), "segs": os.path.abspath(a.segs), "cluster": a.cluster})

    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(db, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"已登记 {a.name} → {path}")

    if os.path.exists(winmatch.CACHE):
        os.remove(winmatch.CACHE)
        print("已清除原型缓存，下次打分自动重建")


if __name__ == "__main__":
    main()
