#!/usr/bin/env python3
"""注册 / 管理声纹库（来源级，带纯度检查）。

  python register.py add --name 张三 --wav a16.wav --segs a.segs.json --cluster 0
  python register.py list
  python register.py status
  python register.py delete --source-id 张三-20261028-... 
  python register.py rebuild

跟旧版的区别：入库前会算「注册片段本身是不是混合物」，不合格直接拒收。
实测教训——把一个混着几个人的簇注册进库，等于给这个人配了一个「谁都像他」的原型，
之后每个陌生人都会被他吸走，而界面上一切正常。
"""
import argparse
import json
import sys

import config
import enroll
import protolib


def cmd_add(a):
    sid, rep = enroll.enroll(a.name, a.wav, a.segs, a.cluster,
                             recording=a.recording or a.wav, strict=not a.force)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if not sid:
        sys.exit(1)
    print("\n" + protolib.status())


def cmd_list(_):
    srcs = protolib.list_sources()
    if not srcs:
        print(f"库里还没有来源（{config.EMB_CACHE_DIR}）")
        return
    for name in sorted({m["name"] for m in srcs.values()}):
        mine = [m for m in srcs.values() if m["name"] == name]
        print(f"{name}: {len(mine)} 条来源，{sum(m['n_win'] for m in mine)} 窗")
        for m in sorted(mine, key=lambda x: x["created_at"]):
            print(f"    {m['source_id']}  {m['n_win']:>5} 窗  "
                  f"录音={m.get('recording') or '—'}  簇={m.get('cluster')}")


def cmd_delete(a):
    ok = protolib.delete_source(a.source_id)
    print("已删除" if ok else "没找到这个来源")
    if ok:
        print("version:", protolib.build_version())
        print(protolib.status())


def cmd_rebuild(_):
    print("version:", protolib.build_version())
    print(protolib.status())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    add = sub.add_parser("add")
    add.add_argument("--name", required=True)
    add.add_argument("--wav", required=True, help="16k 单声道 wav")
    add.add_argument("--segs", required=True, help="分离结果 .segs.json")
    add.add_argument("--cluster", type=int, required=True)
    add.add_argument("--recording", default="", help="这段音频对应的录音 id（剔除同源用）")
    add.add_argument("--force", action="store_true", help="跳过纯度检查（不推荐）")
    add.set_defaults(func=cmd_add)

    sub.add_parser("list").set_defaults(func=cmd_list)
    sub.add_parser("status").set_defaults(func=cmd_rebuild)
    sub.add_parser("rebuild").set_defaults(func=cmd_rebuild)
    d = sub.add_parser("delete")
    d.add_argument("--source-id", required=True)
    d.set_defaults(func=cmd_delete)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
