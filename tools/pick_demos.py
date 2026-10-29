"""本評価の結果から動画にする対を選ぶ。

    python tools/pick_demos.py outputs/colab/main_libero_spatial ...

同じ初期状態・同じ妨害で「素の VLA は失敗、監視ありは成功」した対（救済）と、
監視ありが HELP で止まった例を列挙する。各行の (suite, task, init, disturb) を
run_eval の --tasks / --init-offset / --episodes 1 / --disturb に渡せば同じ条件で撮り直せる。
"""
import json
import sys
from pathlib import Path


def main(dirs):
    rows = [json.loads(l) for d in dirs for l in (Path(d) / "results.jsonl").read_text().splitlines()]
    by = {}
    for r in rows:
        by.setdefault((r["suite"], r["task_id"], r["init_id"], r["disturb"]), {})[r["cond"]] = r
    rescued, helped = [], []
    for k, v in sorted(by.items()):
        b, m = v.get("base"), v.get("monitor")
        if not (b and m):
            continue
        if not b["success"] and m["success"]:
            rescued.append((k, m["steps"], [e["event"] for e in m["events"]]))
        if m["help"]:
            helped.append((k, m["steps"], b["steps"]))
    print("== 救済（素の VLA 失敗 → 監視あり成功）: suite task init disturb | 監視ありの steps | 事象")
    for k, s, ev in rescued:
        print(f"  {k[0]} {k[1]} {k[2]} {'on ' if k[3] else 'off'} | {s} | {ev}")
    print("== HELP で止まった例: suite task init disturb | 監視ありの steps | 素の VLA の steps")
    for k, s, bs in helped[:10]:
        print(f"  {k[0]} {k[1]} {k[2]} {'on ' if k[3] else 'off'} | {s} | {bs}")


if __name__ == "__main__":
    main(sys.argv[1:])
