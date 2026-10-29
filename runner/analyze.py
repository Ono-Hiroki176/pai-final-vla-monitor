"""run_eval の出力を集計する。

    python -m runner.analyze outputs/spatial [outputs/object ...]

出すもの:
  1. 条件（base / monitor）× 妨害（なし / あり）ごとの成功率
  2. L1 検知の精度。「閉じて止まった」step ごとに、検知器の判定（開き幅 < 閾値 = 空）と
     正解（シミュレータの接触判定で何も掴んでいない）を突き合わせる。閾値を振って較正にも使う
  3. 事象ごとの的中率（検知した瞬間に、本当に何も掴んでいなかったか）
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(dirs: list[str]) -> tuple[list[dict], dict[str, dict]]:
    results, steps = [], {}
    for d in dirs:
        d = Path(d)
        for line in (d / "results.jsonl").read_text().splitlines():
            r = json.loads(line)
            results.append(r)
            f = d / "steps" / f"{r['key']}.npz"
            if f.exists():
                steps[r["key"]] = dict(np.load(f, allow_pickle=False))
    return results, steps


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def success_table(results: list[dict]) -> list[dict]:
    g = defaultdict(list)
    for r in results:
        g[(r["suite"], r["cond"], r["disturb"])].append(r)
    rows = []
    for (suite, cond, dist), rs in sorted(g.items()):
        k, n = sum(r["success"] for r in rs), len(rs)
        lo, hi = wilson(k, n)
        rows.append({"suite": suite, "cond": cond, "disturb": dist, "n": n, "success": k,
                     "rate": k / n, "ci95": (lo, hi),
                     "help": sum(r["help"] for r in rs),
                     "mean_retries": float(np.mean([r["retries"] for r in rs])),
                     "mean_steps": float(np.mean([r["steps"] for r in rs]))})
    return rows


def paired(results: list[dict]) -> dict:
    """同じ初期状態・同じ妨害での base と monitor の対。救えた数と壊した数を数える。"""
    by = defaultdict(dict)
    for r in results:
        by[(r["suite"], r["task_id"], r["init_id"], r["disturb"])][r["cond"]] = r["success"]
    rescued = broken = both = neither = 0
    for v in by.values():
        if "base" in v and "monitor" in v:
            b, m = v["base"], v["monitor"]
            rescued += (not b) and m
            broken += b and (not m)
            both += b and m
            neither += (not b) and (not m)
    return {"rescued": rescued, "broken": broken, "both": both, "neither": neither}


def l1_sweep(steps: dict[str, dict], thresholds=(0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010)) -> list[dict]:
    w, gt = [], []
    for s in steps.values():
        m = s["settled"].astype(bool)
        w.append(s["width"][m])
        gt.append(s["gt_holding"][m].astype(bool))
    if not w:
        return []
    w, gt = np.concatenate(w), np.concatenate(gt)
    empty = ~gt
    rows = []
    for th in thresholds:
        pred = w < th
        tp = int(np.sum(pred & empty))
        fp = int(np.sum(pred & ~empty))
        fn = int(np.sum(~pred & empty))
        rows.append({"threshold": th, "n_steps": int(len(w)), "n_empty": int(empty.sum()),
                     "precision": tp / (tp + fp) if tp + fp else float("nan"),
                     "recall": tp / (tp + fn) if tp + fn else float("nan")})
    return rows


def width_stats(steps: dict[str, dict]) -> dict:
    w_hold, w_empty = [], []
    for s in steps.values():
        m = s["settled"].astype(bool)
        g = s["gt_holding"].astype(bool)
        w_hold.append(s["width"][m & g])
        w_empty.append(s["width"][m & ~g])
    out = {}
    for name, arr in (("holding", w_hold), ("empty", w_empty)):
        a = np.concatenate(arr) if arr else np.array([])
        out[name] = ({"n": int(a.size), "min": float(a.min()), "p05": float(np.percentile(a, 5)),
                      "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}
                     if a.size else {"n": 0})
    return out


def event_precision(results: list[dict]) -> dict:
    g = defaultdict(lambda: [0, 0])
    for r in results:
        for e in r["events"]:
            key = e["event"]
            g[key][0] += 1
            # L1 は「空なのに閉じている」が正解。L2（停滞・迷走）は正解の定義が無いので数だけ出す
            if key in ("GRASP_MISS", "DROP"):
                g[key][1] += int(not e["gt_holding"])
    return {k: {"n": v[0], "true_empty": v[1] if k in ("GRASP_MISS", "DROP") else None} for k, v in g.items()}


def main(dirs: list[str]) -> dict:
    results, steps = load(dirs)
    report = {
        "success": success_table(results),
        "paired": paired(results),
        "l1_sweep": l1_sweep(steps),
        "width": width_stats(steps),
        "events": event_precision(results),
    }
    print("== 成功率")
    for r in report["success"]:
        lo, hi = r["ci95"]
        print(f"  {r['suite']:<15} {r['cond']:<8} disturb={'on ' if r['disturb'] else 'off'} "
              f"{r['success']:>3}/{r['n']:<3} = {r['rate']:.0%}  (95%CI {lo:.0%}-{hi:.0%})  "
              f"HELP {r['help']}  retries {r['mean_retries']:.2f}  steps {r['mean_steps']:.0f}")
    print("== 対ごとの比較（同じ初期状態）", report["paired"])
    print("== 閉じて止まった時点の開き幅 [m]")
    for k, v in report["width"].items():
        print(f"  {k:<8}", {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    print("== L1 の閾値ごとの精度（正: 何も掴んでいない）")
    for r in report["l1_sweep"]:
        print(f"  th={r['threshold']:.3f}  precision={r['precision']:.3f}  recall={r['recall']:.3f}  "
              f"(steps {r['n_steps']}, empty {r['n_empty']})")
    print("== 検知した事象", report["events"])
    return report


if __name__ == "__main__":
    rep = main(sys.argv[1:])
    out = Path(sys.argv[1]) / "summary.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float))
