"""監視層あり・なしを同じ初期状態で走らせて比べる。

    python -m runner.run_eval --suite libero_spatial --tasks 0-9 --episodes 5 \
        --conditions base,monitor --disturb off,on --out outputs/spatial

条件:
  base     素の VLA（監視層は影のモードで検知だけ記録し、介入しない）
  monitor  監視層あり（検知 → 退避 → ずらして再開、上限を超えたら HELP）
  disturb  on のとき、ロボットが物を持ち上げた瞬間に物を元の場所へ戻す（1 回）

結果は <out>/results.jsonl に 1 エピソード 1 行で追記する。同じキーの行があれば飛ばすので、
Colab が切れても同じコマンドで続きから再開できる。step ごとの記録は <out>/steps/*.npz。
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from monitor.detectors import DetectorConfig, gripper_width
from monitor.recovery import RecoveryConfig
from monitor.wrapper import MonitorConfig, MonitoredController

from .libero_setup import prepare_libero
from .overlay import FrameComposer, Mp4Writer
from .sim_tools import Disturber, holding_gt, robosuite_env

STICKY = 25  # 検知した事象の表示を、回復動作の間この step 数だけ残す（動画で読めるように）


def parse_ids(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def run_episode(venv, bundle, *, cond: str, disturb: bool, init_id: int, seed: int, max_steps: int,
                det_cfg: DetectorConfig, rec_cfg: RecoveryConfig, max_retries: int,
                video_path: Path | None = None) -> tuple[dict, dict]:
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION

    import torch

    torch.manual_seed(seed)  # SmolVLA の生成（flow matching）の乱数も条件間でそろえる
    le = venv.envs[0].unwrapped
    le.init_state_id = init_id  # 条件をまたいで同じ初期状態から始める
    obs, _ = venv.reset(seed=[seed], options={NEW_ROLLOUT_OPTION: True})
    rs = robosuite_env(venv.envs[0])
    task = le.task_description

    ctrl = MonitoredController(
        policy_fn=lambda o: bundle.act(o, task),
        policy_reset_fn=bundle.reset,
        cfg=MonitorConfig(enabled=(cond == "monitor"), max_retries=max_retries),
        det_cfg=det_cfg, rec_cfg=rec_cfg, seed=seed,
    )
    dist = Disturber(rs, rng=np.random.default_rng(seed)) if disturb else None

    writer = composer = None
    if video_path is not None:
        title = "VLA + monitor" if cond == "monitor" else "VLA only"
        composer = FrameComposer(title, task)
        writer = Mp4Writer(video_path)

    log = {k: [] for k in ("width", "settled", "closing", "gt_holding", "detecting", "status", "event")}
    events, success, t = [], False, 0
    last_evt, last_evt_t = None, -10**9
    t_policy = t_env = 0.0
    try:
        for t in range(max_steps):
            eef = obs["robot_state"]["eef"]["pos"][0]
            gq = obs["robot_state"]["gripper"]["qpos"][0]
            gt = bool(holding_gt(rs))

            t0 = time.perf_counter()
            action, info = ctrl.act(eef, gq, obs)
            t_policy += time.perf_counter() - t0

            # 回復中は検知器を止めているので、開き幅と指令はその時点の観測・行動から直接記録する
            detecting = info["status"] not in ("RECOVERING", "HELP")
            log["width"].append(gripper_width(gq))
            log["settled"].append(ctrl.det.last_settled and detecting)
            log["closing"].append(bool(action[-1] > 0))
            log["gt_holding"].append(gt)
            log["detecting"].append(detecting)
            log["status"].append(info["status"])
            log["event"].append(info["event"])
            if info["event"] != "NONE":
                events.append({"t": t, "event": info["event"], "gt_holding": gt})
                last_evt, last_evt_t = info["event"], t

            note = ""
            if dist is not None and dist.maybe_fire(t):
                note = "disturbance: object taken away"

            if writer is not None:
                shown = info["status"]
                if cond != "monitor":
                    shown = "VLA ONLY"
                    if t - last_evt_t < STICKY:
                        note = note or f"monitor would detect: {last_evt}"
                elif shown == "RECOVERING" and t - last_evt_t < STICKY:
                    shown = last_evt
                if dist is not None and dist.fired_at and t - dist.fired_at[-1] < STICKY:
                    note = note or "disturbance: object taken away"
                writer.add(composer(obs["pixels"]["image"][0], obs["pixels"]["image2"][0], shown, t,
                                    info["retries"], max_retries, note))

            t0 = time.perf_counter()
            obs, _, term, trunc, _ = venv.step(action[None])
            t_env += time.perf_counter() - t0

            if le._env.check_success():
                success = True
                break
            if term[0] or trunc[0]:
                break
            if info["help"]:
                break  # 安全に止まった。以降は時間を使わない
    finally:
        if writer is not None:
            last = "SUCCESS" if success else ("HELP" if ctrl.help else "VLA ONLY" if cond != "monitor" else "NORMAL")
            for _ in range(20):  # 終わりの状態を 1 秒見せる
                writer.add(composer(obs["pixels"]["image"][0], obs["pixels"]["image2"][0], last, t + 1,
                                    ctrl.retries, max_retries))
            writer.close()

    result = {
        "task_id": le.task_id, "task": task, "init_id": init_id, "seed": seed,
        "cond": cond, "disturb": disturb, "success": success, "steps": t + 1,
        "retries": ctrl.retries, "help": ctrl.help, "events": events,
        "disturb_at": dist.fired_at if dist else [],
        "sec_policy": round(t_policy, 2), "sec_env": round(t_env, 2),
    }
    return result, log


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--episodes", type=int, default=5, help="タスクあたりの初期状態の数（0 番から）")
    ap.add_argument("--init-offset", type=int, default=0)
    ap.add_argument("--conditions", default="base,monitor")
    ap.add_argument("--disturb", default="off", help="off / on / off,on")
    ap.add_argument("--max-steps", type=int, default=0, help="0 なら lerobot のスイート既定値")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--empty-width", type=float, default=DetectorConfig.empty_width)
    ap.add_argument("--policy", default=None, help="Hub の repo id かローカルのディレクトリ")
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", default="none", help="none / all / first（タスクごとに最初の初期状態だけ）")
    args = ap.parse_args(argv)

    prepare_libero()
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env

    from .policy import load_policy

    tasks = parse_ids(args.tasks)
    env_cfg = LiberoEnvConfig(task=args.suite, task_ids=tasks)
    # 既定の方策は版を固定して読む。--policy を渡したときは最新版（またはローカル）を読む
    bundle = load_policy(env_cfg, repo_id=args.policy, revision=None) if args.policy else load_policy(env_cfg)
    envs = make_env(env_cfg, n_envs=1)[args.suite]

    out = Path(args.out)
    (out / "steps").mkdir(parents=True, exist_ok=True)
    res_file = out / "results.jsonl"
    done = set()
    if res_file.exists():
        for line in res_file.read_text().splitlines():
            r = json.loads(line)
            done.add(r["key"])

    det_cfg = DetectorConfig(empty_width=args.empty_width)
    rec_cfg = RecoveryConfig()
    (out / "config.json").write_text(json.dumps(
        {"args": vars(args), "detector": asdict(det_cfg), "recovery": asdict(rec_cfg)}, indent=2))

    conds = args.conditions.split(",")
    disturbs = [d == "on" for d in args.disturb.split(",")]
    for tid in tasks:
        venv = envs[tid]
        max_steps = args.max_steps or venv.envs[0].unwrapped._max_episode_steps
        for k in range(args.episodes):
            init_id = args.init_offset + k
            for disturb in disturbs:
                for cond in conds:
                    key = f"{args.suite}_t{tid}_i{init_id}_{cond}_{'d' if disturb else 'n'}"
                    if key in done:
                        continue
                    want_video = args.video == "all" or (args.video == "first" and k == 0)
                    vp = out / "videos" / f"{key}.mp4" if want_video else None
                    t0 = time.time()
                    res, log = run_episode(
                        venv, bundle, cond=cond, disturb=disturb, init_id=init_id, seed=init_id,
                        max_steps=max_steps, det_cfg=det_cfg, rec_cfg=rec_cfg,
                        max_retries=args.max_retries, video_path=vp)
                    res.update(key=key, suite=args.suite, sec_total=round(time.time() - t0, 1))
                    np.savez_compressed(out / "steps" / f"{key}.npz",
                                        **{k2: np.asarray(v) for k2, v in log.items()})
                    with res_file.open("a") as f:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    print(f"{key}: success={res['success']} steps={res['steps']} retries={res['retries']} "
                          f"help={res['help']} events={[e['event'] for e in res['events']]} "
                          f"({res['sec_total']}s, policy {res['sec_policy']}s / env {res['sec_env']}s)", flush=True)
        venv.close()


if __name__ == "__main__":
    main()
