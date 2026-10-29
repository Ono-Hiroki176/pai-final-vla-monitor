"""手元のターミナルから colab-cli で評価を回す。

    python tools/colab_driver.py setup                     # T4 セッション作成 → コード転送 → 環境構築
    python tools/colab_driver.py run calib_spatial --suite libero_spatial --tasks 0-9 --episodes 3 --conditions base
    python tools/colab_driver.py run main_spatial  --suite libero_spatial --tasks 0-9 --episodes 5 --init-offset 5 \
        --conditions base,monitor --disturb off,on --empty-width 0.0034
    python tools/colab_driver.py stop                      # 必ず止める（止めないと CU を消費し続ける）

run はタスク 1 つごとに colab exec を 1 回呼び、終わるたびに結果（results.jsonl と steps/）を
outputs/colab/<名前>/ へ回収する。VM 上の結果は手元の回収分から復元してから走らせるので、
VM が落ちて作り直しても同じコマンドで続きから再開できる。動画は --video を付けたときだけ回収する。

Colab の使い方の約束: 人が見ている前で回す。自動の再接続やカーネルからの切り離しはしない。
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SESSION = "pai"
REMOTE = "/content/pai-final"
LOCAL_OUT = ROOT / "outputs" / "colab"


def colab(*args: str, timeout: float = 600) -> subprocess.CompletedProcess:
    cmd = ["colab", *args]
    print("[driver]", " ".join(shlex.quote(a) for a in cmd), flush=True)
    return subprocess.run(cmd, stdin=subprocess.DEVNULL, timeout=timeout, check=True)


def remote_sh(cmd: str, timeout: float = 3600) -> None:
    colab("exec", "-s", SESSION, "--env", f"CMD={cmd}", "-f", str(ROOT / "tools" / "colab_sh.py"),
          "--timeout", str(int(timeout)), timeout=timeout + 120)


def setup() -> None:
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_zip.py")], check=True)
    colab("new", "-s", SESSION, "--gpu", "T4", timeout=600)
    colab("upload", "-s", SESSION, str(ROOT / "outputs" / "pai-final.zip"), "/content/pai-final.zip")
    colab("exec", "-s", SESSION, "-f", str(ROOT / "tools" / "colab_bootstrap.py"), "--timeout", "1800",
          timeout=2000)


def fetch(name: str, with_video: bool) -> None:
    remote_zip = f"/content/{name}.zip"
    parts = f"{name}/results.jsonl {name}/config.json {name}/steps" + (f" {name}/videos" if with_video else "")
    remote_sh(f"cd outputs && rm -f {remote_zip} && zip -qr {remote_zip} {parts}", timeout=300)
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    local_zip = LOCAL_OUT / f"{name}.zip"
    colab("download", "-s", SESSION, remote_zip, str(local_zip))
    with zipfile.ZipFile(local_zip) as z:
        z.extractall(LOCAL_OUT)


def restore(name: str) -> None:
    """手元に回収済みの結果を VM に戻す（新しい VM で再開するとき用）。"""
    res = LOCAL_OUT / name / "results.jsonl"
    if not res.exists():
        return
    remote_sh(f"mkdir -p outputs/{name}", timeout=120)
    colab("upload", "-s", SESSION, str(res), f"{REMOTE}/outputs/{name}/results.jsonl")


def run(name: str, rest: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--video", default="none")
    known, passthrough = ap.parse_known_args(rest)

    tids: list[int] = []  # run_eval.parse_ids と同じ書式（"0-9" / "0,3,5"）。手元で numpy 等を要さないよう写す
    for part in known.tasks.split(","):
        a, _, b = part.partition("-")
        tids += list(range(int(a), int(b) + 1)) if b else [int(a)]

    restore(name)
    for tid in tids:
        args = ["--tasks", str(tid), "--out", f"outputs/{name}", "--video", known.video, *passthrough]
        remote_sh("/content/venv/bin/python -m runner.run_eval " + " ".join(shlex.quote(a) for a in args),
                  timeout=3000)
        fetch(name, with_video=known.video != "none")
    print(f"[driver] done: {LOCAL_OUT / name}", flush=True)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "setup":
        setup()
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3:])
    elif cmd == "fetch":
        fetch(sys.argv[2], with_video="--video" in sys.argv)
    elif cmd == "stop":
        colab("stop", "-s", SESSION)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
