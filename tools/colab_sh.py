"""colab exec で VM 上のシェルコマンドを走らせ、出力を 1 行ずつ手元に流す。

    colab exec -s pai --env "CMD=..." -f tools/colab_sh.py --timeout 3600
"""
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ["MPLBACKEND"] = "Agg"  # カーネルのノートブック用バックエンドは venv 側に無い
cmd = os.environ["CMD"]
print(f"$ {cmd}", flush=True)
p = subprocess.Popen(cmd, shell=True, cwd="/content/pai-final", stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
print(f"[exit {rc}]", flush=True)
if rc:
    sys.exit(rc)
