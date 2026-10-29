"""Colab の VM 上で環境を作る（colab-cli から実行する）。

    colab upload -s pai outputs/pai-final.zip /content/pai-final.zip
    colab exec   -s pai -f tools/colab_bootstrap.py --timeout 1800

やること:
  1. /content/pai-final.zip を /content/pai-final に展開
  2. EGL の vendor 設定を置く（無いと描画がソフトウェアになり、エラーも出ずに極端に遅くなる）
  3. uv で専用 venv（Python 3.12）を作り requirements.txt を入れる（カーネルの Python には触らない）
  4. torch が GPU を見ているか、LIBERO が import できるかを確かめる
"""
import json
import os
import pathlib
import subprocess
import zipfile

ROOT = pathlib.Path("/content/pai-final")
VENV = pathlib.Path("/content/venv")


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


if ROOT.exists():
    sh(f"rm -rf {ROOT}")
with zipfile.ZipFile("/content/pai-final.zip") as z:
    z.extractall(ROOT)
print("unzipped", len(list(ROOT.rglob("*.py"))), "py files")

egl = pathlib.Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
egl.parent.mkdir(parents=True, exist_ok=True)
egl.write_text(json.dumps({"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}, indent=4))
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")
sh("pip -q install uv")
if not (VENV / "bin" / "python").exists():
    sh(f"uv venv -q -p 3.12 {VENV}")
sh(f"uv pip install -q --python {VENV}/bin/python -r {ROOT}/requirements.txt")
sh(f"{VENV}/bin/python -c \"import torch, lerobot, mujoco; "
   f"print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda, "
   f"'lerobot', lerobot.__version__, 'mujoco', mujoco.__version__); "
   f"assert torch.cuda.is_available()\"")
sh(f"cd {ROOT} && {VENV}/bin/python -m pytest -q tests/test_monitor.py")
print("BOOTSTRAP OK", flush=True)
