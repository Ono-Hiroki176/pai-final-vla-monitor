"""notebooks/run_colab.ipynb を生成する（セルの中身はここで管理する）。

    python notebooks/build_notebook.py
"""
import json
from pathlib import Path

CELLS = [
("md", """# しくじりに気づくロボット — Colab 実行ノート

公開 VLA（`lerobot/smolvla_libero`）を LIBERO で動かし、監視層あり・なしを比べる。

- ランタイム: **T4 GPU**（1時間あたり約 1.2〜1.8 CU）。本評価の全部で数時間の見込み
- 結果は Google Drive に書くので、切断されても同じセルを実行し直せば続きから再開できる
- **手で実行すること。** 自動クリックでの延命やバックグラウンドへの切り離しはしない"""),
("md", "## 1. GPU と取得"),
("code", """!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/pai_final/outputs'   # 結果の置き場（再開に使う）
REPO_URL = ''   # GitHub に上げたら URL を入れる。空なら Drive の zip から展開する
REPO_ZIP = '/content/drive/MyDrive/pai_final/pai-final.zip'
import os, subprocess
if REPO_URL:
    subprocess.run(['git', 'clone', '-q', REPO_URL, '/content/pai-final'], check=True)
else:
    subprocess.run(['unzip', '-q', '-o', REPO_ZIP, '-d', '/content/pai-final'], check=True)
os.chdir('/content/pai-final')
!ls"""),
("md", """## 2. 環境構築（専用 venv / Python 3.12）

Colab カーネルの Python とは別の venv に入れる（カーネルの numpy や torch を動かさない）。"""),
("code", """!pip -q install uv
!uv venv -q -p 3.12 /content/venv
!uv pip install -q --python /content/venv/bin/python -r requirements.txt
!/content/venv/bin/python -c "import torch, lerobot, mujoco; print(torch.__version__, torch.cuda.is_available(), lerobot.__version__, mujoco.__version__); assert torch.cuda.is_available(), 'GPU が見えない: ランタイムの種類を T4 にする'\""""),
("code", """# EGL の vendor 設定。これが無いと描画がソフトウェアになり、エラーも出ずに極端に遅くなる
import json, pathlib
p = pathlib.Path('/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({'file_format_version': '1.0.0', 'ICD': {'library_path': 'libEGL_nvidia.so.0'}}, indent=4))
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'   # 必ず MUJOCO_GL と揃える
PY = '/content/venv/bin/python'"""),
("md", "## 3. 疎通（1 エピソード、動画つき）"),
("code", """!$PY -m runner.run_eval --suite libero_spatial --tasks 0 --episodes 1 --conditions base,monitor --out $OUT/smoke --video all"""),
("md", """## 4. 較正（素の VLA の走行から、L1 の閾値を決める）

`閉じて止まった時点の開き幅` の holding / empty の分布と、閾値ごとの precision / recall を見る。"""),
("code", """!$PY -m runner.run_eval --suite libero_spatial --tasks 0-9 --episodes 3 --conditions base --out $OUT/calib_spatial
!$PY -m runner.run_eval --suite libero_object --tasks 0-9 --episodes 3 --conditions base --out $OUT/calib_object
!$PY -m runner.analyze $OUT/calib_spatial $OUT/calib_object"""),
("code", """EMPTY_WIDTH = 0.003   # ↑の結果を見て決める（holding の最小値より小さく、empty の大半より大きい値）"""),
("md", """## 5. 本評価

各スイート 10 タスク × 5 初期状態 × {base, monitor} × {妨害なし, あり} = 200 エピソード。
初期状態は 5〜9 番を使う（閾値の較正に使った 0〜2 番とは分ける）。
疎通の行に出る 1 エピソードの秒数 × 400 が所要時間の目安。"""),
("code", """for suite in ['libero_spatial', 'libero_object']:
    !$PY -m runner.run_eval --suite $suite --tasks 0-9 --episodes 5 --init-offset 5 --conditions base,monitor --disturb off,on --empty-width $EMPTY_WIDTH --out $OUT/main_$suite
!$PY -m runner.analyze $OUT/main_libero_spatial $OUT/main_libero_object
!$PY -m runner.plot $OUT/main_libero_spatial $OUT/main_libero_object --out $OUT/success.png
from IPython.display import Image; Image(f'{OUT}/success.png')"""),
("md", "## 6. 動画素材（タスクごとに 1 本、条件 4 通り）"),
("code", """!$PY -m runner.run_eval --suite libero_spatial --tasks 0-9 --episodes 1 --init-offset 5 --conditions base,monitor --disturb off,on --empty-width $EMPTY_WIDTH --out $OUT/demo_spatial --video all
!ls $OUT/demo_spatial/videos | head -50"""),
]


def cell(kind, src):
    lines = src.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


nb = {
    "cells": [cell(k, s) for k, s in CELLS],
    "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                 "kernelspec": {"display_name": "Python 3", "name": "python3"}},
    "nbformat": 4, "nbformat_minor": 0,
}
out = Path(__file__).with_name("run_colab.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("wrote", out)
