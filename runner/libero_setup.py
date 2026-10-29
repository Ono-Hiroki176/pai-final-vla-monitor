"""LIBERO を import する前に呼ぶ準備。

LIBERO は初回 import 時に設定ファイルの場所を対話で聞いてくる（Colab や nohup では EOFError で落ちる）。
ここで設定ファイルを先に書き、対話を飛ばす。設定はこのプロジェクト専用の場所に置き、
他の LIBERO 環境（~/.libero）とは混ぜない。

アセット（物体のメッシュ・テクスチャ）は初回の環境生成時に公開リポジトリ lerobot/libero-assets から
~/.cache/libero/assets に自動で落ちてくる。
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prepare_libero(config_dir: str | os.PathLike | None = None, gl: str = "egl") -> Path:
    os.environ.setdefault("MUJOCO_GL", gl)
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])  # 2つは必ず揃える

    config_dir = Path(config_dir or os.environ.get("LIBERO_CONFIG_PATH") or PROJECT_ROOT / "outputs" / "libero_config")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    cfg_file = config_dir / "config.yaml"
    if cfg_file.exists():
        return cfg_file

    # libero.libero を import すると対話が走るので、パッケージの場所だけを調べる
    spec = importlib.util.find_spec("libero.libero")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("libero が入っていない（pip install 'lerobot[libero]'）")
    root = Path(next(iter(spec.submodule_search_locations)))

    import yaml

    config_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(root.parent / "datasets"),
        "assets": str(root / "assets"),
    }
    cfg_file.write_text(yaml.dump(cfg))
    return cfg_file
