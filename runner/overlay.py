"""動画用のフレーム合成。俯瞰カメラと手首カメラを横に並べ、上に監視層の状態を表示する。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

STATUS_COLOR = {
    "NORMAL": (46, 160, 67),
    "GRASP_MISS": (219, 68, 55),
    "DROP": (219, 68, 55),
    "STALL": (230, 145, 30),
    "DITHER": (230, 145, 30),
    "RECOVERING": (52, 120, 246),
    "HELP": (150, 50, 200),
    "SUCCESS": (46, 160, 67),
    "VLA ONLY": (110, 110, 110),
}
LABEL = {
    "NORMAL": "NORMAL",
    "GRASP_MISS": "GRASP MISS detected",
    "DROP": "DROP detected",
    "STALL": "STALL detected",
    "DITHER": "DITHER detected",
    "RECOVERING": "RECOVERING",
    "HELP": "HELP - stopped safely",
    "SUCCESS": "SUCCESS",
    "VLA ONLY": "VLA only (no monitor)",
}


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


class FrameComposer:
    def __init__(self, title: str, task: str, scale: int = 2):
        self.title, self.task, self.scale = title, task, scale
        self.f_big, self.f_small = _font(14 * scale), _font(10 * scale)

    def __call__(self, agentview: np.ndarray, wrist: np.ndarray, status: str, step: int,
                 retries: int, max_retries: int, note: str = "") -> np.ndarray:
        # LIBERO の生画像は上下左右が反転しているので、表示用に戻す
        a = Image.fromarray(np.ascontiguousarray(agentview[::-1, ::-1]))
        w = Image.fromarray(np.ascontiguousarray(wrist[::-1, ::-1]))
        h = a.height * self.scale
        a = a.resize((a.width * self.scale, h))
        w = w.resize((w.width * h // w.height, h))
        bar = 34 * self.scale
        canvas = Image.new("RGB", (a.width + w.width, h + bar), (20, 20, 20))
        canvas.paste(a, (0, bar))
        canvas.paste(w, (a.width, bar))

        d = ImageDraw.Draw(canvas)
        color = STATUS_COLOR.get(status, (90, 90, 90))
        d.rectangle([0, 0, canvas.width, bar], fill=color)
        s = self.scale
        d.text((8 * s, 3 * s), f"{LABEL.get(status, status)}", font=self.f_big, fill="white")
        right = f"{self.title}   step {step}   retry {retries}/{max_retries}"
        tw = d.textlength(right, font=self.f_small)
        d.text((canvas.width - tw - 8 * s, 4 * s), right, font=self.f_small, fill="white")
        d.text((8 * s, 20 * s), self.task + (f"   [{note}]" if note else ""), font=self.f_small, fill="white")
        return np.asarray(canvas)


class Mp4Writer:
    """フレームを溜めずに 1 枚ずつ書き出す（720p × 300 フレームを溜めると約 500MB になる）。"""

    def __init__(self, path: str | Path, fps: int = 20):
        import imageio.v2 as imageio

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.w = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1)

    def add(self, frame: np.ndarray) -> None:
        self.w.append_data(frame)

    def close(self) -> None:
        self.w.close()
