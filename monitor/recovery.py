"""失敗を検知したあとの立て直し動作（退避 → 少しずらす）。

手順:
  OPEN   グリッパを開く（挟みかけの物を離し、指を開いた状態に戻す）
  LIFT   真上に lift_height だけ上がる（物や障害物から離れる）
  SHIFT  水平にランダムな方向へ shift_min〜shift_max ずらす
→ 完了後は VLA に制御を戻す。

ずらす理由: 掴めるかどうかは mm 単位の状態の差で決まることが多く、同じ状態に戻すと
VLA は同じ行動を出して同じ失敗を繰り返す。少し違う状態から再開させる。

行動は LIBERO の OSC_POSE（手先の差分指令、各軸 [-1, 1]、並進 1.0 ≒ 0.05 m の目標移動）。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

OPEN, CLOSE = -1.0, 1.0
ACTION_SCALE = 0.05  # OSC_POSE の並進 1.0 に対応する目標移動 [m]


class Phase(str, Enum):
    IDLE = "IDLE"
    OPEN = "OPEN"
    LIFT = "LIFT"
    SHIFT = "SHIFT"


@dataclass
class RecoveryConfig:
    open_steps: int = 10
    lift_height: float = 0.08
    lift_max_steps: int = 30
    shift_min: float = 0.01
    shift_max: float = 0.025
    shift_max_steps: int = 20
    reach_tol: float = 0.005
    max_speed: float = 0.6   # 行動の並進成分の上限（|a| ≤ 1）


class Recovery:
    def __init__(self, cfg: RecoveryConfig | None = None, rng: np.random.Generator | None = None):
        self.cfg = cfg or RecoveryConfig()
        self.rng = rng or np.random.default_rng(0)
        self.phase = Phase.IDLE
        self.k = 0
        self.target: np.ndarray | None = None

    @property
    def active(self) -> bool:
        return self.phase is not Phase.IDLE

    def start(self, eef_pos: np.ndarray) -> None:
        self.phase, self.k = Phase.OPEN, 0
        self.start_pos = np.asarray(eef_pos, dtype=float).copy()

    def _move_to(self, eef_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        err = (target - eef_pos) / ACTION_SCALE
        n = np.linalg.norm(err)
        if n > self.cfg.max_speed:
            err = err * (self.cfg.max_speed / n)
        return err

    def act(self, eef_pos: np.ndarray) -> np.ndarray:
        """現在の手先位置から次の行動（7次元）を返す。最後の行動を返したあと IDLE に戻る。"""
        c = self.cfg
        eef_pos = np.asarray(eef_pos, dtype=float)
        a = np.zeros(7, dtype=np.float32)
        a[-1] = OPEN
        self.k += 1

        if self.phase is Phase.OPEN:
            if self.k >= c.open_steps:
                self.phase, self.k = Phase.LIFT, 0
                self.target = self.start_pos + np.array([0.0, 0.0, c.lift_height])
            return a

        if self.phase is Phase.LIFT:
            a[:3] = self._move_to(eef_pos, self.target)
            if eef_pos[2] >= self.target[2] - c.reach_tol or self.k >= c.lift_max_steps:
                r = self.rng.uniform(c.shift_min, c.shift_max)
                th = self.rng.uniform(0, 2 * np.pi)
                self.target = eef_pos + np.array([r * np.cos(th), r * np.sin(th), 0.0])
                self.phase, self.k = Phase.SHIFT, 0
            return a

        if self.phase is Phase.SHIFT:
            a[:3] = self._move_to(eef_pos, self.target)
            if np.linalg.norm(self.target - eef_pos) < c.reach_tol or self.k >= c.shift_max_steps:
                self.phase, self.k = Phase.IDLE, 0
            return a

        raise RuntimeError("Recovery.act() called while IDLE")
