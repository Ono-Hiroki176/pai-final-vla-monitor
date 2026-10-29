"""失敗の検知（L1 把持失敗 / L2 停滞・迷走）。

ロボット自身が観測できる量だけで判断する。シミュレータの正解（物体の位置など）は使わない。
入力は毎ステップの手先位置・グリッパの指の位置・直前に送った行動の3つ。

LIBERO（robosuite）の行動の最後の次元はグリッパ指令で、正が「閉じる」、負が「開く」。
Panda の指は左右それぞれ 0〜0.04 m 動くので、開き幅は |q_left| + |q_right|（最大約 0.08 m）。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Event(str, Enum):
    NONE = "NONE"
    GRASP_MISS = "GRASP_MISS"  # 閉じたのに何も挟んでいない（空振り）
    DROP = "DROP"              # 挟んでいた物が抜けた（落とした・取られた）
    STALL = "STALL"            # 手先がほとんど動かない
    DITHER = "DITHER"          # グリッパの開閉を繰り返して先へ進まない


@dataclass
class DetectorConfig:
    # L1: 閉じ指令を出し続けて settle_steps たち、指がほぼ止まったら判定する
    settle_steps: int = 8
    finger_still: float = 2e-4    # 1 step あたりの開き幅の変化 [m] がこれ未満なら「止まった」
    empty_width: float = 0.0034   # 止まった時点の開き幅 [m] がこれ未満なら「何も挟んでいない」（較正で決める）
    #   実測: 何も挟まず閉じると 0.0018、ボウルの縁を挟むと 0.0049。その中間
    # L2: 直近 window ステップの手先の動きと開閉の回数で判定する
    stall_window: int = 60
    stall_eps: float = 0.008      # window 内の手先の移動範囲 [m] がこれ未満なら停滞
    dither_toggles: int = 6       # window 内のグリッパ指令の符号反転がこれ以上なら迷走
    warmup_steps: int = 20        # エピソード冒頭は物理が落ち着くまで判定しない


def gripper_width(gripper_qpos: np.ndarray) -> float:
    q = np.asarray(gripper_qpos, dtype=float).reshape(-1)
    return float(np.abs(q).sum())


class FailureDetector:
    """1 エピソード分の状態を持つ。エピソード開始・回復の完了ごとに reset() する。"""

    def __init__(self, cfg: DetectorConfig | None = None):
        self.cfg = cfg or DetectorConfig()
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.close_run = 0          # 閉じ指令が連続した step 数
        self.prev_width: float | None = None
        self.holding = False        # 「閉じて止まり、何かを挟んでいる」と判断している状態
        self.eef_hist: deque[np.ndarray] = deque(maxlen=self.cfg.stall_window)
        self.cmd_hist: deque[float] = deque(maxlen=self.cfg.stall_window)
        self.last_width = 0.0
        self.last_settled = False

    def step(self, eef_pos: np.ndarray, gripper_qpos: np.ndarray, last_action: np.ndarray | None) -> Event:
        """1 step 分の観測を入れて、検知した事象を返す。

        last_action は「この観測を生んだ直前の行動」。エピソード最初の step は None。
        """
        c = self.cfg
        self.t += 1
        width = gripper_width(gripper_qpos)
        dw = 0.0 if self.prev_width is None else abs(width - self.prev_width)
        self.prev_width = width
        self.last_width = width

        closing = last_action is not None and float(last_action[-1]) > 0
        self.close_run = self.close_run + 1 if closing else 0
        settled = self.close_run >= c.settle_steps and dw < c.finger_still
        self.last_settled = settled

        self.eef_hist.append(np.asarray(eef_pos, dtype=float).reshape(3))
        if last_action is not None:
            self.cmd_hist.append(float(np.sign(last_action[-1])))

        if self.t <= c.warmup_steps:
            return Event.NONE

        # --- L1: 把持の空振り・落下 ---
        if not closing:
            self.holding = False
        elif settled:
            if width >= c.empty_width:
                self.holding = True
            elif self.holding:
                self.holding = False
                return Event.DROP
            else:
                return Event.GRASP_MISS

        # --- L2: 停滞・迷走 ---
        if len(self.eef_hist) == c.stall_window:
            pts = np.stack(self.eef_hist)
            span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
            if span < c.stall_eps:
                return Event.STALL
            cmds = np.asarray(self.cmd_hist)
            toggles = int(np.count_nonzero(np.diff(cmds) != 0)) if len(cmds) > 1 else 0
            if toggles >= c.dither_toggles:
                return Event.DITHER

        return Event.NONE
