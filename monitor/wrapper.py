"""VLA の外側に付ける監視層。

    ctrl = MonitoredController(policy_fn, policy_reset_fn)
    ctrl.reset()
    while ...:
        action, info = ctrl.act(eef_pos, gripper_qpos, policy_obs)

policy_fn(policy_obs) -> (7,) は VLA の 1 step 分の行動、policy_reset_fn() は VLA の
行動キュー（チャンク）を捨てる関数。監視層は VLA の中身を知らないので、方策を差し替えても同じコードで動く。

enabled=False のときは検知だけを行い（影のモード）、行動には一切介入しない。
素の VLA の走行でも「検知器が何を検知したか」を記録でき、検知の精度の評価に使える。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .detectors import DetectorConfig, Event, FailureDetector
from .recovery import OPEN, Recovery, RecoveryConfig


@dataclass
class MonitorConfig:
    enabled: bool = True
    max_retries: int = 2  # これを超えて失敗したら HELP（停止して助けを求める）


class MonitoredController:
    def __init__(
        self,
        policy_fn: Callable[[Any], np.ndarray],
        policy_reset_fn: Callable[[], None],
        cfg: MonitorConfig | None = None,
        det_cfg: DetectorConfig | None = None,
        rec_cfg: RecoveryConfig | None = None,
        seed: int = 0,
    ):
        self.policy_fn = policy_fn
        self.policy_reset_fn = policy_reset_fn
        self.cfg = cfg or MonitorConfig()
        self.det = FailureDetector(det_cfg)
        self.rec_cfg = rec_cfg
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self.det.reset()
        self.rec = Recovery(self.rec_cfg, np.random.default_rng(self.seed))
        self.retries = 0
        self.help = False
        self.last_action: np.ndarray | None = None
        self.last_event = Event.NONE
        self.policy_reset_fn()

    def act(self, eef_pos: np.ndarray, gripper_qpos: np.ndarray, policy_obs: Any) -> tuple[np.ndarray, dict]:
        eef_pos = np.asarray(eef_pos, dtype=float).reshape(3)
        event = Event.NONE

        if self.help:
            action = np.zeros(7, dtype=np.float32)
            action[-1] = OPEN
            status = "HELP"
        elif self.rec.active:
            action = self.rec.act(eef_pos)
            status = "RECOVERING"
            if not self.rec.active:  # 回復動作が終わった: 検知器と VLA を新しい状態から始め直す
                self.det.reset()
                self.policy_reset_fn()
        else:
            event = self.det.step(eef_pos, gripper_qpos, self.last_action)
            if event is not Event.NONE and self.cfg.enabled:
                self.retries += 1
                if self.retries > self.cfg.max_retries:
                    self.help = True
                    action = np.zeros(7, dtype=np.float32)
                    action[-1] = OPEN
                    status = "HELP"
                else:
                    self.rec.start(eef_pos)
                    action = self.rec.act(eef_pos)
                    status = event.value
            else:
                action = np.asarray(self.policy_fn(policy_obs), dtype=np.float32).reshape(-1)
                status = "NORMAL"

        # 影のモードでは同じ事象が毎 step 続くので、切り替わった瞬間だけを「新しい検知」として返す
        new_event = event if event is not self.last_event else Event.NONE
        if not self.rec.active and not self.help:
            self.last_event = event
        self.last_action = action
        info = {
            "status": status,
            "event": new_event.value,
            "retries": self.retries,
            "phase": self.rec.phase.value,
            "width": self.det.last_width,
            "help": self.help,
        }
        return action, info
