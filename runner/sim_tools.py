"""シミュレータの内部を直接触る道具。**評価と実験の演出専用**で、監視層からは使わない。

- holding_gt:  正解の「いま物を掴んでいるか」（指と物体の接触から判定）。検知の精度の採点に使う
- Disturber:   実験者による妨害。ロボットが物を持ち上げた瞬間に、物を元の場所へ戻す
               （人が手から物を取り上げて置き直すのを模す）
"""
from __future__ import annotations

import numpy as np


def robosuite_env(libero_env):
    """lerobot の LiberoEnv から中の robosuite 環境を取り出す。"""
    return libero_env.unwrapped._env.env


def target_object(rs) -> str:
    """指示文で最初に出てくる操作対象（例: akita_black_bowl_1）。"""
    return rs.obj_of_interest[0]


def object_pos(rs, name: str) -> np.ndarray:
    return np.array(rs.sim.data.body_xpos[rs.obj_body_id[name]])


def holding_gt(rs) -> bool:
    gripper = rs.robots[0].gripper
    return any(rs._check_grasp(gripper=gripper, object_geoms=obj) for obj in rs.objects_dict.values())


class Disturber:
    """持ち上げを検出したら 1 エピソードに max_times 回まで、対象物を元の位置へ瞬間移動させる。"""

    def __init__(self, rs, lift_dz: float = 0.04, max_times: int = 1, jitter: float = 0.01,
                 rng: np.random.Generator | None = None):
        self.rs = rs
        self.name = target_object(rs)
        self.obj = rs.objects_dict[self.name]
        self.joint = self.obj.joints[0]
        self.home_qpos = np.array(rs.sim.data.get_joint_qpos(self.joint))
        self.lift_dz = lift_dz
        self.max_times = max_times
        self.jitter = jitter
        self.rng = rng or np.random.default_rng(0)
        self.count = 0
        self.fired_at: list[int] = []

    def maybe_fire(self, t: int) -> bool:
        if self.count >= self.max_times:
            return False
        z = object_pos(self.rs, self.name)[2]
        if z - self.home_qpos[2] < self.lift_dz or not holding_gt(self.rs):
            return False
        q = self.home_qpos.copy()
        q[:2] += self.rng.uniform(-self.jitter, self.jitter, size=2)
        q[2] += 0.005  # 机にめり込まないよう少し浮かせて置く
        sim = self.rs.sim
        sim.data.set_joint_qpos(self.joint, q)
        sim.data.set_joint_qvel(self.joint, np.zeros(6))
        sim.forward()
        self.count += 1
        self.fired_at.append(t)
        return True
