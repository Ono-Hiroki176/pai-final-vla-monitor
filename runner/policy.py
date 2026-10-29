"""公開 VLA（既定: lerobot/smolvla_libero）を読み込み、1 step ずつ行動を返す。

観測 → 行動の変換は lerobot-eval と同じ経路をそのまま通す:
  preprocess_observation → env 用前処理（LIBERO: 画像の180度回転・8次元の状態）
  → 方策の前処理（正規化・トークナイズ） → select_action → 後処理（逆正規化）
自前で前処理を書き直すと、回転や正規化の取り違えで黙って成功率が落ちるため。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DEFAULT_POLICY = "lerobot/smolvla_libero"
DEFAULT_REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"  # 2026-10-12 版に固定
# LIBERO 環境のカメラ名 → smolvla_libero の学習時のカメラ名（俯瞰 = camera1、手首 = camera2）
RENAME_MAP = {"observation.images.image": "observation.images.camera1",
              "observation.images.image2": "observation.images.camera2"}


@dataclass
class PolicyBundle:
    policy: torch.nn.Module
    preprocessor: object
    postprocessor: object
    env_preprocessor: object
    env_postprocessor: object
    device: str

    def reset(self) -> None:
        """行動キュー（チャンクの残り）を捨てる。"""
        self.policy.reset()

    @torch.inference_mode()
    def act(self, raw_obs: dict, task: str) -> np.ndarray:
        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import ACTION

        obs = preprocess_observation(raw_obs)
        obs["task"] = [task]
        obs = self.env_preprocessor(obs)
        obs = self.preprocessor(obs)
        action = self.policy.select_action(obs)
        action = self.postprocessor(action)
        action = self.env_postprocessor({ACTION: action})[ACTION]
        return action.to("cpu").numpy().reshape(-1).astype(np.float32)


def load_policy(env_cfg, repo_id: str = DEFAULT_POLICY, revision: str | None = DEFAULT_REVISION,
                device: str | None = None) -> PolicyBundle:
    from huggingface_hub import snapshot_download
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies import make_policy, make_pre_post_processors

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    local = Path(repo_id) if Path(repo_id).is_dir() else Path(snapshot_download(repo_id, revision=revision))

    policy_cfg = PreTrainedConfig.from_pretrained(str(local))
    policy_cfg.pretrained_path = local
    policy_cfg.device = device
    # SmolVLA のチェックポイントは土台の VLM（SmolVLM2）の重みも含む。True のままだと
    # 土台の重み約 1GB を別に落としてから上書きするだけになるので、設定だけから組み立てる
    if hasattr(policy_cfg, "load_vlm_weights"):
        policy_cfg.load_vlm_weights = False

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=RENAME_MAP)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(local),
        preprocessor_overrides={"device_processor": {"device": device},
                                "rename_observations_processor": {"rename_map": RENAME_MAP}},
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    return PolicyBundle(policy, pre, post, env_pre, env_post, device)
