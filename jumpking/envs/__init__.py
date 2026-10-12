"""Gymnasium environments.

Registered ids:

* ``JumpKing-v0``       macro actions, vector observations (fast default)
* ``JumpKingGrid-v0``   macro actions, tile-grid observations (for CNNs)
* ``JumpKingFrame-v0``  per-frame button presses, vector observations
* ``JumpKingPixels-v0`` per-frame button presses, RGB observations
"""
import gymnasium as gym

from .jump_king_env import FRAME_ACTION_NAMES, VECTOR_FEATURES, JumpKingEnv

_ENTRY = "jumpking.envs.jump_king_env:JumpKingEnv"
_SPECS = {
    "JumpKing-v0": dict(action_mode="macro", obs_mode="vector"),
    "JumpKingGrid-v0": dict(action_mode="macro", obs_mode="grid"),
    "JumpKingFrame-v0": dict(action_mode="frame", obs_mode="vector"),
    "JumpKingPixels-v0": dict(action_mode="frame", obs_mode="rgb"),
}
for _id, _kw in _SPECS.items():
    if _id not in gym.registry:
        gym.register(id=_id, entry_point=_ENTRY, kwargs=_kw)

__all__ = ["JumpKingEnv", "FRAME_ACTION_NAMES", "VECTOR_FEATURES"]
