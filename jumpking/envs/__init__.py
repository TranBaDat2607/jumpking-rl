"""Gymnasium environments.

Registered ids:

* ``JumpKing-v0``       macro actions, vector observations (fast default)
* ``JumpKingGrid-v0``   macro actions, tile-grid observations (for CNNs)
* ``JumpKingFrame-v0``  per-frame button presses, vector observations
* ``JumpKingPixels-v0`` per-frame button presses, RGB observations
* ``JumpKingCampaign-v0`` the whole game (three stories, NPCs, items, all 48
  achievements; needs a local Jump King install), per-frame actions
* ``JumpKingCampaignMacro-v0`` the same with macro actions

Use ``import jumpking.envs`` first, or the gymnasium module syntax
``gym.make("jumpking.envs:JumpKing-v0")`` which imports it for you.
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


_CAMPAIGN = "jumpking.envs.campaign_env:JumpKingCampaignEnv"
_CAMPAIGN_SPECS = {
    "JumpKingCampaign-v0": dict(action_mode="frame"),
    "JumpKingCampaignMacro-v0": dict(action_mode="macro"),
}


def register_envs():
    """Register the environments with gymnasium (idempotent)."""
    for _id, _kw in _SPECS.items():
        if _id not in gym.registry:
            gym.register(id=_id, entry_point=_ENTRY, kwargs=_kw)
    for _id, _kw in _CAMPAIGN_SPECS.items():
        if _id not in gym.registry:
            gym.register(id=_id, entry_point=_CAMPAIGN, kwargs=_kw)


register_envs()

__all__ = ["JumpKingEnv", "FRAME_ACTION_NAMES", "VECTOR_FEATURES", "register_envs"]
