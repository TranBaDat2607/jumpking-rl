"""Gymnasium environment for the whole game: three stories and 48 achievements.

``JumpKingCampaignEnv`` wraps ``jumpking.campaign.Campaign``: the original
world with every NPC, merchant, bird, item and ending, and a persistent
profile. It needs a local Jump King install (the content is read from it).

Episodes and the profile
------------------------
An episode is one run (one new game) in ``mode`` (``normal``,
``new_babe_plus`` or ``ghost_of_the_babe``); it ends when any babe is reached
or after ``max_episode_steps``. The profile (inventory, worn items, story
flags, lifetime statistics, achievements) carries over between episodes by
default, exactly like the game's permanent save, so multi-run goals (the
gold rings for Big Spender, the crown for Double King, the boots for the
Giant Leaps runs) can be learned. ``reset(options=...)`` accepts ``mode``,
``profile`` (a ``Profile`` or ``"new"``), ``equip`` (items to put on),
``sacrifice_crown`` and ``state`` (a ``Campaign.snapshot()``).

Actions and observations
------------------------
``action_mode`` ``frame`` (default): ``Discrete(6)`` button combinations held
for ``frame_skip`` frames; ``macro``: whole jumps and walks as in
``JumpKingEnv``. NPCs keep talking and birds keep flying during a macro.
Observations are ``JumpKingEnv``'s vector features followed by the run
statistics, the worn items and the 48-bit achievement mask.

Rewards
-------
``achievement_bonus`` per newly unlocked achievement, ``ending_bonus`` for
reaching the ending of ``mode`` (or of ``ending``), plus an optional shaping
term ``route`` (+1 per new screen along the route of the mode).
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .. import constants as C
from ..achievements import CODES
from ..routes import RouteProgress, ending_id, route_for
from .jump_king_env import FRAME_ACTIONS, VECTOR_FEATURES, vector_features

ITEM_FEATURES = ("Crown", "Shoes", "CrownNBP", "SnakeRing", "GiantBoots", "Cap", "GnomeHat",
                 "Tunic", "YellowShoes", "CrownOwl", "CapeOwl")
CAMPAIGN_FEATURES = (VECTOR_FEATURES
                     + ("run_jumps", "run_falls", "run_minutes")
                     + tuple("wear_" + i for i in ITEM_FEATURES)
                     + tuple("ach_" + c for c in CODES))


class JumpKingCampaignEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": C.FPS}

    def __init__(self, mode="normal", action_mode="frame", frame_skip=4,
                 charges=tuple(range(2, 37)), walk_ticks=(4, 8, 16, 32),
                 achievement_bonus=1.0, ending_bonus=10.0, route_reward=0.0,
                 ending=None, max_episode_steps=None, keep_profile=True,
                 sacrifice_crown=True, legacy_teleport=False, level=None, content=None,
                 render_mode=None):
        super().__init__()
        from ..campaign import Campaign

        self.camp = Campaign(level=level, content=content, legacy_teleport=legacy_teleport)
        self.mode = mode
        self.ending = ending or mode
        self.ending_id = ending_id(self.ending)
        self.action_mode = action_mode
        self.frame_skip = int(frame_skip)
        self.achievement_bonus = float(achievement_bonus)
        self.ending_bonus = float(ending_bonus)
        self.route_reward = float(route_reward)
        self.keep_profile = keep_profile
        self.sacrifice_crown = sacrifice_crown
        self.render_mode = render_mode
        if action_mode == "macro":
            self._macro = [("jump", d, c) for d in (-1, 0, 1) for c in charges]
            self._macro += [("walk", d, t) for d in (-1, 1) for t in walk_ticks]
            self.action_space = spaces.Discrete(len(self._macro))
        elif action_mode == "frame":
            self._macro = None
            self.action_space = spaces.Discrete(len(FRAME_ACTIONS))
        else:
            raise ValueError("action_mode must be 'frame' or 'macro'")
        if max_episode_steps is None:
            max_episode_steps = 5000 if action_mode == "macro" else (60 * 60 * 60) // self.frame_skip
        self.max_episode_steps = int(max_episode_steps)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(CAMPAIGN_FEATURES),),
                                            dtype=np.float32)
        self._route = RouteProgress(route_for(self.ending))
        self._steps = 0
        self._renderer = None

    # ------------------------------------------------------------ helpers
    @property
    def profile(self):
        return self.camp.profile

    def action_meaning(self, action):
        if self._macro is not None:
            return self._macro[int(action)]
        return FRAME_ACTIONS[int(action)]

    def _obs(self):
        c = self.camp
        base = vector_features(c.game.state, c.level)[0]
        st = c.run_stats()
        extra = [st["jumps"] / 1000.0, st["falls"] / 100.0, st["seconds"] / 3600.0]
        extra += [1.0 if c.wearing(i) else 0.0 for i in ITEM_FEATURES]
        unlocked = c.unlocked
        extra += [1.0 if code in unlocked else 0.0 for code in CODES]
        return np.concatenate([base, np.asarray(extra, np.float32)]).astype(np.float32)

    def _info(self, new=(), route_steps=0):
        c = self.camp
        info = c.game.info()
        info.update(mode=self.mode, ended=c.ended, ending=c.ending,
                    success=bool(c.ended and (self.ending_id == 0 or c.ending == self.ending_id)),
                    new_achievements=list(new), achievements=len(c.unlocked),
                    run=c.run_stats(), route_progress=int(self._route.index[0]),
                    route_steps=int(route_steps))
        return info

    # ---------------------------------------------------------------- api
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        from ..campaign import Profile

        options = options or {}
        c = self.camp
        if "state" in options:
            c.restore(options["state"])
        else:
            prof = options.get("profile")
            if prof == "new" or (prof is None and not self.keep_profile):
                c.profile = Profile()
            elif prof is not None:
                c.profile = prof
                c._start_skins()       # owned, enabled items are worn at start-up
            self.mode = options.get("mode", self.mode)
            c.new_game(self.mode, sacrifice_crown=options.get("sacrifice_crown", self.sacrifice_crown))
            for item in options.get("equip", ()):
                c.equip(item)
        self._steps = 0
        self._route.reset([c.camera])
        return self._obs(), self._info()

    def step(self, action):
        c = self.camp
        action = int(action)
        gained = []
        if self._macro is not None:
            before = set(c.profile.achievements)
            kind, d, v = self._macro[action]
            if kind == "jump":
                c.jump(d, v)
            else:
                c.walk(d, v)
            gained = [k for k in c.profile.achievements if k not in before]
        else:
            d, j = FRAME_ACTIONS[action]
            for _ in range(self.frame_skip):
                c.step(d, j)
                gained.extend(c.new_achievements)
                if c.ended:
                    break
        self._steps += 1
        route_steps = int(self._route.update([c.camera])[0])
        info = self._info(gained, route_steps)
        reward = self.achievement_bonus * len(gained) + self.route_reward * route_steps
        if info["success"]:
            reward += self.ending_bonus
        terminated = bool(c.ended)
        truncated = (not terminated) and self._steps >= self.max_episode_steps
        return self._obs(), float(reward), terminated, truncated, info

    def get_state(self):
        return self.camp.snapshot()

    def set_state(self, state):
        self.camp.restore(state)

    def render(self):
        if self._renderer is None:
            from ..render import Renderer

            self._renderer = Renderer(self.camp.level)
        return self._renderer.frame(self.camp.game.state)
