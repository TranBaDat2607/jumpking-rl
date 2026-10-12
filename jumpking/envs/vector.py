"""Natively batched environments (no per-env Python objects or processes).

``JumpKingVectorEnv`` is a ``gymnasium.vector.VectorEnv`` (macro actions,
vector observations) backed by ``BatchSim``: all N kings are advanced by one
multi-threaded numba call per step. Finished sub-environments are reset in
the same step (``AutoresetMode.SAME_STEP``); their last observation is in
``infos["final_obs"]``.

``SB3VecEnv`` adapts it to stable-baselines3's ``VecEnv`` interface:

    from jumpking.envs.vector import SB3VecEnv
    venv = SB3VecEnv(num_envs=64)
    model = PPO("MlpPolicy", venv)
"""
import numpy as np
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from .. import constants as C
from .. import physics as P
from ..batch import KIND_JUMP, KIND_WALK, BatchSim
from ..game import Game
from ..level import Level
from .jump_king_env import VECTOR_FEATURES, vector_features


class JumpKingVectorEnv(VectorEnv):
    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP, "render_modes": []}

    def __init__(self, num_envs=64, level="builtin", reward="height", goal_bonus=10.0,
                 step_penalty=0.0, max_episode_steps=2000, charges=tuple(range(2, 37)),
                 walk_ticks=(4, 8, 16, 32), start="start", max_start_screen=None,
                 wind_phase="random", splat_time=C.SPLAT_TIME, macro_timeout=60 * 20, seed=None):
        if isinstance(level, str):
            level = Level.load(level)
        self.level = level
        self.num_envs = int(num_envs)
        self.sim = BatchSim(level, self.num_envs, splat_time=splat_time)
        self.reward_mode = reward
        self.goal_bonus = float(goal_bonus)
        self.step_penalty = float(step_penalty)
        self.max_episode_steps = int(max_episode_steps)
        self.start_mode = start
        self.max_start_screen = level.n_screens - 1 if max_start_screen is None else int(max_start_screen)
        self.wind_phase = wind_phase
        self.macro_timeout = int(macro_timeout)
        table = [(KIND_JUMP, d, c) for d in (-1, 0, 1) for c in charges]
        table += [(KIND_WALK, d, t) for d in (-1, 1) for t in walk_ticks]
        self._table = np.array(table, np.int64)
        self.single_action_space = spaces.Discrete(len(table))
        self.single_observation_space = spaces.Box(-np.inf, np.inf, shape=(len(VECTOR_FEATURES),),
                                                   dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self._rng = np.random.default_rng(seed)
        self._game = Game(level, splat_time=splat_time)
        self._steps = np.zeros(self.num_envs, np.int64)
        self._best_screen = np.zeros(self.num_envs, np.int64)

    # ------------------------------------------------------------ helpers
    def _reset_indices(self, idx, options=None):
        g = self._game
        options = options or {}
        for i in idx:
            if self.wind_phase == "random":
                g.params[P.P_WIND_OFFSET] = float(self._rng.uniform(0.0, 13.06))
            elif isinstance(self.wind_phase, (int, float)):
                g.params[P.P_WIND_OFFSET] = float(self.wind_phase)
            else:
                g.params[P.P_WIND_OFFSET] = 0.0
            start = options.get("screen", self.start_mode)
            if start == "start" or start is None:
                g.reset()
            elif start == "random_screen":
                g.reset_to_screen(int(self._rng.integers(0, self.max_start_screen + 1)), rng=self._rng)
            elif isinstance(start, (list, tuple)):
                g.reset_to_screen(int(start[int(self._rng.integers(len(start)))]), rng=self._rng)
            else:
                g.reset_to_screen(int(start), rng=self._rng)
            g.settle()
            self.sim.states[i] = g.state
            self._steps[i] = 0
            self._best_screen[i] = g.screen

    def _infos(self):
        s = self.sim.states
        return {
            "height": self.sim.heights, "max_height": self.sim.max_heights,
            "screen": self.sim.screens, "goal": self.sim.done.copy(),
            "ticks": s[:, P.S_TICK].astype(np.int64), "jumps": s[:, P.S_JUMPS].astype(np.int64),
            "falls": s[:, P.S_FALLS].astype(np.int64),
        }

    # ---------------------------------------------------------------- api
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_indices(range(self.num_envs), options)
        return vector_features(self.sim.states, self.level), self._infos()

    def step(self, actions):
        actions = np.asarray(actions, np.int64).reshape(self.num_envs)
        acts = self._table[actions]
        h0 = self.sim.heights
        mh0 = self.sim.max_heights
        s0 = self.sim.screens
        events, ticks = self.sim.macro(acts[:, 0], acts[:, 1], acts[:, 2], max_ticks=self.macro_timeout)
        h1 = self.sim.heights
        goal = self.sim.done
        if self.reward_mode == "height":
            rew = (h1 - h0) / C.SCREEN_H
        elif self.reward_mode == "max_height":
            rew = (self.sim.max_heights - mh0) / C.SCREEN_H
        elif self.reward_mode == "screen":
            sc = self.sim.screens
            rew = np.maximum(sc - self._best_screen, 0).astype(np.float64)
            rew -= np.maximum(s0 - sc, 0)
            self._best_screen = np.maximum(self._best_screen, sc)
        elif self.reward_mode == "sparse":
            rew = np.zeros(self.num_envs)
        else:
            raise ValueError(self.reward_mode)
        rew = rew + self.goal_bonus * goal - self.step_penalty
        self._steps += 1
        terminated = goal.copy()
        truncated = (~terminated) & (self._steps >= self.max_episode_steps)
        obs = vector_features(self.sim.states, self.level)
        infos = self._infos()
        infos["events"] = events
        infos["ticks_used"] = ticks
        done = terminated | truncated
        if done.any():
            idx = np.nonzero(done)[0]
            infos["final_obs"] = obs.copy()
            infos["_final_obs"] = done.copy()
            self._reset_indices(idx)
            obs[idx] = vector_features(self.sim.states[idx], self.level)
        return obs, rew.astype(np.float32), terminated, truncated, infos

    def get_states(self):
        return self.sim.states.copy()

    def set_states(self, states, idx=None):
        if idx is None:
            self.sim.states[:] = states
        else:
            self.sim.states[idx] = states

    def close_extras(self, **kwargs):
        pass


try:
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv as _SB3VecEnv
except ImportError:  # pragma: no cover - optional dependency
    _SB3VecEnv = None


if _SB3VecEnv is not None:

    class SB3VecEnv(_SB3VecEnv):
        """stable-baselines3 VecEnv over ``JumpKingVectorEnv``."""

        def __init__(self, num_envs=64, **kwargs):
            self.venv = JumpKingVectorEnv(num_envs=num_envs, **kwargs)
            super().__init__(num_envs, self.venv.single_observation_space,
                             self.venv.single_action_space)
            self._actions = None

        def reset(self):
            seeds = getattr(self, "_seeds", None)
            seed = seeds[0] if seeds and seeds[0] is not None else None
            obs, _ = self.venv.reset(seed=seed)
            if hasattr(self, "_reset_seeds"):
                self._reset_seeds()
            return obs

        def step_async(self, actions):
            self._actions = actions

        def step_wait(self):
            obs, rew, term, trunc, info = self.venv.step(self._actions)
            dones = term | trunc
            infos = []
            for i in range(self.num_envs):
                d = {"height": int(info["height"][i]), "max_height": int(info["max_height"][i]),
                     "screen": int(info["screen"][i]), "goal": bool(info["goal"][i])}
                if dones[i]:
                    d["terminal_observation"] = info["final_obs"][i]
                    d["TimeLimit.truncated"] = bool(trunc[i] and not term[i])
                infos.append(d)
            return obs, rew, dones, infos

        def close(self):
            pass

        def get_attr(self, attr_name, indices=None):
            return [getattr(self.venv, attr_name)] * len(self._get_indices(indices))

        def set_attr(self, attr_name, value, indices=None):
            setattr(self.venv, attr_name, value)

        def env_method(self, method_name, *args, indices=None, **kwargs):
            return [getattr(self.venv, method_name)(*args, **kwargs)
                    for _ in self._get_indices(indices)]

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False] * len(self._get_indices(indices))

        def _get_indices(self, indices):
            if indices is None:
                return range(self.num_envs)
            if isinstance(indices, int):
                return [indices]
            return indices
