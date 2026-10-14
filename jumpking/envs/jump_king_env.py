"""Gymnasium environment for the Jump King clone.

Action modes
------------
``macro`` (default)
    ``Discrete(3 * len(charges) + 2 * len(walk_ticks))``. One action is a whole
    jump (direction x charge ticks; power = charge / 36) or a short walk, and
    the simulation runs until the king has landed and stands still. This is
    the usual abstraction for Jump King RL and makes episodes ~100x shorter.
``frame``
    ``Discrete(6)``: noop, left, right, jump, jump+left, jump+right, held for
    ``frame_skip`` ticks (exactly like pressing buttons in the real game).

Observation modes
-----------------
``vector``  float32 features (position, velocity, flags, wind, progress).
``grid``    uint8 (3, 90, 60) tile view of the current screen and the one
            above it: blocking terrain, special surfaces, player.
``rgb``     rendered frame, ``image_size`` (H, W, 3).
``dict``    ``{"vector": ..., "grid": ...}``.

Endings
-------
The original world has three babes (``level.endings``): ``normal`` (screen
43), ``new_babe_plus`` (screen 100) and ``ghost_of_the_babe`` (screen 154).
Reaching any babe ends the game, as in the real one; ``ending`` picks the one
that counts as success (``"any"`` accepts all three). Use ``start="imp_room"``
for the New Babe+ crown-sacrifice start.

Rewards
-------
``height``      potential-based: change of height / 360 (+ goal bonus)
``max_height``  only new record heights are rewarded (+ goal bonus)
``screen``      +1 per new highest screen, -1 per screen fallen (+ goal bonus)
``route``       +1 per new screen along the route of ``ending`` (routes.py);
                the right reward for the DLC routes, which go sideways and down
``sparse``      1 on reaching the babe, 0 otherwise
or any callable ``f(prev_info, info, events) -> float``. The goal bonus is
only paid for the selected ending.
"""
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .. import constants as C
from .. import physics as P
from ..game import Game
from ..level import Level
from ..routes import START_POSITIONS, RouteProgress, ending_id, route_for

FRAME_ACTIONS = (
    (0, False),   # 0 noop
    (-1, False),  # 1 left
    (1, False),   # 2 right
    (0, True),    # 3 jump
    (-1, True),   # 4 jump + left
    (1, True),    # 5 jump + right
)
FRAME_ACTION_NAMES = ("noop", "left", "right", "jump", "jump_left", "jump_right")
VECTOR_FEATURES = (
    "x", "y_in_screen", "vx", "vy", "screen", "on_ground", "knocked", "on_ice",
    "on_snow", "charging", "charge", "splat", "wind", "wind_active", "facing",
    "height", "max_height", "jump_held", "x_tile_frac", "screen_has_wind",
    "wind_phase_sin", "wind_phase_cos", "in_water", "in_sand", "giant_boots", "snake_ring",
)

# grid observation, channel 1 ("special" surfaces)
SPECIAL_VALUES = {P.T_ICE: 255, P.T_SNOW: 160, P.T_WATER: 208, P.T_SAND: 128, P.T_QUARK: 64,
                  P.T_NOWIND: 48}
BLOCKING_TILES = (P.T_SOLID, P.T_ICE, P.T_SNOW, P.T_SLOPE_TL, P.T_SLOPE_TR, P.T_SLOPE_BL,
                  P.T_SLOPE_BR)


def wind_force(ticks, offsets, scale=1.0):
    """Vectorised numpy port of physics.wind_velocity (float32 like the game)."""
    f = np.float32
    seconds = ticks * C.TARGET_ELAPSED_SECONDS + offsets
    t = (np.floor(seconds * 1000.0 + 0.5) / 1000.0).astype(f)
    phase = (t * f(C.WIND_PHASE_RATE)).astype(f)
    wave = np.sin(phase).astype(f)
    wave = np.where(np.cos(phase).astype(f) > 0, wave * f(2) + f(1), wave * f(2) - f(1))
    wave = np.clip(wave, -1.0, 1.0).astype(f)
    return (wave * f(C.WIND_STRENGTH) * f(scale)).astype(np.float64)


def vector_features(states, level):
    """Observation vectors (N, len(VECTOR_FEATURES)) for a batch of states."""
    s = np.atleast_2d(states)
    screen = s[:, P.S_SCREEN].astype(np.int64)
    top = -C.SCREEN_H * screen
    n = max(1, level.n_screens - 1)
    total = float(C.SCREEN_H * level.n_screens)
    codes = level.wind[screen].astype(np.int64)
    has_wind = (codes > 0).astype(np.float64)
    wind = has_wind * wind_force(s[:, P.S_TICK], s[:, P.S_WIND_OFFSET])
    for i in np.nonzero(codes >= 2)[0]:   # coloured wind variants (not in the original map)
        wind[i] = P.wind_velocity(s[i, P.S_TICK], s[i, P.S_WIND_OFFSET], 1.0, int(codes[i]))
    t = s[:, P.S_TICK] * C.TARGET_ELAPSED_SECONDS + s[:, P.S_WIND_OFFSET]
    phase = t * C.WIND_PHASE_RATE
    height = C.SCREEN_H - (s[:, P.S_Y].astype(np.int64) + C.PLAYER_H)
    max_height = C.SCREEN_H - s[:, P.S_MAX_HEIGHT]
    cols = [
        s[:, P.S_X] / C.SCREEN_W,
        (s[:, P.S_Y] - top) / C.SCREEN_H,
        s[:, P.S_VX] / C.MAX_FALL,
        s[:, P.S_VY] / C.MAX_FALL,
        screen / n,
        s[:, P.S_ON_GROUND], s[:, P.S_KNOCKED], s[:, P.S_ON_ICE], s[:, P.S_ON_SNOW],
        s[:, P.S_CHARGING], s[:, P.S_CHARGE] / C.JUMP_TIME, s[:, P.S_SPLAT],
        wind / C.WIND_STRENGTH, s[:, P.S_WIND_LATCH], s[:, P.S_FACING],
        height / total, max_height / total,
        s[:, P.S_JUMP_HELD], (s[:, P.S_X] % 8) / 8.0, has_wind,
        has_wind * np.sin(phase), has_wind * np.cos(phase),
        s[:, P.S_IN_WATER], s[:, P.S_IN_SAND], s[:, P.S_BOOTS], s[:, P.S_RING],
    ]
    return np.stack(cols, axis=1).astype(np.float32)


class JumpKingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": C.FPS}

    def __init__(self, level="auto", action_mode="macro", obs_mode="vector",
                 reward="height", goal_bonus=10.0, step_penalty=0.0, ending="normal",
                 max_episode_steps=None, frame_skip=4,
                 charges=tuple(range(2, 37)), walk_ticks=(4, 8, 16, 32),
                 start="start", max_start_screen=None, wind_phase="random",
                 splat_time=C.SPLAT_TIME, image_size=(90, 120),
                 macro_timeout=60 * 20, giant_boots=False, snake_ring=False,
                 landed_macros=False, render_mode=None):
        super().__init__()
        if isinstance(level, str):
            level = Level.load(level)
        self.level = level
        self.game = Game(level, splat_time=splat_time)
        self.game.wear(giant_boots=giant_boots, snake_ring=snake_ring)
        self.ending = ending
        self.ending_id = ending_id(ending)
        self._route = RouteProgress(route_for(ending))
        self.action_mode = action_mode
        self.obs_mode = obs_mode
        self.reward_mode = reward
        self.goal_bonus = float(goal_bonus)
        self.step_penalty = float(step_penalty)
        self.frame_skip = int(frame_skip)
        self.charges = tuple(int(c) for c in charges)
        self.walk_ticks = tuple(int(t) for t in walk_ticks)
        self.start_mode = start
        self.max_start_screen = (min(level.n_screens, C.BASE_SCREENS) - 1
                                 if max_start_screen is None else int(max_start_screen))
        self.wind_phase = wind_phase
        self.image_size = tuple(image_size)
        self.macro_timeout = int(macro_timeout)
        self.landed_macros = bool(landed_macros)
        self.render_mode = render_mode
        if max_episode_steps is None:
            max_episode_steps = 2000 if action_mode == "macro" else (60 * 60 * 20) // self.frame_skip
        self.max_episode_steps = int(max_episode_steps)

        # ---- action space
        if action_mode == "macro":
            self._macro = [("jump", d, c) for d in (-1, 0, 1) for c in self.charges]
            self._macro += [("walk", d, t) for d in (-1, 1) for t in self.walk_ticks]
            self.action_space = spaces.Discrete(len(self._macro))
        elif action_mode == "frame":
            self._macro = None
            self.action_space = spaces.Discrete(len(FRAME_ACTIONS))
        else:
            raise ValueError("action_mode must be 'macro' or 'frame'")

        # ---- observation space
        vec = spaces.Box(-np.inf, np.inf, shape=(len(VECTOR_FEATURES),), dtype=np.float32)
        grid = spaces.Box(0, 255, shape=(3, 2 * C.SCREEN_TILES_H, C.SCREEN_TILES_W), dtype=np.uint8)
        if obs_mode == "vector":
            self.observation_space = vec
        elif obs_mode == "grid":
            self.observation_space = grid
        elif obs_mode == "rgb":
            h, w = self.image_size
            self.observation_space = spaces.Box(0, 255, shape=(h, w, 3), dtype=np.uint8)
        elif obs_mode == "dict":
            self.observation_space = spaces.Dict({"vector": vec, "grid": grid})
        else:
            raise ValueError("obs_mode must be vector, grid, rgb or dict")

        self._renderer = None
        self._window = None
        self._vec_buf = np.zeros(len(VECTOR_FEATURES), np.float32)
        self._static_grid = self._build_static_grids()
        self._steps = 0
        self._info = None
        self._best_screen = 0

    # ------------------------------------------------------------ helpers
    def action_meaning(self, action):
        if self.action_mode == "macro":
            return self._macro[int(action)]
        return FRAME_ACTION_NAMES[int(action)]

    def macro_index(self, kind, direction, value):
        """Inverse of ``action_meaning`` for macro mode."""
        return self._macro.index((kind, int(direction), int(value)))

    def _build_static_grids(self):
        """Per-screen terrain / surface channels (uint8, 45 x 60)."""
        out = []
        for i in range(self.level.n_screens):
            g = self.level.screen_grid(i)
            blocking = np.isin(g, BLOCKING_TILES).astype(np.uint8) * 255
            special = np.zeros_like(g, dtype=np.uint8)
            special[(g >= P.T_SLOPE_TL) & (g <= P.T_SLOPE_BR)] = 96
            for code, value in SPECIAL_VALUES.items():
                special[g == code] = value
            out.append((blocking, special))
        return out

    def _vector_obs(self):
        # scalar fast path (same features as vector_features, ~10x faster for one row)
        s = self.game.state
        screen = int(s[P.S_SCREEN])
        top = -C.SCREEN_H * screen
        total = float(C.SCREEN_H * self.level.n_screens)
        has_wind = 1.0 if self.level.wind[screen] else 0.0
        wind = self.game.wind()
        phase = (s[P.S_TICK] * C.TARGET_ELAPSED_SECONDS + s[P.S_WIND_OFFSET]) * C.WIND_PHASE_RATE
        height = C.SCREEN_H - (int(s[P.S_Y]) + C.PLAYER_H)
        out = self._vec_buf
        out[:] = (
            s[P.S_X] / C.SCREEN_W, (s[P.S_Y] - top) / C.SCREEN_H,
            s[P.S_VX] / C.MAX_FALL, s[P.S_VY] / C.MAX_FALL,
            screen / max(1, self.level.n_screens - 1),
            s[P.S_ON_GROUND], s[P.S_KNOCKED], s[P.S_ON_ICE], s[P.S_ON_SNOW],
            s[P.S_CHARGING], s[P.S_CHARGE] / C.JUMP_TIME, s[P.S_SPLAT],
            wind / C.WIND_STRENGTH, s[P.S_WIND_LATCH], s[P.S_FACING],
            height / total, (C.SCREEN_H - s[P.S_MAX_HEIGHT]) / total,
            s[P.S_JUMP_HELD], (s[P.S_X] % 8) / 8.0, has_wind,
            has_wind * math.sin(phase), has_wind * math.cos(phase),
            s[P.S_IN_WATER], s[P.S_IN_SAND], s[P.S_BOOTS], s[P.S_RING],
        )
        return out.copy()

    def _grid_obs(self):
        s = self.game.state
        screen = int(s[P.S_SCREEN])
        obs = np.zeros((3, 2 * C.SCREEN_TILES_H, C.SCREEN_TILES_W), np.uint8)
        H = C.SCREEN_TILES_H
        if screen + 1 < self.level.n_screens:
            b, sp = self._static_grid[screen + 1]
            obs[0, :H] = b
            obs[1, :H] = sp
        b, sp = self._static_grid[screen]
        obs[0, H:] = b
        obs[1, H:] = sp
        # player hitbox (tile resolution) in the 2-screen window
        top_world = -C.SCREEN_H * (screen + 1)
        hx, hy = int(s[P.S_X]), int(s[P.S_Y]) - top_world
        c0, c1 = max(0, hx // 8), min(C.SCREEN_TILES_W - 1, (hx + C.PLAYER_W - 1) // 8)
        r0, r1 = max(0, hy // 8), min(2 * H - 1, (hy + C.PLAYER_H - 1) // 8)
        if r0 <= r1 and c0 <= c1:
            obs[2, r0:r1 + 1, c0:c1 + 1] = 255
        return obs

    def _rgb_obs(self):
        if self._renderer is None:
            from ..render import Renderer

            self._renderer = Renderer(self.level)
        frame = self._renderer.frame(self.game.state)
        h, w = self.image_size
        if (h, w) == frame.shape[:2]:
            return frame
        ys = (np.arange(h) * frame.shape[0] / h).astype(np.int64)
        xs = (np.arange(w) * frame.shape[1] / w).astype(np.int64)
        return frame[ys][:, xs]

    def _obs(self):
        if self.obs_mode == "vector":
            return self._vector_obs()
        if self.obs_mode == "grid":
            return self._grid_obs()
        if self.obs_mode == "rgb":
            return self._rgb_obs()
        return {"vector": self._vector_obs(), "grid": self._grid_obs()}

    def _reward(self, prev, info, events):
        mode = self.reward_mode
        if callable(mode):
            r = float(mode(prev, info, events))
        elif mode == "height":
            r = (info["height"] - prev["height"]) / C.SCREEN_H
        elif mode == "max_height":
            r = (info["max_height"] - prev["max_height"]) / C.SCREEN_H
        elif mode == "screen":
            r = 0.0
            if info["screen"] > self._best_screen:
                r += info["screen"] - self._best_screen
                self._best_screen = info["screen"]
            if info["screen"] < prev["screen"]:
                r -= prev["screen"] - info["screen"]
        elif mode == "route":
            r = float(info["route_steps"])
        elif mode == "sparse":
            r = 0.0
        else:
            raise ValueError("unknown reward mode %r" % (mode,))
        if info["success"] and not prev["success"]:
            r += self.goal_bonus
        return r - self.step_penalty

    def _make_info(self, steps=0):
        info = self.game.info()
        info["success"] = bool(info["goal"]) and (self.ending_id == 0
                                                  or info["ending"] == self.ending_id)
        info["route_steps"] = int(steps)
        info["route_progress"] = int(self._route.index[0])
        return info

    # ---------------------------------------------------------------- api
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        g = self.game
        # wind clock phase
        if self.wind_phase == "random":
            g.params[P.P_WIND_OFFSET] = float(self.np_random.uniform(0.0, 13.06))
        elif isinstance(self.wind_phase, (int, float)):
            g.params[P.P_WIND_OFFSET] = float(self.wind_phase)
        else:
            g.params[P.P_WIND_OFFSET] = 0.0
        if "state" in options:
            g.restore(np.asarray(options["state"], dtype=np.float64))
        else:
            start = options.get("screen", self.start_mode)
            if start == "start" or start is None:
                g.reset()
            elif start in START_POSITIONS:
                g.reset(*START_POSITIONS[start])
            elif start == "random_screen":
                screen = int(self.np_random.integers(0, self.max_start_screen + 1))
                g.reset_to_screen(screen, rng=self.np_random)
            elif isinstance(start, (list, tuple)):
                screen = int(start[int(self.np_random.integers(len(start)))])
                g.reset_to_screen(screen, rng=self.np_random)
            else:
                g.reset_to_screen(int(start), rng=self.np_random)
            g.settle()
        self._steps = 0
        self._best_screen = g.screen
        self._route.reset([g.screen])
        self._info = self._make_info()
        info = dict(self._info, events=0, ticks_used=0)
        return self._obs(), info

    def step(self, action):
        g = self.game
        prev = self._info
        action = int(action)
        if self.action_mode == "macro":
            kind, d, v = self._macro[action]
            if kind == "jump":
                events, ticks = g.jump(d, v, timeout=self.macro_timeout, landed=self.landed_macros)
            else:
                events, ticks = g.walk(d, v, timeout=self.macro_timeout, landed=self.landed_macros)
        else:
            d, j = FRAME_ACTIONS[action]
            events = g.run([(d, 1 if j else 0)] * self.frame_skip)
            ticks = self.frame_skip
        self._steps += 1
        info = self._make_info(int(self._route.update([g.screen])[0]))
        reward = self._reward(prev, info, events)
        self._info = info
        terminated = bool(info["goal"])
        truncated = (not terminated) and self._steps >= self.max_episode_steps
        out = dict(info, events=int(events), ticks_used=int(ticks))
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, out

    # ----------------------------------------------------- state control
    def get_state(self):
        """Opaque snapshot of the full simulation state (numpy array)."""
        return self.game.snapshot()

    def set_state(self, state):
        self.game.restore(np.asarray(state, dtype=np.float64))
        self._info = self._make_info()

    # ------------------------------------------------------------- render
    def render(self):
        if self._renderer is None:
            from ..render import Renderer

            self._renderer = Renderer(self.level)
        frame = self._renderer.frame(self.game.state)
        if self.render_mode == "rgb_array" or self.render_mode is None:
            return frame
        if self.render_mode == "human":
            import pygame

            if self._window is None:
                pygame.init()
                self._window = pygame.display.set_mode((C.SCREEN_W * 2, C.SCREEN_H * 2))
                pygame.display.set_caption("Jump King clone - agent")
                self._clock = pygame.time.Clock()
            pygame.event.pump()
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            surf = pygame.transform.scale(surf, self._window.get_size())
            self._window.blit(surf, (0, 0))
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
        return None

    def close(self):
        if self._window is not None:
            import pygame

            pygame.display.quit()
            self._window = None
