"""High-level, programmatic interface to the simulation.

Example::

    from jumpking import Game
    g = Game()                       # built-in map (or Game(level="original"))
    g.step(direction=1, jump=False)  # one 60 Hz tick
    g.jump(direction=-1, charge=36)  # full jump to the left, waits for landing
    snap = g.snapshot(); ...; g.restore(snap)
"""
import numpy as np

from . import constants as C
from . import physics as P
from .level import Level

DEFAULT_MACRO_TIMEOUT = 60 * 30  # ticks


class Game:
    def __init__(self, level="auto", splat_time=C.SPLAT_TIME, wind_scale=1.0,
                 wind_offset=0.0):
        if isinstance(level, str):
            level = Level.load(level)
        self.level = level
        self.params = level.params(splat_time=splat_time, wind_offset=wind_offset,
                                   wind_scale=wind_scale)
        self._args = (level.tiles, level.wind, level.tp_left, level.tp_right, self.params)
        self.state = None
        self.reset()

    # ------------------------------------------------------------- control
    def reset(self, x=None, y=None, screen=None):
        """Place the player (defaults to the level start) standing still."""
        if x is None or y is None:
            x, y = self.level.start
        self.state = P.new_state(x, y, screen=screen, n_screens=self.level.n_screens,
                                 wind_offset=float(self.params[P.P_WIND_OFFSET]))
        if self.level.collides(int(x), int(y) + 1) and not self.level.collides(int(x), int(y)):
            self.state[P.S_ON_GROUND] = 1
            self.state[P.S_WAS_GROUNDED] = 1
        return self

    def reset_to_screen(self, screen, rng=None):
        """Start on a random standing spot of ``screen`` (curriculum learning)."""
        spots = self.spawn_points(screen)
        if not spots:
            raise ValueError("screen %d has no standing spot" % screen)
        i = 0 if rng is None else int(rng.integers(len(spots)))
        x, y = spots[i]
        return self.reset(x, y)

    def spawn_points(self, screen, stride=8):
        """Integer positions on screen ``screen`` where the player can stand."""
        cache = self.__dict__.setdefault("_spawn_cache", {})
        if (screen, stride) not in cache:
            cache[(screen, stride)] = self._spawn_points(screen, stride)
        return cache[(screen, stride)]

    def _spawn_points(self, screen, stride):
        lvl = self.level
        pts = []
        top = -C.SCREEN_H * screen
        for hy in range(top, top + C.SCREEN_H - C.PLAYER_H, 1):
            for hx in range(0, C.SCREEN_W - C.PLAYER_W + 1, stride):
                if not lvl.collides(hx, hy) and lvl.collides(hx, hy + 1):
                    # must be flat support (not a slope)
                    info = P.collide(lvl.tiles, lvl.offset_y, hx, hy + 1)
                    if info & P.CI_BOX:
                        pts.append((float(hx), float(hy)))
        return pts

    def step(self, direction=0, jump=False):
        """Advance one tick with the given inputs; returns event flags."""
        return int(P.tick(self.state, int(direction), 1 if jump else 0, *self._args))

    def run(self, inputs):
        """Run a list of (direction, jump) pairs. Returns OR-ed events."""
        if not len(inputs):
            return 0
        arr = np.asarray(inputs, dtype=np.int64).reshape(-1, 2)
        return int(P.run_inputs(self.state, arr[:, 0].copy(), arr[:, 1].copy(), *self._args))

    def jump(self, direction, charge, timeout=DEFAULT_MACRO_TIMEOUT):
        """Charge for ``charge`` ticks (2..36), jump, wait until landed.

        Returns (events, ticks)."""
        ev, n = P.macro_jump(self.state, int(direction), int(charge), *self._args, int(timeout))
        return int(ev), int(n)

    def walk(self, direction, ticks, timeout=DEFAULT_MACRO_TIMEOUT):
        ev, n = P.macro_walk(self.state, int(direction), int(ticks), *self._args, int(timeout))
        return int(ev), int(n)

    def settle(self, timeout=DEFAULT_MACRO_TIMEOUT):
        ev, n = P.settle(self.state, *self._args, int(timeout))
        return int(ev), int(n)

    # ------------------------------------------------------------ snapshot
    def snapshot(self):
        return self.state.copy()

    def restore(self, snap):
        n = min(len(snap), P.STATE_SIZE)  # older (shorter) snapshots stay loadable
        self.state[:n] = snap[:n]

    # --------------------------------------------------------------- query
    s = property(lambda self: self.state)
    x = property(lambda self: float(self.state[P.S_X]))
    y = property(lambda self: float(self.state[P.S_Y]))
    vx = property(lambda self: float(self.state[P.S_VX]))
    vy = property(lambda self: float(self.state[P.S_VY]))
    screen = property(lambda self: int(self.state[P.S_SCREEN]))
    on_ground = property(lambda self: bool(self.state[P.S_ON_GROUND]))
    settled = property(lambda self: bool(P.is_settled(self.state)))
    done = property(lambda self: bool(self.state[P.S_GOAL]))
    ticks = property(lambda self: int(self.state[P.S_TICK]))

    @property
    def height(self):
        """Height of the player's feet above the bottom of screen 1 (px)."""
        return C.SCREEN_H - (int(self.state[P.S_Y]) + C.PLAYER_H)

    @property
    def max_height(self):
        return C.SCREEN_H - int(self.state[P.S_MAX_HEIGHT])

    @property
    def total_height(self):
        return C.SCREEN_H * self.level.n_screens

    def wind(self):
        """Current wind acceleration (px/tick^2) if the screen is windy."""
        if not self.level.wind[self.screen]:
            return 0.0
        return float(P.wind_velocity(self.state[P.S_TICK], self.state[P.S_WIND_OFFSET],
                                     self.params[P.P_WIND_SCALE]))

    def info(self):
        s = self.state
        return {
            "x": float(s[P.S_X]), "y": float(s[P.S_Y]),
            "vx": float(s[P.S_VX]), "vy": float(s[P.S_VY]),
            "screen": int(s[P.S_SCREEN]),
            "area": self.level.area_names[int(s[P.S_SCREEN])],
            "height": self.height, "max_height": self.max_height,
            "on_ground": bool(s[P.S_ON_GROUND]), "knocked": bool(s[P.S_KNOCKED]),
            "on_ice": bool(s[P.S_ON_ICE]), "on_snow": bool(s[P.S_ON_SNOW]),
            "charging": bool(s[P.S_CHARGING]), "charge": float(s[P.S_CHARGE]),
            "splat": bool(s[P.S_SPLAT]), "wind": self.wind(),
            "ticks": int(s[P.S_TICK]), "jumps": int(s[P.S_JUMPS]),
            "falls": int(s[P.S_FALLS]), "splats": int(s[P.S_SPLATS]),
            "goal": bool(s[P.S_GOAL]),
        }
