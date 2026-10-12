"""Batched simulation: advance many kings with one (multi-threaded) call.

``BatchSim`` keeps N state vectors in one array and applies macro actions to
all of them inside numba (``prange`` over the batch), avoiding per-env Python
overhead. It is the backend of ``JumpKingVectorEnv`` and is handy for
population-based methods (genetic algorithms, ES, CEM), Go-Explore and MCTS.

    sim = BatchSim(Level.builtin(), n=1024)
    events, ticks = sim.macro(kind, direction, value)   # arrays of shape (n,)
"""
import numpy as np

from . import constants as C
from . import physics as P

KIND_JUMP = 0
KIND_WALK = 1

if P.NUMBA_ENABLED:
    import numba

    _prange = numba.prange

    def _pjit(fn):
        return numba.njit(cache=True, parallel=True, nogil=True)(fn)
else:  # pragma: no cover - slow fallback
    _prange = range

    def _pjit(fn):
        return fn


@_pjit
def _batch_macro(states, kinds, dirs, vals, mask, tiles, wind_en, tp_left, tp_right, params,
                 max_ticks, events_out, ticks_out):
    n = states.shape[0]
    for i in _prange(n):
        if not mask[i]:
            events_out[i] = 0
            ticks_out[i] = 0
            continue
        s = states[i]
        if kinds[i] == KIND_JUMP:
            ev, k = P.macro_jump(s, dirs[i], vals[i], tiles, wind_en, tp_left, tp_right, params, max_ticks)
        else:
            ev, k = P.macro_walk(s, dirs[i], vals[i], tiles, wind_en, tp_left, tp_right, params, max_ticks)
        events_out[i] = ev
        ticks_out[i] = k


@_pjit
def _batch_frames(states, dirs, jumps, repeat, tiles, wind_en, tp_left, tp_right, params, events_out):
    n = states.shape[0]
    for i in _prange(n):
        ev = 0
        for _ in range(repeat):
            ev |= P.tick(states[i], dirs[i], jumps[i], tiles, wind_en, tp_left, tp_right, params)
        events_out[i] = ev


class BatchSim:
    def __init__(self, level, n, splat_time=C.SPLAT_TIME, wind_scale=1.0):
        if isinstance(level, str):
            from .level import Level

            level = Level.load(level)
        self.level = level
        self.n = int(n)
        self.params = level.params(splat_time=splat_time, wind_scale=wind_scale)
        self.states = np.zeros((self.n, P.STATE_SIZE), np.float64)
        self._args = (level.tiles, level.wind, level.tp_left, level.tp_right, self.params)
        self._ev = np.zeros(self.n, np.int64)
        self._ticks = np.zeros(self.n, np.int64)
        self.reset()

    def reset(self, idx=None, x=None, y=None, wind_offsets=None):
        """Reset all (or the given indices) to the start position."""
        from .game import Game

        g = Game(self.level)
        idx = np.arange(self.n) if idx is None else np.asarray(idx)
        for j, i in enumerate(idx):
            g.params[P.P_WIND_OFFSET] = 0.0 if wind_offsets is None else wind_offsets[j]
            g.reset(x, y)
            g.settle()
            self.states[i] = g.state
        return self

    def macro(self, kinds, dirs, vals, mask=None, max_ticks=60 * 20):
        """Apply one macro action per king. ``kinds``: 0 jump / 1 walk."""
        kinds = np.broadcast_to(np.asarray(kinds, np.int64), (self.n,)).copy()
        dirs = np.broadcast_to(np.asarray(dirs, np.int64), (self.n,)).copy()
        vals = np.broadcast_to(np.asarray(vals, np.int64), (self.n,)).copy()
        mask = np.ones(self.n, np.bool_) if mask is None else np.asarray(mask, np.bool_)
        _batch_macro(self.states, kinds, dirs, vals, mask, *self._args, int(max_ticks),
                     self._ev, self._ticks)
        return self._ev.copy(), self._ticks.copy()

    def frames(self, dirs, jumps, repeat=1):
        """Hold the given buttons for ``repeat`` ticks on every king."""
        dirs = np.broadcast_to(np.asarray(dirs, np.int64), (self.n,)).copy()
        jumps = np.broadcast_to(np.asarray(jumps, np.int64), (self.n,)).copy()
        _batch_frames(self.states, dirs, jumps, int(repeat), *self._args, self._ev)
        return self._ev.copy()

    # ------------------------------------------------------------- queries
    @property
    def heights(self):
        return C.SCREEN_H - (self.states[:, P.S_Y].astype(np.int64) + C.PLAYER_H)

    @property
    def max_heights(self):
        return C.SCREEN_H - self.states[:, P.S_MAX_HEIGHT].astype(np.int64)

    @property
    def screens(self):
        return self.states[:, P.S_SCREEN].astype(np.int64)

    @property
    def done(self):
        return self.states[:, P.S_GOAL] > 0
