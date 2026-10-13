"""Breadth-first search over *settled* states using macro actions.

A node is a settled player state (standing still, ready for a command). Edges
are macro actions: jumps (direction x charge), short walks and waits (to
change the wind phase). The search is exact with respect to the simulation,
so a found route replays deterministically.

Uses:
* verifying that every screen of a map can be climbed (map generation/tests),
* producing expert demonstrations for imitation learning or replay buffers,
* measuring the difficulty of a screen (number of distinct winning actions).
"""
from collections import deque

import numpy as np

from . import physics as P

JUMP_CHARGES = tuple(range(2, 37))
WALK_TICKS = (2, 6, 16, 40)
WAIT_TICKS = (48, 192)


def default_actions(wind=False, charges=JUMP_CHARGES, walks=WALK_TICKS):
    acts = []
    for d in (-1, 0, 1):
        for c in charges:
            acts.append(("jump", d, c))
    for d in (-1, 1):
        for t in walks:
            acts.append(("walk", d, t))
    if wind:
        for t in WAIT_TICKS:
            acts.append(("wait", 0, t))
    return acts


def apply_action(game, action, landed=False):
    kind, d, v = action[:3]
    if len(action) > 3:
        landed = bool(action[3])
    if kind == "jump":
        return game.jump(d, v, landed=landed)
    if kind == "walk":
        return game.walk(d, v, landed=landed)
    # wait: idle for v ticks then settle
    ev = game.run([(0, 0)] * v) if v else 0
    e2, n = game.settle(landed=landed)
    return ev | e2, v + n


def state_key(s, wind_screen, wind_bucket=96):
    key = (int(round(s[P.S_X] * 4)), int(s[P.S_Y]), int(s[P.S_SCREEN]),
           int(s[P.S_ON_ICE]), int(round(s[P.S_VX] * 20)))
    if wind_screen:
        key += (int(s[P.S_TICK]) // wind_bucket,)
    return key


def bfs(game, start_state, success, keep=None, actions=None, max_nodes=20000,
        max_depth=60, return_path=True, landed=False):
    """Generic BFS.

    success(state) -> bool : goal test on settled states
    keep(state) -> bool    : prune states that leave the region of interest
    landed                 : nodes are landings (possibly still sliding on ice)
                             instead of standstills, so momentum can be kept;
                             actions then carry a 4th element ``True``
    Returns (path, goal_state, n_expanded) where path is a list of actions or
    None when no route was found within the budget.
    """
    lvl = game.level
    saved = game.snapshot()
    start = np.array(start_state, dtype=np.float64)
    wind_any = bool(lvl.wind.any())
    if actions is None:
        actions = default_actions(wind=wind_any)
    key0 = state_key(start, wind_any and lvl.wind[int(start[P.S_SCREEN])])
    parents = {key0: (None, None)}
    states = {key0: start}
    frontier = deque([(key0, 0)])
    expanded = 0
    try:
        if success(start):
            return [], start, 0
        while frontier and expanded < max_nodes:
            key, depth = frontier.popleft()
            if depth >= max_depth:
                continue
            expanded += 1
            base = states[key]
            for a in actions:
                game.restore(base)
                if landed:
                    a = tuple(a[:3]) + (True,)
                apply_action(game, a)
                s = game.state
                if not (P.is_landed(s) if landed else P.is_settled(s)):
                    continue  # timed out mid-air (stuck sliding etc.)
                if s[P.S_SPLAT] > 0 and s[P.S_SPLAT_DONE] == 0:
                    continue
                if keep is not None and not keep(s):
                    continue
                k = state_key(s, wind_any and lvl.wind[int(s[P.S_SCREEN])])
                if k in parents:
                    continue
                parents[k] = (key, a)
                states[k] = s.copy()
                if success(s):
                    path = []
                    if return_path:
                        cur = k
                        while parents[cur][0] is not None:
                            pk, act = parents[cur]
                            path.append(act)
                            cur = pk
                        path.reverse()
                    return path, states[k], expanded
                frontier.append((k, depth + 1))
        return None, None, expanded
    finally:
        game.restore(saved)


def screen_exit_route(game, screen, start_state=None, max_nodes=20000):
    """Find a route from ``start_state`` (or the screen's first spawn point)
    to a settled state on any higher screen without falling below ``screen``."""
    if start_state is None:
        game.reset_to_screen(screen)
        start_state = game.snapshot()

    def success(s):
        return s[P.S_SCREEN] > screen or s[P.S_GOAL] > 0

    def keep(s):
        return s[P.S_SCREEN] >= screen

    return bfs(game, start_state, success, keep, max_nodes=max_nodes)


def solve_map(game, start_state=None, max_nodes_per_screen=20000, verbose=False):
    """Climb the whole map screen by screen. Returns a list of
    (screen, actions) segments, or raises RuntimeError at the first screen
    that cannot be left upwards."""
    if start_state is None:
        game.reset()
        start_state = game.snapshot()
    state = np.array(start_state)
    segments = []
    while state[P.S_GOAL] == 0:
        screen = int(state[P.S_SCREEN])
        path, end, n = screen_exit_route(game, screen, state, max_nodes_per_screen)
        if path is None:
            raise RuntimeError("no route out of screen %d (expanded %d nodes)" % (screen + 1, n))
        if verbose:
            print("screen %2d: %2d actions, %5d nodes expanded" % (screen + 1, len(path), n))
        segments.append((screen, path))
        state = end
    return segments


def replay(game, segments, start_state=None):
    """Replay solver output; returns the final game state."""
    if start_state is None:
        game.reset()
    else:
        game.restore(start_state)
    for _, path in segments:
        for a in path:
            apply_action(game, a)
    return game.snapshot()
