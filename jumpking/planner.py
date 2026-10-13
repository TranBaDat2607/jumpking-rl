"""Route planning for the campaign: reach places by actually jumping there.

``Planner`` searches macro actions (jumps, walks, waits) with the BFS of
``solver.py`` on a copy of the physics, from the campaign's current state, to
a goal such as "stand on screen k", "touch this box" or "come within reach of
this bird". ``Planner.go`` then plays the found actions in the ``Campaign``
frame by frame, so NPCs talk, birds flee and items are picked up exactly as
they would for a player. Every route is just button presses; nothing is
teleported.

``Planner.log`` keeps every action played, so a whole multi-run playthrough
can be saved and replayed with ``replay`` (``scripts/achievement_run.py``).
"""
import numpy as np

from . import physics as P
from . import solver
from .game import Game


def rect_touch(rect):
    """Goal: the hitbox overlaps ``rect`` (x, y, w, h)."""
    x, y, w, h = rect

    def goal(s):
        hx, hy = int(s[P.S_X]), int(s[P.S_Y])
        return x < hx + P.PW and hx < x + w and y < hy + P.PH and hy < y + h
    return goal


def near_point(px, py, dist):
    """Goal: the hitbox centre is within ``dist`` of (px, py) (a bird's trigger)."""
    d2 = dist * dist

    def goal(s):
        cx = int(s[P.S_X]) + P.PW // 2
        cy = int(s[P.S_Y]) + P.PH // 2
        return (cx - px) ** 2 + (cy - py) ** 2 < d2
    return goal


def on_screen(k):
    def goal(s):
        return int(s[P.S_SCREEN]) == k
    return goal


def screens_between(lo, hi):
    def keep(s):
        return lo <= int(s[P.S_SCREEN]) <= hi
    return keep


class Planner:
    """BFS planning on a private copy of the campaign's physics."""

    def __init__(self, campaign, max_nodes=6000, verbose=False):
        self.camp = campaign
        self.game = Game(campaign.level)
        self.game.params[:] = campaign.game.params
        self.max_nodes = max_nodes
        self.verbose = verbose
        self.log = []                       # macro actions played in the campaign

    def search(self, goal, keep=None, start=None, max_nodes=None, actions=None, landed=False):
        """BFS from ``start`` (default: the campaign's current state)."""
        if start is None:
            start = self.camp.game.snapshot()
        path, end, n = solver.bfs(self.game, start, goal, keep, actions=actions,
                                  max_nodes=max_nodes or self.max_nodes, max_depth=200,
                                  landed=landed)
        if self.verbose:
            print("    search: %s actions, %d nodes" % (None if path is None else len(path), n))
        return path, end

    def play(self, actions):
        """Play logged actions in the campaign, frame by frame: ``jump``,
        ``walk`` and ``wait`` macros, ``frames`` (raw (direction, jump)
        inputs, e.g. a TAS) and ``equip`` (the pause menu)."""
        c = self.camp
        for a in actions:
            kind = a[0]
            landed = len(a) > 3 and bool(a[3])
            if kind == "jump":
                c.jump(a[1], a[2], landed=landed)
            elif kind == "walk":
                c.walk(a[1], a[2], landed=landed)
            elif kind == "wait":
                for _ in range(a[2]):
                    c.step(0, False)
                    if c.ended:
                        break
                c.settle(landed=landed)
            elif kind == "frames":
                for d, j in a[1]:
                    c.step(d, j)
                    if c.ended:
                        break
            elif kind == "equip":
                c.equip(a[1], bool(a[2]))
            else:
                raise ValueError(kind)
            self.log.append(list(a))
            if c.ended:
                break

    def settle(self):
        self.play([("wait", 0, 0)])

    def wait(self, frames):
        self.play([("wait", 0, int(frames))])

    def go(self, goal, keep=None, max_nodes=None, budgets=(1, 4, 16)):
        """Search a route to ``goal`` and play it. Standstill routes are tried
        first; then routes that keep sliding momentum between jumps (ice).
        Raises if none is found."""
        base = max_nodes or self.max_nodes
        for mult in budgets:
            for landed in (False, True):
                if not landed and not P.is_settled(self.camp.game.state):
                    self.settle()
                path, end = self.search(goal, keep, max_nodes=base * mult, landed=landed)
                if path is not None:
                    self.play(path)
                    return path
        raise RuntimeError("no route found from screen %d" % (self.camp.camera + 1))

    def follow(self, screens, final_goal=None, max_nodes=None):
        """Visit ``screens`` (0-based) in order, one screen at a time, staying
        within the neighbourhood of the current and next screen."""
        for nxt in screens:
            cur = self.camp.camera
            if cur == nxt:
                continue
            lo, hi = min(cur, nxt), max(cur, nxt)
            keep = None if abs(nxt - cur) > 1 else screens_between(lo - 1, hi + 1)
            if self.verbose:
                print("  screen %d -> %d" % (cur + 1, nxt + 1))
            self.go(on_screen(nxt), keep, max_nodes)
            if self.camp.ended:
                return
        if final_goal is not None:
            self.go(final_goal, None, max_nodes)


def replay(campaign, runs, progress=None):
    """Replay a saved playthrough (a list of runs: ``mode``,
    ``sacrifice_crown`` and the logged ``actions``) on ``campaign``."""
    for i, run in enumerate(runs):
        campaign.new_game(run["mode"], sacrifice_crown=run.get("sacrifice_crown", True))
        Planner(campaign).play(run["actions"])
        if progress:
            progress(i, run, campaign)
    return campaign
