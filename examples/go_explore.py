"""Go-Explore on the Jump King clone, built on the snapshot/restore API.

Jump King is a textbook hard-exploration problem: one bad jump undoes
minutes of progress. Go-Explore keeps an archive of *cells* (screen, coarse
x, coarse y), returns to a promising cell by restoring its saved state, and
explores from there with random macro actions. Because ``get_state`` /
``set_state`` are O(1), this climbs the whole built-in map in minutes on a
laptop, and the best trajectory can be exported as a .tas file or used as a
demonstration for imitation learning.

    python examples/go_explore.py --iterations 20000 --save-tas best.tas
    python examples/go_explore.py --ending new_babe_plus --start imp_room
    python examples/go_explore.py --ending ghost_of_the_babe

Cells are scored by progress along the route of the chosen ending (the
order of screens of ``jumpking.routes``), then by height within the screen:
in the full world plain height is misleading, because Redcrown Woods' side
passage leads into Philosopher's Forest, the highest screens of the atlas.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from jumpking import physics as P  # noqa: E402
from jumpking.envs import JumpKingEnv  # noqa: E402


def cell_of(info, cell_px=48):
    return (info["screen"], int(info["x"]) // cell_px, int(info["y"]) // cell_px)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", default="auto")
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--explore-steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ending", default="normal",
                    choices=["normal", "new_babe_plus", "ghost_of_the_babe"])
    ap.add_argument("--start", default="start", help="start | imp_room (New Babe+)")
    ap.add_argument("--save-tas")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = JumpKingEnv(level=args.level, max_episode_steps=10**9, wind_phase=0.0,
                      ending=args.ending, start=args.start)
    obs, info = env.reset(seed=args.seed)
    archive = {}  # cell -> dict(state, score, actions, visits)

    def score(info):
        local = 360 - (int(info["y"]) + 26 + 360 * info["screen"])   # height within the screen
        return info["route_progress"] * 360 + local

    def add(info, state, actions):
        c = cell_of(info)
        e = archive.get(c)
        # prefer reaching a cell with fewer actions
        if e is None or len(actions) < len(e["actions"]):
            archive[c] = {"state": state.copy(), "score": score(info), "actions": list(actions),
                          "visits": 0 if e is None else e["visits"], "info": info,
                          "progress": info["route_progress"]}
            return e is None
        return False

    add(info, env.get_state(), [])
    t0 = time.time()
    best_h = score(info)
    solved = None
    for it in range(args.iterations):
        cells = list(archive.values())
        # selection: favour rarely visited and high cells
        w = np.array([1.0 / np.sqrt(e["visits"] + 1) + 2.0 * (e["score"] >= best_h - 360)
                      for e in cells])
        e = cells[int(rng.choice(len(cells), p=w / w.sum()))]
        e["visits"] += 1
        env.set_state(e["state"])
        env._route.index[0] = e["progress"]
        actions = list(e["actions"])
        for _ in range(args.explore_steps):
            a = int(rng.integers(env.action_space.n))
            obs, r, term, trunc, info = env.step(a)
            actions.append(env.action_meaning(a))
            if info["splat"] and not info["on_ground"]:
                continue
            if term:
                if info["success"]:
                    solved = actions
                break          # another babe: a dead end for this ending
            add(info, env.get_state(), actions)
            if score(info) > best_h:
                best_h = score(info)
        if solved:
            break
        if it % 500 == 0:
            best = max(archive.values(), key=lambda e: e["score"])
            print("iter %6d  cells %5d  route step %3d (screen %3d)  %.0fs" % (
                it, len(archive), best["progress"], best["info"]["screen"] + 1, time.time() - t0))
    if solved:
        print("reached the babe after %d iterations, %d macro actions, %.0fs" % (
            it, len(solved), time.time() - t0))
        best_actions = solved
    else:
        best = max(archive.values(), key=lambda e: e["score"])
        best_actions = best["actions"]
        print("best route step %d (screen %d) after %d iterations" % (
            best["progress"], best["info"]["screen"] + 1, args.iterations))
    if args.save_tas:
        from jumpking import tas
        from jumpking.game import Game

        env.reset(seed=args.seed)
        g = Game(env.level, legacy_teleport=False)
        g.params[:] = env.game.params
        g.restore(env.get_state())
        start = g.snapshot()
        inputs = tas.macro_to_inputs(g, best_actions)
        tas.save(args.save_tas, inputs, header="go-explore", state=start)
        print("saved", args.save_tas)


if __name__ == "__main__":
    main()
