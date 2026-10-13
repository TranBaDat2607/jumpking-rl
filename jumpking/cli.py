"""Command line entry point: ``jumpking <command>`` (or ``python -m jumpking``)."""
import argparse
import json
import os
import sys
import time


def _level(name):
    from .level import Level

    return Level.load(name)


def cmd_play(a):
    from .play import play

    play(_level(a.level), scale=a.scale, hz=a.hz)


def cmd_replay(a):
    from . import tas
    from .game import Game
    from .play import watch_inputs

    segs = tas.load(a.file)
    start, inputs = tas.to_inputs(segs)
    game = Game(_level(a.level))
    if a.headless:
        trace = tas.replay(game, segs)
        info = game.info()
        print(json.dumps({k: info[k] for k in ("screen", "height", "max_height", "goal", "ticks",
                                                "jumps", "falls")}, indent=1))
        return
    watch_inputs(game, inputs, start=start, speed=a.speed, title=os.path.basename(a.file))


def cmd_demo(a):
    """Watch the verified solution of the built-in map."""
    from . import tas
    from .game import Game
    from .level import BUILTIN_META, Level
    from .play import watch_inputs

    with open(BUILTIN_META, "r", encoding="utf-8") as fh:
        route = [tuple(x) for x in json.load(fh)["route"]]
    game = Game(Level.builtin())
    game.reset()
    game.settle()
    start = game.snapshot()
    inputs = tas.macro_to_inputs(game, route)
    if a.save:
        tas.save(a.save, inputs, header="built-in map solution", state=start)
        print("saved", a.save)
        return
    game.restore(start)
    watch_inputs(game, inputs, start=start, speed=a.speed, title="built-in map solution")


def cmd_preview(a):
    from .render import Renderer, save_image

    lvl = _level(a.level)
    r = Renderer(lvl)
    if a.screens:
        first, last = [int(v) - 1 for v in a.screens.split("-")]
        img = r.tall_preview(first, last, scale=a.scale)
    else:
        img = r.map_preview(scale=a.scale)
    save_image(img, a.out)
    print("wrote", a.out)


def cmd_info(a):
    from . import physics as P
    from .xnb import find_game_dir, find_level_xnb

    print("numba JIT       :", "enabled" if P.NUMBA_ENABLED else "disabled (slow fallback)")
    print("game directory  :", find_game_dir() or "not found")
    print("original map    :", find_level_xnb() or "not found (install Jump King to use level='original')")
    lvl = _level(a.level)
    print("level           :", lvl)
    if lvl.endings:
        for e in lvl.endings:
            print("ending          : %-18s screen %d" % (e.name, e.screen + 1))
    else:
        print("goal (babe)     :", lvl.goal)
    if find_game_dir():
        try:
            from .content import GameContent

            c = GameContent.load()
            print("game content    : %d NPCs, %d merchants, %d birds, %d world items, %d strings"
                  % (len(c.old_men), len(c.merchants), len(c.ravens), len(c.world_items),
                     len(c.strings)))
        except Exception as exc:  # pragma: no cover - depends on install
            print("game content    : unreadable (%s)" % exc)
    names = []
    for n in lvl.area_names:
        if not names or names[-1][0] != n:
            names.append([n, 0])
        names[-1][1] += 1
    for n, k in names:
        print("  %-20s %d screens" % (n, k))


def cmd_achievements(a):
    from .achievements import ACHIEVEMENTS

    unlocked = {}
    if a.profile:
        from .campaign import Profile

        unlocked = Profile.load(a.profile).achievements
    for i, ach in enumerate(ACHIEVEMENTS, 1):
        mark = "x" if ach.code in unlocked else " "
        print("[%s] %2d %-36s %-28s %s" % (mark, i, ach.code, ach.name, ach.how))
    if a.profile:
        print("%d / %d unlocked" % (len(unlocked), len(ACHIEVEMENTS)))


def cmd_benchmark(a):
    import numpy as np

    from .game import Game

    g = Game(_level(a.level))
    g.jump(1, 10)  # JIT warm-up
    rng = np.random.default_rng(0)
    t = time.perf_counter()
    n = 0
    ticks = 0
    while time.perf_counter() - t < a.seconds:
        g.reset()
        g.settle()
        for _ in range(50):
            ev, k = g.jump(int(rng.integers(-1, 2)), int(rng.integers(2, 37)))
            ticks += k
            n += 1
    dt = time.perf_counter() - t
    print("macro actions/s : %.0f" % (n / dt))
    print("game ticks/s    : %.0f  (%.0fx real time)" % (ticks / dt, ticks / dt / 60))
    import gymnasium as gym

    import jumpking.envs  # noqa: F401

    env = gym.make("JumpKing-v0")
    env.reset(seed=0)
    t = time.perf_counter()
    k = 0
    while time.perf_counter() - t < a.seconds:
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        k += 1
        if term or trunc:
            env.reset()
    print("env steps/s     : %.0f (JumpKing-v0, single process)" % (k / (time.perf_counter() - t)))
    from .envs.vector import JumpKingVectorEnv

    v = JumpKingVectorEnv(num_envs=512, level=g.level, seed=0)
    v.reset(seed=0)
    v.step(v.action_space.sample())
    t = time.perf_counter()
    k = 0
    while time.perf_counter() - t < a.seconds:
        v.step(v.action_space.sample())
        k += v.num_envs
    print("batched steps/s : %.0f (JumpKingVectorEnv, 512 kings, numba threads)" % (k / (time.perf_counter() - t)))


def cmd_solve(a):
    from . import solver
    from .game import Game

    g = Game(_level(a.level))
    if a.screen:
        path, end, n = solver.screen_exit_route(g, a.screen - 1, max_nodes=a.max_nodes)
        print("screen %d: %s (expanded %d)" % (a.screen, "solved" if path is not None else "no route", n))
        if path:
            for act in path:
                print("  ", act)
        return
    segs = solver.solve_map(g, max_nodes_per_screen=a.max_nodes, verbose=True)
    print("solved in %d actions" % sum(len(p) for _, p in segs))


def cmd_build_map(a):
    from . import mapgen
    from .level import BUILTIN_META, BUILTIN_PNG

    level, route = mapgen.build(seed=a.seed, verbose=True)
    mapgen.save(level, route, BUILTIN_PNG, BUILTIN_META, a.seed)
    print("saved", BUILTIN_PNG)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jumpking", description="Jump King clone for RL research")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_level(p):
        p.add_argument("--level", default="auto",
                       help="builtin | original | original-base | auto (original if installed) | "
                            "path to .png/.xnb")

    p = sub.add_parser("play", help="play with the keyboard")
    add_level(p)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--hz", type=float, default=60)
    p.set_defaults(fn=cmd_play)

    p = sub.add_parser("replay", help="watch a .tas recording")
    p.add_argument("file")
    add_level(p)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--headless", action="store_true", help="just print the final state")
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("demo", help="watch the verified solution of the built-in map")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--save", help="write the solution as a .tas file instead")
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("preview", help="render the map to an image")
    add_level(p)
    p.add_argument("--out", default="map.png")
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--screens", help="e.g. 1-5 for a vertical strip")
    p.set_defaults(fn=cmd_preview)

    p = sub.add_parser("info", help="environment / level information")
    add_level(p)
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("achievements", help="the 48 Steam achievements and how to unlock them")
    p.add_argument("--profile", help="a saved campaign profile (JSON) to show progress")
    p.set_defaults(fn=cmd_achievements)

    p = sub.add_parser("benchmark", help="simulation speed")
    add_level(p)
    p.add_argument("--seconds", type=float, default=3.0)
    p.set_defaults(fn=cmd_benchmark)

    p = sub.add_parser("solve", help="run the BFS solver")
    add_level(p)
    p.add_argument("--screen", type=int, help="1-based screen to solve (default: whole map)")
    p.add_argument("--max-nodes", type=int, default=20000)
    p.set_defaults(fn=cmd_solve)

    p = sub.add_parser("build-map", help="regenerate the built-in map")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_build_map)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
