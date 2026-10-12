"""Evaluate a policy on the Jump King clone.

    python scripts/evaluate.py --model runs/ppo-.../final_model.zip --episodes 20
    python scripts/evaluate.py --random --episodes 50          # baseline
    python scripts/evaluate.py --model ... --watch             # pygame window

Reports success rate, best/mean height and screens reached, and can save the
best episode as a .tas recording.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="stable-baselines3 .zip")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--level", default="builtin")
    ap.add_argument("--obs", default="vector")
    ap.add_argument("--episode-steps", type=int, default=1000)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--save-tas", help="write the best episode as .tas")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from jumpking import tas
    from jumpking.envs import JumpKingEnv

    env = JumpKingEnv(level=args.level, obs_mode=args.obs, max_episode_steps=args.episode_steps)
    if args.model:
        from stable_baselines3 import PPO

        model = PPO.load(args.model)
        policy = lambda o: int(model.predict(o, deterministic=args.deterministic)[0])  # noqa: E731
    else:
        rng = np.random.default_rng(args.seed)
        policy = lambda o: int(rng.integers(env.action_space.n))  # noqa: E731

    if args.watch:
        from jumpking.play import watch_policy

        watch_policy(env, policy, episodes=args.episodes)
        return

    results = []
    best = (-1, None, None)
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        start = env.get_state()
        actions = []
        done = False
        while not done:
            a = policy(obs)
            actions.append(env.action_meaning(a))
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
        results.append({"max_height": info["max_height"], "screen": info["screen"] + 1,
                        "goal": info["goal"], "jumps": info["jumps"], "falls": info["falls"],
                        "ticks": info["ticks"]})
        if info["max_height"] > best[0]:
            best = (info["max_height"], start, actions)
        print("episode %3d  best height %6d  screen %2d  goal %s" % (
            ep, info["max_height"], info["screen"] + 1, info["goal"]))
    h = np.array([r["max_height"] for r in results])
    summary = {
        "episodes": len(results),
        "success_rate": float(np.mean([r["goal"] for r in results])),
        "max_height_mean": float(h.mean()),
        "max_height_best": int(h.max()),
        "best_screen": int(max(r["screen"] for r in results)),
    }
    print(json.dumps(summary, indent=1))
    if args.save_tas and best[1] is not None:
        from jumpking.game import Game

        g = Game(env.level)
        g.restore(best[1])
        inputs = tas.macro_to_inputs(g, best[2])
        tas.save(args.save_tas, inputs, header="evaluation episode", state=best[1])
        print("saved", args.save_tas)


if __name__ == "__main__":
    main()
