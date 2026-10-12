"""Train a PPO agent (stable-baselines3) on the Jump King clone.

Examples::

    python scripts/train_ppo.py --timesteps 2_000_000 --n-envs 8
    python scripts/train_ppo.py --obs grid --curriculum --timesteps 5e6
    tensorboard --logdir runs

Requires ``pip install -e .[train]`` (torch + stable-baselines3).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402


def make_env(rank, args):
    def _init():
        from jumpking.envs import JumpKingEnv

        env = JumpKingEnv(
            level=args.level, action_mode=args.action_mode, obs_mode=args.obs,
            reward=args.reward, step_penalty=args.step_penalty,
            max_episode_steps=args.episode_steps,
            start="random_screen" if args.curriculum else "start",
            max_start_screen=args.max_start_screen,
        )
        env.reset(seed=args.seed + rank)
        return env

    return _init


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", default="builtin")
    ap.add_argument("--obs", default="vector", choices=["vector", "grid", "dict"])
    ap.add_argument("--action-mode", default="macro", choices=["macro", "frame"])
    ap.add_argument("--reward", default="height", choices=["height", "max_height", "screen", "sparse"])
    ap.add_argument("--step-penalty", type=float, default=0.0)
    ap.add_argument("--episode-steps", type=int, default=300)
    ap.add_argument("--curriculum", action="store_true", help="start episodes on random screens")
    ap.add_argument("--max-start-screen", type=int, default=None)
    ap.add_argument("--timesteps", type=float, default=1e6)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--subproc", action="store_true", help="use one process per env")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    name = args.name or time.strftime("ppo-%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.out, name)
    os.makedirs(run_dir, exist_ok=True)

    fns = [make_env(i, args) for i in range(args.n_envs)]
    venv = SubprocVecEnv(fns) if args.subproc else DummyVecEnv(fns)
    venv = VecMonitor(venv)

    class ProgressCallback(BaseCallback):
        """Logs the best height / screen reached per rollout."""

        def __init__(self):
            super().__init__()
            self.best_height = 0
            self.rollout_best = 0
            self.rollout_screen = 0

        def _on_step(self):
            for info in self.locals["infos"]:
                self.rollout_best = max(self.rollout_best, info.get("max_height", 0))
                self.rollout_screen = max(self.rollout_screen, info.get("screen", 0) + 1)
            return True

        def _on_rollout_end(self):
            self.best_height = max(self.best_height, self.rollout_best)
            self.logger.record("jumpking/rollout_best_height", self.rollout_best)
            self.logger.record("jumpking/rollout_best_screen", self.rollout_screen)
            self.logger.record("jumpking/best_height_ever", self.best_height)
            self.rollout_best = 0
            self.rollout_screen = 0

    policy = {"vector": "MlpPolicy", "grid": "CnnPolicy", "dict": "MultiInputPolicy"}[args.obs]
    model = PPO(policy, venv, n_steps=512, batch_size=1024, n_epochs=4, gamma=0.995,
                gae_lambda=0.95, ent_coef=0.01, learning_rate=3e-4, verbose=1,
                seed=args.seed, tensorboard_log=args.out)
    callbacks = [ProgressCallback(),
                 CheckpointCallback(max(1, 200_000 // args.n_envs), run_dir, name_prefix="ppo")]
    t = time.time()
    model.learn(total_timesteps=int(args.timesteps), callback=callbacks, tb_log_name=name)
    path = os.path.join(run_dir, "final_model")
    model.save(path)
    print("saved %s.zip (%.0fs)" % (path, time.time() - t))
    venv.close()


if __name__ == "__main__":
    main()
