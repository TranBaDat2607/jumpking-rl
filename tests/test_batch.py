import numpy as np
import pytest

from jumpking import Game, Level
from jumpking import physics as P
from jumpking.batch import KIND_JUMP, KIND_WALK, BatchSim
from jumpking.envs import JumpKingEnv
from jumpking.envs.vector import JumpKingVectorEnv


def test_batch_matches_single_game():
    lvl = Level.builtin()
    n = 16
    sim = BatchSim(lvl, n)
    games = [Game(lvl) for _ in range(n)]
    for g in games:
        g.reset()
        g.settle()
    np.testing.assert_array_equal(sim.states, np.stack([g.state for g in games]))
    rng = np.random.default_rng(0)
    for _ in range(30):
        kinds = rng.integers(0, 2, n)
        dirs = rng.integers(-1, 2, n)
        dirs[kinds == KIND_WALK] = rng.choice([-1, 1], (kinds == KIND_WALK).sum())
        vals = np.where(kinds == KIND_JUMP, rng.integers(2, 37, n), rng.integers(2, 40, n))
        sim.macro(kinds, dirs, vals)
        for i, g in enumerate(games):
            if kinds[i] == KIND_JUMP:
                g.jump(int(dirs[i]), int(vals[i]), timeout=60 * 20)
            else:
                g.walk(int(dirs[i]), int(vals[i]), timeout=60 * 20)
    np.testing.assert_array_equal(sim.states, np.stack([g.state for g in games]))


def test_batch_frames():
    sim = BatchSim(Level.builtin(), 4)
    sim.frames([1, -1, 0, 1], [0, 0, 0, 1], repeat=10)
    x = sim.states[:, P.S_X]
    assert x[0] > x[2] > x[1]
    assert sim.states[3, P.S_CHARGING] == 1


def test_vector_env_matches_single_env_observations():
    v = JumpKingVectorEnv(num_envs=3, wind_phase="zero", seed=0)
    obs, info = v.reset(seed=0)
    e = JumpKingEnv(wind_phase="zero")
    o1, _ = e.reset(seed=0)
    np.testing.assert_allclose(obs[0], o1, atol=1e-6)
    a = e.macro_index("jump", 1, 20)
    obs, rew, term, trunc, info = v.step(np.array([a, a, a]))
    o1, r1, *_ = e.step(a)
    np.testing.assert_allclose(obs[1], o1, atol=1e-6)
    assert rew[2] == pytest.approx(r1)


def test_vector_env_autoreset():
    v = JumpKingVectorEnv(num_envs=4, max_episode_steps=3, seed=1)
    obs, _ = v.reset(seed=1)
    for k in range(3):
        obs, rew, term, trunc, info = v.step(v.action_space.sample())
    assert trunc.all()
    assert "final_obs" in info and info["final_obs"].shape == obs.shape
    # after the automatic reset every king is back at the start
    np.testing.assert_allclose(obs[:, 0], 231 / 480.0)


def test_sb3_adapter_trains():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from jumpking.envs.vector import SB3VecEnv

    venv = SB3VecEnv(num_envs=8, max_episode_steps=20)
    model = PPO("MlpPolicy", venv, n_steps=16, batch_size=64, n_epochs=1, verbose=0)
    model.learn(total_timesteps=256)
    obs = venv.reset()
    actions, _ = model.predict(obs)
    assert actions.shape == (8,)
