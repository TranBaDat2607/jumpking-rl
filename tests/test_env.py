import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import jumpking.envs  # noqa: F401
from jumpking import physics as P
from jumpking.envs import JumpKingEnv

IDS = ["JumpKing-v0", "JumpKingGrid-v0", "JumpKingFrame-v0", "JumpKingPixels-v0"]


@pytest.mark.parametrize("env_id", IDS)
def test_gymnasium_api(env_id):
    env = gym.make(env_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(env.unwrapped, skip_render_check=True)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(20):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(r)
    env.close()


def test_dict_observations():
    env = JumpKingEnv(obs_mode="dict")
    obs, _ = env.reset(seed=1)
    assert set(obs) == {"vector", "grid"}
    assert obs["grid"].shape == (3, 90, 60)
    assert obs["grid"][2].sum() > 0  # player visible


def test_same_seed_same_trajectory():
    def rollout(seed):
        env = JumpKingEnv()
        env.reset(seed=seed)
        rng = np.random.default_rng(5)
        out = []
        for _ in range(50):
            o, r, *_ = env.step(int(rng.integers(env.action_space.n)))
            out.append(np.concatenate([o, [r]]))
        return np.array(out)

    np.testing.assert_array_equal(rollout(3), rollout(3))


def test_height_reward_is_potential_based():
    env = JumpKingEnv(reward="height", goal_bonus=0.0)
    obs, info = env.reset(seed=0)
    h0 = info["height"]
    total = 0.0
    for _ in range(100):
        _, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
    assert total == pytest.approx((info["height"] - h0) / 360.0)


def test_get_set_state_and_macro_index():
    env = JumpKingEnv()
    env.reset(seed=0)
    a = env.macro_index("jump", 1, 20)
    assert env.action_meaning(a) == ("jump", 1, 20)
    s = env.get_state()
    o1, r1, *_ = env.step(a)
    env.set_state(s)
    o2, r2, *_ = env.step(a)
    np.testing.assert_array_equal(o1, o2)
    assert r1 == r2


def test_state_carries_wind_phase():
    env = JumpKingEnv(start=25, wind_phase="random")
    env.reset(seed=4)
    s = env.get_state()
    a = env.macro_index("jump", 1, 30)
    o1, *_ = env.step(a)
    env.reset(seed=5)          # different wind phase in the env parameters
    env.set_state(s)
    o2, *_ = env.step(a)
    np.testing.assert_array_equal(o1, o2)


def test_curriculum_start_screens():
    env = JumpKingEnv(start="random_screen", max_start_screen=10)
    seen = set()
    for i in range(30):
        _, info = env.reset(seed=i)
        seen.add(info["screen"])
        assert info["on_ground"]
    assert max(seen) <= 10 and len(seen) > 3
    _, info = env.reset(options={"screen": 20})
    assert info["screen"] == 20


def test_goal_terminates_episode():
    env = JumpKingEnv(goal_bonus=10.0)
    env.reset(seed=0)
    s = env.get_state()
    if env.level.endings:
        # stand where the normal ending triggers (EndingManager.CheckWin)
        e = env.level.ending("normal")
        spots = [p for p in env.game.spawn_points(e.screen, stride=1)
                 if int(p[0]) + 9 >= e.x]
        px, py = spots[0]
        d = 1
        s[P.S_SCREEN] = e.screen
    else:
        x0, y0, x1, y1 = env.level.goal
        py = int(y1) - 26
        # stand next to the babe on whichever side has floor, then walk to her
        for px, d in ((x0 - 24, 1), (x1 + 6, -1)):
            if env.level.collides(int(px), py + 1) and not env.level.collides(int(px), py):
                break
        s[P.S_SCREEN] = env.level.n_screens - 1
    s[P.S_X] = px
    s[P.S_Y] = py
    s[P.S_ON_GROUND] = 1
    env.set_state(s)
    term = False
    for _ in range(5):
        _, r, term, trunc, info = env.step(env.macro_index("walk", d, 32))
        if term:
            break
    assert term and info["goal"] and info["success"]
    assert r >= 10.0


def test_other_babe_ends_the_game_without_success():
    from jumpking.level import Level

    try:
        world = Level.original()
    except FileNotFoundError:
        pytest.skip("Jump King is not installed")
    env = JumpKingEnv(level=world, ending="new_babe_plus", goal_bonus=10.0)
    env.reset(seed=0)
    s = env.get_state()
    e = world.ending("normal")
    px, py = [p for p in env.game.spawn_points(e.screen, stride=1) if int(p[0]) + 9 >= e.x][0]
    s[P.S_X], s[P.S_Y], s[P.S_SCREEN], s[P.S_ON_GROUND] = px, py, e.screen, 1
    env.set_state(s)
    _, r, term, _, info = env.step(env.macro_index("walk", 1, 4))
    assert term and info["goal"] and info["ending"] == 1 and not info["success"]
    assert r < 10.0


def test_vector_env():
    envs = gym.make_vec("JumpKing-v0", num_envs=4, vectorization_mode="sync")
    obs, info = envs.reset(seed=0)
    assert obs.shape == (4, len(jumpking.envs.VECTOR_FEATURES))
    obs, r, term, trunc, info = envs.step(envs.action_space.sample())
    assert r.shape == (4,)
    envs.close()
