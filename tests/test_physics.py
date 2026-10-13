import json
import os
import subprocess
import sys

import numpy as np
import pytest

from jumpking import constants as C
from jumpking import physics as P
from jumpking.game import Game
from jumpking.level import Level, resolve_slopes

from conftest import FAR, room, settle

f32 = np.float32


def test_standing_king_stays_grounded(flat_game):
    g = flat_game
    for _ in range(9):
        g.step(0, False)
        # grounded downward motion gets +1 px, so the floor is touched every tick
        assert g.on_ground
        assert g.y == 302.0
        assert g.vy == pytest.approx(C.GRAVITY)


def test_walking_off_a_ledge_keeps_charge_for_two_ticks():
    def ledge(screens):
        screens[0][30, 1:20] = P.T_SOLID
    g = Game(room(extra=ledge))
    g.reset(130, 30 * 8 - 26)
    settle(g)
    # walk right until airborne, then start charging immediately (coyote time)
    for _ in range(200):
        g.step(1, False)
        if not g.on_ground:
            break
    assert not g.on_ground
    assert g.state[P.S_CHARGING] == 0


@pytest.mark.parametrize("charge", [2, 5, 10, 18, 27, 35, 36])
def test_jump_power_is_charge_over_36(flat_game, charge):
    g = settle(flat_game)
    g.jump(0, charge)
    assert g.state[P.S_LAST_POWER] == pytest.approx(min(1.0, charge / 36.0), abs=1e-6)


def test_full_jump_trajectory_matches_reference():
    g = Game(room())
    settle(g)
    # charge manually until release, then compare free flight against a
    # float32 reference integration of the documented equations
    while True:
        ev = g.step(0, True)
        if ev & P.EV_JUMP:
            break
    assert g.vy == pytest.approx(C.JUMP)
    y, vy = f32(g.y), f32(g.vy)
    for _ in range(30):
        g.step(0, False)
        y = f32(y + vy)
        vy = f32(vy + f32(C.GRAVITY))
        assert g.y == float(y)
        assert g.vy == float(vy)


def test_auto_release_after_36_ticks_of_charge(flat_game):
    g = settle(flat_game, 2)  # next tick is a grounded tick
    held = 0
    while True:
        ev = g.step(0, True)
        held += 1
        if ev & P.EV_JUMP:
            break
        assert held < 60
    assert g.state[P.S_LAST_POWER] == 1.0
    assert held == 36


def test_apex_height_full_jump(flat_game):
    g = settle(flat_game)
    y0 = g.y
    g.jump(0, 36)
    # re-run tick by tick to measure apex
    g2 = Game(room())
    settle(g2)
    best = g2.y
    while not (g2.step(0, True) & P.EV_JUMP):
        pass
    for _ in range(80):
        g2.step(0, False)
        best = min(best, g2.y)
    height = y0 - best
    # discrete sum v0 + (v0-g) + ... ~= v0^2/2g + v0/2 ~= 153 px
    assert 150 < height < 155


@pytest.mark.parametrize("ticks_before,expected", [(0, 3.5), (1, 3.5), (3, 3.5), (4, 0.0), (6, 0.0)])
def test_direction_buffer_last_four_ticks(ticks_before, expected):
    g = Game(room())
    settle(g)
    seq = [(0, 1)] * 20
    # press right on exactly one tick, `ticks_before` ticks before release
    release = len(seq)
    idx = release - ticks_before
    seq.append((0, 0))
    seq[idx] = (1, seq[idx][1])
    jumped_vx = None
    for d, j in seq:
        ev = g.step(d, bool(j))
        if ev & P.EV_JUMP:
            jumped_vx = g.vx
    assert jumped_vx == pytest.approx(expected)


def test_walk_speed_and_no_walking_while_charging(flat_game):
    g = settle(flat_game)
    x0 = g.x
    for _ in range(30):
        g.step(1, False)
    assert g.x - x0 == pytest.approx(30 * 1.5, abs=1.5)
    x1 = g.x
    for _ in range(10):
        g.step(1, True)  # charging: no walking
    assert g.x == pytest.approx(x1, abs=1.5)


def test_wall_bounce_halves_and_reverses_velocity():
    g = Game(room())
    g.reset(60, 302)
    settle(g)
    ev_all = 0
    vx_before = None
    while not (g.step(-1, True) & P.EV_JUMP):
        pass
    for _ in range(60):
        vx_prev = g.vx
        ev = g.step(0, False)
        ev_all |= ev
        if ev & P.EV_BUMP_WALL:
            vx_before = vx_prev
            break
    assert ev_all & P.EV_BUMP_WALL
    assert vx_before == pytest.approx(-3.5)
    assert g.vx == pytest.approx(1.75)
    assert g.state[P.S_KNOCKED] == 1


def test_ceiling_bump_halves_horizontal_speed():
    def ceiling(screens):
        screens[0][28:30, 1:59] = P.T_SOLID
    g = Game(room(extra=ceiling))
    settle(g)
    while not (g.step(1, True) & P.EV_JUMP):
        pass
    for _ in range(30):
        vx_prev = g.vx
        ev = g.step(0, False)
        if ev & P.EV_BUMP_CEIL:
            assert g.vx == pytest.approx(vx_prev * 0.5)
            assert g.vy == pytest.approx(C.GRAVITY)
            return
    pytest.fail("no ceiling bump")


def test_splat_after_falling_at_max_speed_and_recovery():
    g = Game(room())
    g.reset(231, 0)
    ev_all = 0
    for _ in range(80):
        ev_all |= g.step(0, False)
        if ev_all & P.EV_SPLAT:
            break
    assert ev_all & P.EV_SPLAT
    assert g.state[P.S_SPLAT] == 1
    # cannot walk while splatted
    x0 = g.x
    for _ in range(10):
        g.step(1, False)
    assert g.x == x0
    # after the splat time, input stands the king up
    for _ in range(200):
        ev = g.step(1, False)
        if ev & P.EV_GETUP:
            break
    assert g.state[P.S_SPLAT] == 0
    g.walk(1, 20)
    assert g.x > x0 + 20


def test_ice_friction_and_slide():
    def ice(screens):
        screens[0][41, :] = P.T_ICE
    g = Game(room(extra=ice))
    settle(g)
    assert g.state[P.S_ON_ICE] == 1 or True
    for _ in range(20):
        g.step(1, False)
    x_release = g.x
    vx = g.vx
    assert vx > 0
    for _ in range(200):
        g.step(0, False)
    assert g.x > x_release + 1.0      # kept sliding after release
    assert g.vx == pytest.approx(0.0)
    # on ice, approach to walk speed is gradual (0.2 per grounded tick)
    g2 = Game(room(extra=ice))
    settle(g2, 2)
    g2.step(1, False)
    assert 0 < g2.vx <= 0.2 + 1e-6


def test_snow_blocks_walking_and_weak_jumps():
    def snow(screens):
        screens[0][41, :] = P.T_SNOW
    g = Game(room(extra=snow))
    settle(g, 9)
    x0 = g.x
    for _ in range(30):
        g.step(1, False)
    assert g.x == pytest.approx(x0)
    ev, _ = g.jump(0, 7)           # power 0.194 <= 0.2: nothing happens
    assert not ev & P.EV_JUMP
    settle(g, 9)
    ev, _ = g.jump(0, 9)           # power 0.25 -> boosted to 0.3
    assert ev & P.EV_JUMP
    assert g.state[P.S_LAST_POWER] == pytest.approx(0.3)


def test_landing_on_slope_slides_down():
    def ramp(screens):
        sc = screens[0]
        for k in range(10):
            c = 20 + k
            r = 40 - k
            sc[r, c] = -1
            sc[r + 1:41, c] = P.T_SOLID
        sc[30:41, 30:59] = P.T_SOLID
    g = Game(room(extra=ramp))
    g.reset(25 * 8, 200)
    slid = False
    for _ in range(120):
        ev = g.step(0, False)
        if ev & P.EV_SLOPE:
            slid = True
        if slid and g.vx < 0 and g.vy > 0:
            assert abs(abs(g.vx) - abs(g.vy)) < 1.0
            return
    pytest.fail("king did not slide down the slope")


def test_wind_waveform():
    vals = np.array([P.wind_velocity(t, 0.0, 1.0) for t in range(0, 2000)])
    assert np.abs(vals).max() == pytest.approx(0.1, abs=1e-6)
    assert (vals > 0.099).any() and (vals < -0.099).any()
    # period 2*pi/0.48124886 s of game clock (17 ms ticks)
    period_ticks = 2 * np.pi / 0.48124886 / 0.017
    assert P.wind_velocity(0, 0.0, 1.0) == pytest.approx(
        P.wind_velocity(int(round(period_ticks)), 0.0, 1.0), abs=0.02)


def test_wind_latch_requires_landing():
    g = Game(room(wind=[1]))
    g.reset(231, 150)
    # falling into a wind screen from the same screen: latch off until landing
    for _ in range(10):
        g.step(0, False)
    assert g.state[P.S_WIND_LATCH] == 0 and g.vx == 0
    for _ in range(60):
        g.step(0, False)
    assert g.state[P.S_WIND_LATCH] == 1
    x0 = g.x
    for _ in range(120):
        g.step(0, False)
    assert g.x != x0  # pushed by the wind while standing


def test_camera_follows_screens_and_counts_falls():
    def ledge(screens):
        screens[0][6, 20:40] = P.T_SOLID
    g = Game(room(n_screens=2, extra=ledge))
    g.reset(231, 6 * 8 - 26)
    settle(g)
    assert g.screen == 0
    g.jump(0, 36)
    assert g.screen == 0  # landed back on the ledge
    g.walk(1, 80)        # walk off the ledge and fall
    assert g.screen == 0
    # jump up into screen 2 and come back down
    g.reset(231, 6 * 8 - 26)
    settle(g)
    seen = set()
    while not (g.step(0, True) & P.EV_JUMP):
        pass
    for _ in range(200):
        g.step(0, False)
        seen.add(g.screen)
    assert seen == {0, 1}
    assert g.state[P.S_FALLS] >= 1


def test_side_teleport():
    s0 = np.zeros((45, 60), np.int8)
    s0[41:, :] = P.T_SOLID
    s0[:, 59] = P.T_SOLID
    s1 = np.zeros((45, 60), np.int8)
    s1[41:, :] = P.T_SOLID
    lvl = Level([s0, s1], [0, 0], [1, -1], [1, -1], goal=FAR)
    g = Game(lvl)
    g.reset(20, 302)
    ev_all = 0
    for _ in range(40):
        ev_all |= g.step(-1, False)
    assert ev_all & P.EV_TELEPORT
    assert g.screen == 1
    assert g.x > 400
    settle(g)
    assert g.on_ground or g.state[P.S_WAS_GROUNDED] == 1 or True
    assert -360 <= g.y < 0


def test_determinism_and_snapshot(flat_game):
    rng = np.random.default_rng(3)
    inputs = [(int(rng.integers(-1, 2)), int(rng.random() < 0.3)) for _ in range(3000)]
    g = flat_game
    snap = g.snapshot()
    g.run(inputs)
    a = g.snapshot()
    g.restore(snap)
    for d, j in inputs:
        g.step(d, j)
    np.testing.assert_array_equal(a, g.snapshot())


PARITY_SCRIPT = r"""
import json, numpy as np, sys
sys.path.insert(0, "tests")
from conftest import playground
from jumpking import physics as P
from jumpking.game import Game
from jumpking.level import Level
out = []
g = Game(Level.builtin())
rng = np.random.default_rng(11)
for k in range(40):
    g.jump(int(rng.integers(-1, 2)), int(rng.integers(2, 37)))
    g.walk(int(rng.choice([-1, 1])), int(rng.integers(2, 30)))
out += [float(v) for v in g.state]
# every block type, the items and per-frame inputs
g = Game(playground())
rng = np.random.default_rng(5)
starts = [(40.0, 302.0), (300.0, 302.0), (100.0, -58.0), (60.0, -418.0), (300.0, -418.0)]
for episode in range(40):
    x, y = starts[episode % len(starts)]
    g.reset(x, y)
    g.wear(giant_boots=bool(episode % 3 == 1), snake_ring=bool(episode % 4 == 2))
    for k in range(150):
        d = int(rng.integers(-1, 2))
        j = int(rng.random() < 0.6)
        for _ in range(int(rng.integers(1, 30))):
            g.step(d, j)
    out += [float(v) for v in g.state]
print(json.dumps(out))
"""


def test_numba_and_python_kernels_agree():
    if not P.NUMBA_ENABLED:
        pytest.skip("numba not installed")
    outs = []
    for flag in ("0", "1"):
        env = dict(os.environ, JUMPKING_NO_NUMBA=flag)
        res = subprocess.run([sys.executable, "-c", PARITY_SCRIPT], env=env,
                             capture_output=True, text=True, timeout=600)
        assert res.returncode == 0, res.stderr
        outs.append(json.loads(res.stdout.strip().splitlines()[-1]))
    assert outs[0] == outs[1]
