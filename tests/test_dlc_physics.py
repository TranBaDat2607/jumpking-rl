"""DLC blocks and items (water, sand, quark, wind variants, Giant Boots,
Snake Ring), endings and camera rules. Expected values follow the game's
behaviours (docs/RESEARCH.md)."""
import numpy as np
import pytest

from jumpking import constants as C
from jumpking import physics as P
from jumpking.game import Game
from jumpking.level import Ending, Level

from conftest import FAR, room


def _level(extra, n_screens=1, wind=None, endings=()):
    lvl = room(n_screens=n_screens, extra=extra, wind=wind)
    if endings:
        lvl.endings = tuple(endings)
    return lvl


def _stand(g, x, y):
    g.reset(x, y)
    for _ in range(3):
        g.step(0, False)
    return g


def _full_charge_ticks(g):
    n = 0
    while True:
        n += 1
        if g.step(0, True) & P.EV_JUMP:
            return n
        assert n < 500


def test_water_halves_gravity_movement_and_charge():
    def pool(screens):
        screens[0][20:41, 1:59] = P.T_WATER
    g = _stand(Game(_level(pool)), 200, 302)
    assert g.in_water
    # a full charge takes twice as long (JumpState: timer += dt * 0.5, in
    # float32, which needs one extra tick to reach 0.6 s)
    t, n = np.float32(0), 0
    while t < np.float32(C.JUMP_TIME):
        t = np.float32(t + np.float32(np.float32(C.DT) * np.float32(0.5)))
        n += 1
    assert n == 73
    assert _full_charge_ticks(g) == n
    y0, vy = g.y, g.vy
    assert vy == pytest.approx(C.JUMP)
    g.step(0, False)
    # the move is halved and gravity is halved
    assert g.y - y0 == pytest.approx(C.JUMP * 0.5, abs=1e-4)
    assert g.vy == pytest.approx(C.JUMP + C.GRAVITY * 0.5, abs=1e-5)
    # outside water a full charge is 36 ticks
    g2 = _stand(Game(room()), 200, 302)
    assert _full_charge_ticks(g2) == 36


def test_sand_sinks_slowly_and_allows_jumping_out():
    def pit(screens):
        screens[0][30:41, 10:40] = P.T_SAND
    g = Game(_level(pit))
    g.reset(150, 100)
    vys = []
    for _ in range(200):
        g.step(0, False)
        if g.in_sand:
            vys.append(g.vy)
    assert vys and max(vys) <= C.SAND_FALL_SPEED + 1e-6
    assert g.in_sand and not g.on_ground
    # still counts as ground for the controls: a jump works, upward motion is halved
    y0 = g.y
    for _ in range(10):
        g.step(0, True)
    g.step(0, False)
    assert g.state[P.S_JUMPS] == 1
    g.step(0, False)
    assert g.y < y0


def test_sand_blocks_from_below_and_slows_walking():
    def pit(screens):
        screens[0][38:41, 20:40] = P.T_SAND
    g = _stand(Game(_level(pit)), 100, 278)   # standing left of the sand on the floor
    g.reset(200, 274)                          # inside the sand, feet on the floor
    for _ in range(5):
        g.step(0, False)
    x0 = g.x
    g.step(1, False)
    g.step(1, False)
    # walking 1.5 px/tick is applied at 0.25x inside sand
    assert g.x - x0 == pytest.approx(1.5 * 0.25, abs=1e-5)


def test_quark_snaps_a_max_speed_fall_to_the_grid():
    def quark(screens):
        screens[0][30, 20:40] = P.T_QUARK
    g = Game(_level(quark))
    g.reset(200, -400.3)
    snapped = False
    for _ in range(120):
        g.step(0, False)
        if 0 < g.y < 300 and g.vy == C.MAX_FALL and float(g.y).is_integer() and int(g.y) % 8 == 0:
            snapped = True
    assert snapped


def test_constant_wind_variant_pushes_one_way():
    lvl = room(wind=[2 + 2 + 16 * 2])      # intensity 3, constant, rightwards
    g = _stand(Game(lvl), 200, 302)
    assert g.wind() == pytest.approx(3 * 0.0125)
    x0 = g.x
    for _ in range(30):
        g.step(0, False)
    assert g.x > x0


def test_giant_boots_disable_walking_and_freeze_on_landing():
    g = _stand(Game(room()), 200, 302).wear(giant_boots=True)
    x0 = g.x
    for _ in range(20):
        g.step(1, False)
    assert g.x == x0
    # jump, then count the ticks the body stays frozen after landing
    for _ in range(10):
        g.step(0, True)
    g.step(0, False)
    frozen = 0
    for _ in range(80):
        g.step(0, False)
        if g.state[P.S_BODY_OFF]:
            frozen += 1
    # PauseNode(0.05) at 60 Hz: the body skips two updates
    assert frozen == 2
    assert g.on_ground


def test_snake_ring_makes_floors_icy_and_ignores_snow():
    def snow(screens):
        screens[0][41, :] = P.T_SNOW
    g = _stand(Game(_level(snow)), 200, 302)
    # on snow a weak jump (power <= 0.2) does nothing and walking is impossible
    g.step(1, False)
    assert g.vx == 0
    g.wear(snake_ring=True)
    for _ in range(10):
        g.step(1, False)
    assert g.vx > 0                     # walking works and accelerates like on ice
    assert g.vx < C.WALK_SPEED
    for _ in range(5):
        g.step(0, False)
    assert 0 < g.vx < C.WALK_SPEED      # and slides after letting go


def test_early_exit_when_starting_inside_a_wall():
    g = Game(room())
    g.reset(200, 340)                  # hitbox inside the floor
    g.state[P.S_VX] = 1.0
    g.state[P.S_VY] = 2.0
    g.step(0, False)
    assert (g.x, g.y) == (201.0, 342.0)
    assert g.vy == 2.0                 # no gravity on an early exit


def test_babe_check_runs_before_the_tick_and_needs_ground():
    lvl = room()
    lvl.endings = (Ending("normal", 0, +1, 300),)
    g = Game(lvl)
    _stand(g, 295, 302)               # centre 304 >= 300
    # the win was detected on the first tick that started on the ground
    assert g.done and g.ending == 1


def test_camera_follows_a_grounded_king_up_and_skips_ending_screens():
    def floor(screens):
        screens[1][30, :] = P.T_SOLID

    def stand_above(lvl):
        g = Game(lvl)
        # standing on screen 2's floor while the camera still shows screen 1
        g.reset(200, -360 + 240 - 26, screen=0)
        g.state[P.S_ON_GROUND] = 1
        g.step(0, False)
        return g

    # a grounded king always pulls the camera to his screen (forced update)
    assert stand_above(room(n_screens=3, extra=floor)).screen == 1
    # ... except onto the screen right above a babe (Camera.UpdateCamera)
    lvl = room(n_screens=3, extra=floor)
    lvl.endings = (Ending("normal", 0, +1, 1e9),)
    assert stand_above(lvl).screen == 0
