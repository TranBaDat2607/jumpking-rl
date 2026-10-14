import numpy as np
import pytest

from jumpking import physics as P
from jumpking.game import Game
from jumpking.level import Level, resolve_slopes

FAR = (-1e6, -1e6, -1e6 + 1, -1e6 + 1)


def room(n_screens=1, floor_row=41, walls=True, wind=None, extra=None):
    """Simple level: floor on screen 0 and side walls on every screen."""
    screens = []
    for i in range(n_screens):
        sc = np.zeros((45, 60), np.int8)
        if walls:
            sc[:, 0] = P.T_SOLID
            sc[:, 59] = P.T_SOLID
        if i == 0:
            sc[floor_row:, :] = P.T_SOLID
        screens.append(sc)
    if extra:
        extra(screens)
    screens = [resolve_slopes(s) for s in screens]
    wind = wind or [0] * n_screens
    return Level(screens, wind, [-1] * n_screens, [-1] * n_screens, goal=FAR)


@pytest.fixture
def flat_game():
    return Game(room())


def settle(g, n=6):
    for _ in range(n):
        g.step(0, False)
    return g


def playground():
    """Three screens exercising every block type: slopes of all four
    orientations, ice, snow, water, sand, quark, wind with a no-wind pocket
    and a pair of side teleports."""
    screens = [np.zeros((45, 60), np.int8) for _ in range(3)]
    s0, s1, s2 = screens
    for sc in screens:
        sc[:, 0] = P.T_SOLID
    s0[41:, :] = P.T_SOLID
    for k in range(8):                       # ramps up to the right and left
        s0[40 - k, 4 + k] = -1
        s0[41 - k:41, 4 + k] = P.T_SOLID
        s0[40 - k, 27 - k] = -1
        s0[41 - k:41, 27 - k] = P.T_SOLID
    s0[18, 2:40] = P.T_SOLID                  # ceiling with downward slopes
    for k in range(4):
        s0[19 + k, 10 + k] = -1
        s0[19:19 + k, 10 + k] = P.T_SOLID
        s0[19 + k, 30 - k] = -1
        s0[19:19 + k, 30 - k] = P.T_SOLID
    s0[41, 30:40] = P.T_ICE
    s0[30:41, 42:58] = P.T_WATER
    s0[41, 58:] = P.T_SNOW
    s1[41:, 1:] = P.T_SOLID
    s1[28:41, 4:26] = P.T_SAND
    s1[10:12, 30:42] = P.T_QUARK
    s1[30, 30:50] = P.T_SNOW
    s1[44, 28:59] = P.T_SOLID
    s2[44, 1:20] = P.T_SOLID
    s2[44, 40:] = P.T_SOLID
    s2[20:40, 10:30] = P.T_NOWIND
    screens = [s.copy() for s in screens]
    stacked = resolve_slopes(np.concatenate(screens[::-1], axis=0))
    screens = [stacked[(2 - i) * 45:(3 - i) * 45] for i in range(3)]
    return Level(screens, [0, 0, 1], [-1, -1, 1], [-1, -1, 1], goal=FAR)
