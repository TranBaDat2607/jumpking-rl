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
