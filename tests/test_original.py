"""Checks against the original game (skipped unless Jump King is installed)."""
import os
import urllib.request

import numpy as np
import pytest

from jumpking import constants as C
from jumpking import physics as P
from jumpking import tas
from jumpking.game import Game
from jumpking.level import IMP_ROOM, Level
from jumpking.xnb import find_game_dir, find_level_xnb

pytestmark = pytest.mark.skipif(find_level_xnb() is None, reason="Jump King is not installed")

TAS_URL = "https://raw.githubusercontent.com/ShootMe/JumpKingTAS/master/LevelFiles/NewGame.tas"


@pytest.fixture(scope="module")
def original():
    return Level.original()


@pytest.fixture(scope="module")
def base_tas(tmp_path_factory):
    path = os.environ.get("JUMPKING_TAS")
    if path and os.path.isfile(path):
        return tas.load(path)
    try:
        text = urllib.request.urlopen(TAS_URL, timeout=20).read().decode("utf-8", "replace")
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip("cannot download the base-game TAS: %s" % exc)
    return tas.parse(text)


def test_only_base_game_screens_are_loaded(original):
    assert original.n_screens == IMP_ROOM + 1
    # wind exactly on Windswept Bluff + Stormwall Pass
    assert [i + 1 for i in np.nonzero(original.wind)[0]] == list(range(26, 33))
    # snow in the windy areas, ice at the end of Chapel Perilous / Blue Ruin
    def screens_with(code):
        return [i + 1 for i in range(original.n_screens) if (original.screen_grid(i) == code).any()]
    assert screens_with(P.T_SNOW) == list(range(26, 32))
    assert screens_with(P.T_ICE) == [36, 37, 38, 39]
    # the only side teleport pair: Bargainburg 4 <-> the hidden Imp Room
    linked = {i + 1: (int(original.tp_left[i]) + 1, int(original.tp_right[i]) + 1)
              for i in range(original.n_screens) if original.tp_left[i] >= 0}
    assert linked == {18: (45, 45), 45: (18, 18)}
    # DLC screens are blank
    assert not original.screen_grid(43).any()


def test_start_position_stands_on_screen_one(original):
    assert original.collides(231, 303) and not original.collides(231, 302)


def test_base_game_tas_reproduced_tick_for_tick(original, base_tas):
    """Replay the community speed-run of the base game (3:44.859) on the real
    map. Any physics deviation would derail the frame-perfect route."""
    start, inputs = tas.to_inputs(base_tas)
    g = Game(original)
    g.reset(start[0], start[1])
    g.state[P.S_TICK] = -2   # the TAS tool's "X" sets the run clock to -2
    g.step(0, False)          # the "@" line itself lasts one frame
    goal_tick = None
    for i, (d, j) in enumerate(inputs):
        if g.step(d, j) & P.EV_GOAL:
            goal_tick = i + 1
            break
    assert goal_tick == len(inputs)
    # the run time matches the recorded 3:44.859 within two frames
    assert abs(len(inputs) * C.TARGET_ELAPSED_SECONDS - 224.859) < 0.04


def test_constants_match_executable():
    pytest.importorskip("dnfile")
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from extract_constants import extract

    exe = os.path.join(find_game_dir(), "JumpKing.exe")
    vals = extract(exe)
    assert vals["SPLAT_TIME"] == pytest.approx(C.SPLAT_TIME)
    assert vals["JUMP_TIME"] == pytest.approx(C.JUMP_TIME)
    assert vals["BOUNCE"] == pytest.approx(C.BOUNCE)
    assert (vals["PLAYER_WIDTH"], vals["PLAYER_HEIGHT"]) == (C.PLAYER_W, C.PLAYER_H)
