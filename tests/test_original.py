"""Checks against the original game (skipped unless Jump King is installed)."""
import os

import numpy as np
import pytest

from jumpking import constants as C
from jumpking import physics as P
from jumpking import verify
from jumpking.level import IMP_ROOM, Level
from jumpking.xnb import find_game_dir, find_level_xnb

pytestmark = pytest.mark.skipif(find_level_xnb() is None, reason="Jump King is not installed")


@pytest.fixture(scope="module")
def world():
    return Level.original()


@pytest.fixture(scope="module")
def base_only():
    return Level.original(content="base")


def _route_text(route):
    """A local copy (JUMPKING_TAS_DIR, or JUMPKING_TAS for the base route) or
    a download from ShootMe/JumpKingTAS."""
    folder = os.environ.get("JUMPKING_TAS_DIR")
    path = os.path.join(folder, route.filename) if folder else None
    if route.filename == "NewGame.tas" and os.environ.get("JUMPKING_TAS"):
        path = os.environ["JUMPKING_TAS"]
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    try:
        return verify.download(route, timeout=20)
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip("cannot download %s: %s" % (route.filename, exc))


def test_whole_world_is_loaded(world):
    assert world.n_screens == C.ATLAS_SCREENS
    assert [e.name for e in world.endings] == ["normal", "new_babe_plus", "ghost_of_the_babe"]
    assert [e.screen + 1 for e in world.endings] == [43, 100, 154]
    # wind: Windswept Bluff + Stormwall Pass, and Underburg in New Babe+
    assert [i + 1 for i in np.nonzero(world.wind)[0]] == list(range(26, 33)) + list(range(64, 71))
    # DLC blocks where the DLCs put them
    assert world.screens_with(P.T_WATER) == list(range(90, 96)) + list(range(102, 109)) + \
        list(range(140, 147))
    assert world.screens_with(P.T_SAND) == list(range(148, 155))
    assert world.screens_with(P.T_QUARK) == [109, 146, 147, 148]
    assert world.screens_with(P.T_NOWIND) == list(range(65, 71)) + [79]
    assert world.screens_with(P.T_ICE) == [36, 37, 38, 39] + list(range(52, 59)) + \
        list(range(117, 123))
    # every side teleport of the world
    links = {}
    for i in range(world.n_screens):
        a, b = world.teleport_target(i, True), world.teleport_target(i, False)
        if a >= 0 or b >= 0:
            links[i + 1] = (a + 1, b + 1)
    assert links == {
        3: (156, 156), 156: (3, 3),          # Redcrown Woods <-> Philosopher's Forest
        18: (45, 45), 45: (18, 18),          # Bargainburg <-> the Imp Room
        46: (47, 47), 47: (46, 46),          # New Babe+ entrance
        160: (164, 164), 164: (160, 160),    # Philosopher's Forest <-> the Hole
        161: (108, 108), 108: (161, 161),    # the Hole <-> the Bog
    }


def test_base_content_mode(base_only):
    assert base_only.n_screens == IMP_ROOM + 1
    assert [e.name for e in base_only.endings] == ["normal"]
    assert [i + 1 for i in np.nonzero(base_only.wind)[0]] == list(range(26, 33))
    assert base_only.screens_with(P.T_SNOW) == list(range(26, 32))
    assert base_only.screens_with(P.T_ICE) == [36, 37, 38, 39]
    assert base_only.teleport_target(17, True) == IMP_ROOM
    assert not base_only.screen_grid(43).any()


def test_start_position_stands_on_screen_one(world):
    assert world.collides(231, 303) and not world.collides(231, 302)


@pytest.mark.parametrize("key", ["base", "nbp", "ghost"])
def test_tas_route_reproduced_tick_for_tick(world, key):
    """Replay the community speed-runs on the real world. Every section's
    screen must be visited and the babe reached on the exact frame of the
    recorded in-game timer (3:44.859, 5:20.637 and 5:35.138)."""
    route = verify.ROUTES[key]
    rep = verify.replay(world, route, _route_text(route))
    missed = [s for s in rep.sections if not s[3]]
    assert not missed, missed[:3]
    assert rep.ending == route.ending
    assert rep.exact, (rep.win_time, rep.recorded)


def test_ghost_route_needs_the_2019_teleport():
    """Today's game truncates y on side teleports; the 2019 Ghost route only
    works with the old behaviour (see verify.py)."""
    world = Level.original()
    route = verify.ROUTES["ghost"]
    text = _route_text(route)
    modern = verify.Route(route.name, route.filename, route.sections, route.ending,
                          route.start_tick, legacy_teleport=False)
    assert not verify.replay(world, modern, text).exact


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
