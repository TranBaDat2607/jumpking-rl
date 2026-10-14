import json
import os

import numpy as np
import pytest
from PIL import Image

from jumpking import Game, Level, constants as C, mapgen, solver, tas
from jumpking.level import BUILTIN_META


def _route():
    with open(BUILTIN_META, "r", encoding="utf-8") as fh:
        return [tuple(a) for a in json.load(fh)["route"]]


def test_builtin_map_shape():
    lvl = Level.builtin()
    assert lvl.n_screens == C.BASE_SCREENS
    # wind only in Windswept Bluff and Stormwall Pass (screens 26-32)
    assert [i + 1 for i in np.nonzero(lvl.wind)[0]] == list(range(26, 33))
    # start position stands on the floor of screen 1
    assert lvl.collides(231, 303) and not lvl.collides(231, 302)


def test_readme_overview_is_current():
    if not os.path.exists(mapgen.OVERVIEW_PNG):
        pytest.skip("not a source checkout")
    stored = np.asarray(Image.open(mapgen.OVERVIEW_PNG).convert("RGB"))
    assert np.array_equal(stored, mapgen.render_overview(Level.builtin())), \
        "docs/images/map.png is stale: run scripts/build_map.py"


def test_stored_route_reaches_the_babe():
    g = mapgen.verify_route(Level.builtin(), _route())
    assert g.done
    assert g.screen == C.BASE_SCREENS - 1


def test_route_as_tas_roundtrip(tmp_path):
    lvl = Level.builtin()
    g = Game(lvl)
    g.reset()
    g.settle()
    start = g.snapshot()
    inputs = tas.macro_to_inputs(g, _route())
    path = tmp_path / "route.tas"
    tas.save(str(path), inputs, state=start)
    g2 = Game(lvl)
    tas.replay(g2, tas.load(str(path)))
    assert g2.done


def test_solver_finds_exit_of_first_screens():
    g = Game(Level.builtin())
    for screen in (0, 1):
        g.reset_to_screen(screen)
        g.settle()
        path, end, n = solver.screen_exit_route(g, screen, g.snapshot(), max_nodes=4000)
        assert path is not None, "screen %d" % (screen + 1)
        assert end[2 + 10] >= 0  # state vector returned
