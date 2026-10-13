import struct

import numpy as np
import pytest

from jumpking import physics as P
from jumpking.level import Level, resolve_slopes
from jumpking.xnb import read_texture


def test_slope_orientation_rules():
    g = np.zeros((5, 5), np.int8)
    g[2, 2] = -1
    g[2, 3] = P.T_SOLID   # right
    g[3, 2] = P.T_SOLID   # below
    assert resolve_slopes(g)[2, 2] == P.T_SLOPE_TL
    g = np.zeros((5, 5), np.int8)
    g[2, 2] = -1
    g[2, 1] = P.T_SOLID
    g[3, 2] = P.T_SOLID
    assert resolve_slopes(g)[2, 2] == P.T_SLOPE_TR
    g = np.zeros((5, 5), np.int8)
    g[2, 2] = -1
    g[2, 3] = P.T_FAKE    # fake blocks orient slopes too
    g[1, 2] = P.T_SOLID
    assert resolve_slopes(g)[2, 2] == P.T_SLOPE_BL
    g = np.zeros((5, 5), np.int8)
    g[2, 2] = -1          # no neighbours: a slope without orientation that
    assert resolve_slopes(g)[2, 2] == P.T_SLOPE_NONE   # never collides
    g = np.zeros((5, 5), np.int8)
    g[2, 0] = -1          # the screen edge counts as solid (IsBlockEdge)
    g[3, 0] = P.T_SOLID
    assert resolve_slopes(g)[2, 0] == P.T_SLOPE_TR
    g = np.zeros((5, 5), np.int8)
    g[2, 2] = -1          # sand orients slopes, water does not
    g[2, 3] = P.T_SAND
    g[3, 2] = P.T_WATER
    g[1, 2] = P.T_SOLID
    assert resolve_slopes(g)[2, 2] == P.T_SLOPE_BL


def test_slopes_are_oriented_across_screen_seams():
    atlas = np.zeros((585, 780, 4), np.uint8)
    # screen 1 (index 0): a slope on its top row; the solid block that orients
    # it sits on the bottom row of screen 2 (index 1), right above it
    atlas[0, 10] = (255, 0, 0, 255)
    atlas[0, 11] = (0, 0, 0, 255)
    atlas[45 + 44, 10] = (0, 0, 0, 255)
    lvl = Level.from_atlas(atlas, n_screens=2)
    assert lvl.screen_grid(0)[0, 10] == P.T_SLOPE_BL


def test_atlas_roundtrip_builtin():
    lvl = Level.builtin()
    atlas = lvl.to_atlas()
    assert atlas.shape == (585, 780, 4)
    again = Level.from_atlas(atlas, goal=lvl.goal, start=lvl.start)
    np.testing.assert_array_equal(lvl.tiles, again.tiles)
    np.testing.assert_array_equal(lvl.wind, again.wind)


def test_atlas_column_major_layout_and_markers():
    atlas = np.zeros((585, 780, 4), np.uint8)
    # screen index 14 (0-based) sits in column 1, row 1 of the atlas
    x0, y0 = 60, 45
    atlas[y0 + 44, x0:x0 + 60] = (0, 0, 0, 255)             # floor
    atlas[y0 + 10, x0 + 30] = (0, 255, 0, 255)              # wind marker
    atlas[y0 + 5, x0 + 2] = (3, 0, 255, 255)                # teleport -> screen 3
    atlas[y0 + 20, x0 + 20] = (0, 255, 255, 255)            # ice
    atlas[y0 + 21, x0 + 20] = (255, 255, 0, 255)            # snow
    atlas[y0 + 22, x0 + 20] = (0, 170, 170, 255)            # water
    atlas[y0 + 23, x0 + 20] = (255, 106, 0, 255)            # sand
    atlas[y0 + 24, x0 + 20] = (182, 255, 0, 255)            # quark
    lvl = Level.from_atlas(atlas, n_screens=15)
    sc = lvl.screen_grid(14)
    assert (sc[44] == P.T_SOLID).all()
    assert sc[20, 20] == P.T_ICE and sc[21, 20] == P.T_SNOW
    assert sc[22, 20] == P.T_WATER and sc[23, 20] == P.T_SAND and sc[24, 20] == P.T_QUARK
    assert lvl.wind[14] == 1 and lvl.wind[13] == 0
    # a teleport pixel in the left half sets the left link; with a single
    # link both exits use it (LevelScreen.IsTwoTeleports)
    assert lvl.tp_left[14] == 2 and lvl.tp_right[14] == -1
    assert lvl.teleport_target(14, exit_left=False) == 2


def test_wind_variants_and_two_teleports():
    atlas = np.zeros((585, 780, 4), np.uint8)
    atlas[44, 0:60] = (0, 0, 0, 255)
    atlas[10, 5] = (192 + 16 + 4, 255, 0, 255)   # intensity 5, constant, leftwards
    atlas[11, 5] = (7, 0, 255, 255)               # left exit -> screen 7
    atlas[11, 50] = (9, 0, 255, 255)              # right exit -> screen 9
    lvl = Level.from_atlas(atlas, n_screens=10)
    assert lvl.wind[0] == 2 + 4 + 16
    assert lvl.teleport_target(0, exit_left=True) == 6
    assert lvl.teleport_target(0, exit_left=False) == 8
    w = P.wind_velocity(0.0, 0.0, 1.0, int(lvl.wind[0]))
    assert w == pytest.approx(-5 * 0.0125)
    again = Level.from_atlas(lvl.to_atlas(), n_screens=10)
    assert again.wind[0] == lvl.wind[0]
    assert (again.tp_left[0], again.tp_right[0]) == (6, 8)


def _write_xnb(path, rgba):
    h, w = rgba.shape[:2]
    reader = b"Microsoft.Xna.Framework.Content.Texture2DReader, Microsoft.Xna.Framework.Graphics"
    body = bytes([1, len(reader)]) + reader + struct.pack("<i", 0) + bytes([0, 1])
    body += struct.pack("<iiiiI", 0, w, h, 1, w * h * 4) + rgba.tobytes()
    data = b"XNBw" + bytes([5, 0]) + struct.pack("<I", 10 + len(body)) + body
    with open(path, "wb") as fh:
        fh.write(data)


def test_xnb_reader_and_level_from_xnb(tmp_path):
    rgba = np.zeros((585, 780, 4), np.uint8)
    rgba[41:45, 0:60] = (0, 0, 0, 255)
    p = tmp_path / "level.xnb"
    _write_xnb(p, rgba)
    np.testing.assert_array_equal(read_texture(str(p)), rgba)
    lvl = Level.from_xnb(str(p))
    assert lvl.n_screens == 43
    assert lvl.collides(231, 303) and not lvl.collides(231, 302)


def test_original_map_if_installed():
    from jumpking.xnb import find_level_xnb

    path = find_level_xnb()
    if path is None:
        pytest.skip("Jump King is not installed")
    lvl = Level.original()
    assert lvl.n_screens >= 43
    # the start position stands on the floor of screen 1 (from the TAS)
    assert lvl.collides(231, 303) and not lvl.collides(231, 302)


def test_achievements_match_the_steam_schema():
    """The 48 codes, names and descriptions equal Steam's schema for the app
    (the data SteamDB displays), read from a local Steam client."""
    from jumpking.achievements import ACHIEVEMENTS, steam_schema

    schema = steam_schema()
    if schema is None:
        pytest.skip("no Steam stats schema cached for Jump King")
    assert sorted(schema) == sorted((a.code, a.name, a.description) for a in ACHIEVEMENTS)
