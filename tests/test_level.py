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
    g[2, 2] = -1          # no neighbours: degrades to a box
    assert resolve_slopes(g)[2, 2] == P.T_SOLID


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
    lvl = Level.from_atlas(atlas, n_screens=15)
    sc = lvl.screen_grid(14)
    assert (sc[44] == P.T_SOLID).all()
    assert sc[20, 20] == P.T_ICE and sc[21, 20] == P.T_SNOW
    assert lvl.wind[14] == 1 and lvl.wind[13] == 0
    assert lvl.tp_left[14] == 2 and lvl.tp_right[14] == 2


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
