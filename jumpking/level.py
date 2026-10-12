"""Level data: tile grid, per-screen wind flags, side teleports and the goal.

Levels use Jump King's native hitbox atlas format: a 780x585 image holding
13x13 screens (column-major), each 60x45 pixels where one pixel is one 8x8
tile. The built-in map ships as ``data/level.png``; the original map can be
read from a local Steam install (``Content/level.xnb``, see ``xnb.py``).
"""
import os
import warnings

import numpy as np

from . import constants as C
from . import physics as P

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BUILTIN_PNG = os.path.join(DATA_DIR, "level.png")
BUILTIN_META = os.path.join(DATA_DIR, "level_meta.json")

# --- native block colours (base game subset) ---------------------------------
RGB_SOLID = (0, 0, 0)
RGB_SLOPE = (255, 0, 0)
RGB_FAKE = (128, 128, 128)
RGB_ICE = (0, 255, 255)
RGB_SNOW = (255, 255, 0)
RGB_WIND = (0, 255, 0)
RGB_NOWIND = (255, 255, 255)
# teleport: (R, 0, 255) with R = 1-based destination screen

# DLC / mod colours the clone deliberately ignores (reported once)
_IGNORED = {
    (0, 170, 170): "water",
    (255, 106, 0): "sand",
    (182, 255, 0): "quark",
    (128, 255, 255): "low gravity",
    (255, 255, 128): "thin snow",
}

# 0-based screens of the base game inside the original atlas
IMP_ROOM = 44
BASE_GAME_SCREENS = tuple(range(C.BASE_SCREENS)) + (IMP_ROOM,)

# babe position on the last base-game screen (screen-local, see RESEARCH.md)
# calibrated so the base-game TAS ending triggers on its last input frame
BASE_GOAL_LOCAL = (347, 100, 410, 144)


def screen_origin_in_atlas(index, cols=C.ATLAS_COLS):
    """Top-left pixel of screen ``index`` (0-based) in the atlas."""
    return (index // cols) * C.SCREEN_TILES_W, (index % cols) * C.SCREEN_TILES_H


def world_y_of_screen(index):
    """World y of the top edge of screen ``index``."""
    return -C.SCREEN_H * index


def _classify(rgba):
    """Map an (H, W, 4) RGBA screen image to raw block codes (slopes unresolved).

    Returns (codes, wind, teleports) where slopes are marked with -1."""
    h, w = rgba.shape[:2]
    codes = np.zeros((h, w), dtype=np.int8)
    a = rgba[..., 3] > 0
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    key = (r << 16) | (g << 8) | b

    def k(rgb):
        return (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]

    codes[a & (key == k(RGB_SOLID))] = P.T_SOLID
    codes[a & (key == k(RGB_SLOPE))] = -1
    codes[a & (key == k(RGB_FAKE))] = P.T_FAKE
    codes[a & (key == k(RGB_ICE))] = P.T_ICE
    codes[a & (key == k(RGB_SNOW))] = P.T_SNOW
    codes[a & (key == k(RGB_NOWIND))] = P.T_NOWIND
    wind = bool(np.any(a & (key == k(RGB_WIND))))
    tele = a & (g == 0) & (b == 255) & (r >= 1)
    teleports = []
    for yy, xx in zip(*np.nonzero(tele)):
        teleports.append((int(xx), int(r[yy, xx])))
    ignored = set()
    for rgb, name in _IGNORED.items():
        if np.any(a & (key == k(rgb))):
            ignored.add(name)
    return codes, wind, teleports, ignored


def resolve_slopes(codes):
    """Turn slope markers (-1) into oriented slope tiles from their neighbours.

    A slope needs one solid/fake neighbour left or right and one above or
    below; otherwise it degrades to a solid block."""
    h, w = codes.shape
    solidish = (codes == P.T_SOLID) | (codes == -1) | (codes == P.T_FAKE) | \
               (codes == P.T_ICE) | (codes == P.T_SNOW)
    out = codes.copy()
    ys, xs = np.nonzero(codes == -1)
    for y, x in zip(ys, xs):
        right = x + 1 < w and solidish[y, x + 1]
        left = x > 0 and solidish[y, x - 1]
        below = y + 1 < h and solidish[y + 1, x]
        above = y > 0 and solidish[y - 1, x]
        if right and below:
            t = P.T_SLOPE_TL
        elif left and below:
            t = P.T_SLOPE_TR
        elif right and above:
            t = P.T_SLOPE_BL
        elif left and above:
            t = P.T_SLOPE_BR
        else:
            t = P.T_SOLID
        out[y, x] = t
    return out


class Level:
    """Immutable level description consumed by the physics kernel."""

    def __init__(self, screens, wind, tp_left, tp_right, goal=None, start=None,
                 name="level", area_names=None):
        n = len(screens)
        self.n_screens = n
        self.name = name
        self.screen_tiles = [np.asarray(sc, dtype=np.int8) for sc in screens]
        # global grid: row 0 is the top row of the highest screen
        self.tiles = np.zeros((n * C.SCREEN_TILES_H, C.SCREEN_TILES_W), dtype=np.int8)
        for i, sc in enumerate(self.screen_tiles):
            r0 = (n - 1 - i) * C.SCREEN_TILES_H
            self.tiles[r0:r0 + C.SCREEN_TILES_H] = sc
        self.wind = np.asarray(wind, dtype=np.int8)
        self.tp_left = np.asarray(tp_left, dtype=np.int32)
        self.tp_right = np.asarray(tp_right, dtype=np.int32)
        if goal is None:
            gx0, gy0, gx1, gy1 = BASE_GOAL_LOCAL
            top = world_y_of_screen(n - 1)
            goal = (gx0, top + gy0, gx1, top + gy1)
        self.goal = tuple(float(v) for v in goal)
        self.start = start if start is not None else (C.START_X, C.START_Y)
        self.area_names = area_names or [C.area_of_screen(i) for i in range(n)]
        self.offset_y = C.SCREEN_H * (n - 1)

    # ------------------------------------------------------------------ io
    @classmethod
    def from_atlas(cls, rgba, n_screens=C.BASE_SCREENS, name="atlas", goal=None,
                   start=None, include=None):
        """Build from an atlas image array (H, W, 4) in the native layout.

        ``n_screens`` screens (0-based 0..n-1) are loaded and the babe is put on
        the last of them. ``include`` optionally lists extra / restricted
        0-based screen indices: screens outside it are loaded empty and side
        teleports into them are dropped (used to exclude DLC content)."""
        rgba = np.asarray(rgba)
        if rgba.shape[2] == 3:
            alpha = np.full(rgba.shape[:2] + (1,), 255, dtype=rgba.dtype)
            rgba = np.concatenate([rgba, alpha], axis=2)
        height, width = rgba.shape[:2]
        if width % C.SCREEN_TILES_W or height % C.SCREEN_TILES_H:
            raise ValueError("atlas must contain whole 60x45 screens")
        cols = width // C.SCREEN_TILES_W

        def screen_image(i):
            x0, y0 = (i // cols) * C.SCREEN_TILES_W, (i % cols) * C.SCREEN_TILES_H
            return rgba[y0:y0 + C.SCREEN_TILES_H, x0:x0 + C.SCREEN_TILES_W]

        keep = set(range(n_screens)) if include is None else set(include)
        count = max(keep) + 1
        ignored_all = set()
        screens, wind, tl, tr = [], [], [], []
        for i in range(count):
            if i not in keep:
                screens.append(np.zeros((C.SCREEN_TILES_H, C.SCREEN_TILES_W), np.int8))
                wind.append(0)
                tl.append(-1)
                tr.append(-1)
                continue
            codes, w, tele, ignored = _classify(screen_image(i))
            ignored_all.update(ignored)
            screens.append(resolve_slopes(codes))
            wind.append(1 if w else 0)
            left = right = -1
            targets = [(x, t - 1) for x, t in tele if (t - 1) in keep]
            dropped = sorted({t for _, t in tele if (t - 1) not in keep})
            if dropped:
                warnings.warn("screen %d: side teleport to excluded screen(s) %s dropped"
                              % (i + 1, dropped))
            if targets:
                distinct = sorted(set(targets))
                if len({t for _, t in distinct}) == 1:
                    left = right = distinct[0][1]
                else:
                    left = min(distinct)[1]
                    right = max(distinct)[1]
            tl.append(left)
            tr.append(right)
        if ignored_all:
            warnings.warn("ignored non-base-game blocks: %s" % ", ".join(sorted(ignored_all)))
        if goal is None:
            gx0, gy0, gx1, gy1 = BASE_GOAL_LOCAL
            top = world_y_of_screen(n_screens - 1)
            goal = (gx0, top + gy0, gx1, top + gy1)
        names = [C.area_of_screen(i) if i < C.BASE_SCREENS else
                 ("Imp Room (secret)" if i == IMP_ROOM else "Excluded") for i in range(count)]
        return cls(screens, wind, tl, tr, goal=goal, start=start, name=name, area_names=names)

    @classmethod
    def from_png(cls, path, **kw):
        from PIL import Image

        img = np.array(Image.open(path).convert("RGBA"))
        kw.setdefault("name", os.path.splitext(os.path.basename(path))[0])
        return cls.from_atlas(img, **kw)

    @classmethod
    def from_xnb(cls, path, **kw):
        from .xnb import read_texture

        kw.setdefault("name", "original")
        return cls.from_atlas(read_texture(path), **kw)

    @classmethod
    def builtin(cls):
        """The clone's own 43-screen base-game-style map."""
        import json

        goal = start = None
        if os.path.exists(BUILTIN_META):
            with open(BUILTIN_META, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            goal = tuple(meta["goal"])
            start = tuple(meta["start"])
        return cls.from_png(BUILTIN_PNG, n_screens=C.BASE_SCREENS, name="builtin",
                            goal=goal, start=start)

    @classmethod
    def original(cls, game_dir=None):
        """Original base-game geometry from the local Steam install.

        Loads screens 1-43 and the hidden Imp Room (screen 45, reached through
        Bargainburg's side teleport). New Babe+ (46-100), Ghost of the Babe
        (102-155) and Philosopher's Forest (156-160) are excluded, following
        Content/gui/location_settings.xml."""
        from .xnb import find_level_xnb

        path = find_level_xnb(game_dir)
        if path is None:
            raise FileNotFoundError(
                "Jump King's Content/level.xnb was not found. Install the game "
                "through Steam (or pass game_dir=...).")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cls.from_xnb(path, n_screens=C.BASE_SCREENS, include=BASE_GAME_SCREENS)

    @classmethod
    def load(cls, source="auto"):
        """``builtin``, ``original``, ``auto`` (original if installed) or a path."""
        if source == "builtin":
            return cls.builtin()
        if source == "original":
            return cls.original()
        if source == "auto":
            from .xnb import find_level_xnb

            if find_level_xnb() is not None:
                try:
                    return cls.original()
                except Exception as exc:  # pragma: no cover - depends on install
                    warnings.warn("could not read the original map (%s); using builtin" % exc)
            return cls.builtin()
        if source.lower().endswith(".xnb"):
            return cls.from_xnb(source)
        return cls.from_png(source)

    def to_atlas(self):
        """Encode back into a 780x585 RGBA atlas (round-trips through from_atlas)."""
        cols = C.ATLAS_COLS
        atlas = np.zeros((cols * C.SCREEN_TILES_H, cols * C.SCREEN_TILES_W, 4), dtype=np.uint8)
        colour = {
            P.T_SOLID: RGB_SOLID, P.T_ICE: RGB_ICE, P.T_SNOW: RGB_SNOW,
            P.T_NOWIND: RGB_NOWIND, P.T_FAKE: RGB_FAKE,
            P.T_SLOPE_TL: RGB_SLOPE, P.T_SLOPE_TR: RGB_SLOPE,
            P.T_SLOPE_BL: RGB_SLOPE, P.T_SLOPE_BR: RGB_SLOPE,
        }
        for i, sc in enumerate(self.screen_tiles):
            x0, y0 = screen_origin_in_atlas(i)
            view = atlas[y0:y0 + C.SCREEN_TILES_H, x0:x0 + C.SCREEN_TILES_W]
            for code, rgb in colour.items():
                m = sc == code
                view[m, :3] = rgb
                view[m, 3] = 255
            free = list(zip(*np.nonzero(sc == P.T_EMPTY)))
            markers = []
            if self.wind[i]:
                markers.append(RGB_WIND)
            if self.tp_left[i] >= 0 or self.tp_right[i] >= 0:
                if self.tp_left[i] == self.tp_right[i]:
                    markers.append((self.tp_left[i] + 1, 0, 255))
                else:
                    markers.append(("L", self.tp_left[i] + 1))
                    markers.append(("R", self.tp_right[i] + 1))
            for m in markers:
                if m[0] == "L":
                    cand = [p for p in free if p[1] < 30]
                    rgb = (m[1], 0, 255)
                elif m[0] == "R":
                    cand = [p for p in free if p[1] >= 30][::-1]
                    rgb = (m[1], 0, 255)
                else:
                    cand = free
                    rgb = m
                if not cand:
                    raise ValueError("no free pixel for a marker on screen %d" % (i + 1))
                yy, xx = cand[0]
                free.remove((yy, xx))
                view[yy, xx, :3] = rgb
                view[yy, xx, 3] = 255
        return atlas

    def save_png(self, path):
        from PIL import Image

        Image.fromarray(self.to_atlas(), "RGBA").save(path)

    # ------------------------------------------------------------- helpers
    def params(self, splat_time=C.SPLAT_TIME, wind_offset=0.0, wind_scale=1.0):
        p = np.zeros(P.PARAM_SIZE, dtype=np.float64)
        p[P.P_SPLAT_TIME] = splat_time
        p[P.P_GOAL_X0], p[P.P_GOAL_Y0], p[P.P_GOAL_X1], p[P.P_GOAL_Y1] = self.goal
        p[P.P_WIND_OFFSET] = wind_offset
        p[P.P_N_SCREENS] = self.n_screens
        p[P.P_OFFSET_Y] = self.offset_y
        p[P.P_WIND_SCALE] = wind_scale
        return p

    def screen_grid(self, index):
        """(45, 60) tile codes of a screen (zeros outside the map)."""
        if 0 <= index < self.n_screens:
            return self.screen_tiles[index]
        return np.zeros((C.SCREEN_TILES_H, C.SCREEN_TILES_W), dtype=np.int8)

    def tile_at(self, wx, wy):
        r = (int(np.floor(wy)) + self.offset_y) // C.TILE
        c = int(np.floor(wx)) // C.TILE
        if r < 0 or c < 0 or c >= C.SCREEN_TILES_W:
            return P.T_EMPTY
        if r >= self.tiles.shape[0]:
            return P.T_SOLID
        return int(self.tiles[r, c])

    def collides(self, hx, hy):
        return bool(P.collide(self.tiles, self.offset_y, int(hx), int(hy)) & P.CI_COLL)

    def __repr__(self):
        return "Level(%r, screens=%d, wind=%d, teleports=%d)" % (
            self.name, self.n_screens, int(self.wind.sum()),
            int(((self.tp_left >= 0) | (self.tp_right >= 0)).sum()))
