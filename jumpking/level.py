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

# --- native block colours (BaseBlockFactory / LevelManager) ------------------
RGB_SOLID = (0, 0, 0)
RGB_SLOPE = (255, 0, 0)
RGB_FAKE = (128, 128, 128)
RGB_ICE = (0, 255, 255)
RGB_SNOW = (255, 255, 0)
RGB_WIND = (0, 255, 0)
RGB_NOWIND = (255, 255, 255)
RGB_WATER = (0, 170, 170)
RGB_SAND = (255, 106, 0)
RGB_QUARK = (182, 255, 0)
# teleport: (R, 0, 255) with R = 1-based destination screen
# wind variants: (R, 255, 0) with R in 192..207 (oscillating), 208..223
# (constant, leftwards) or 224..239 (constant, rightwards); see wind_code()

BLOCK_COLOURS = {
    RGB_SOLID: P.T_SOLID, RGB_FAKE: P.T_FAKE, RGB_ICE: P.T_ICE, RGB_SNOW: P.T_SNOW,
    RGB_NOWIND: P.T_NOWIND, RGB_WATER: P.T_WATER, RGB_SAND: P.T_SAND, RGB_QUARK: P.T_QUARK,
}
# blocks that orient a neighbouring slope (BaseBlockFactory.IsSolidBlock + fake)
_EDGE_CODES = (P.T_SOLID, P.T_ICE, P.T_SNOW, P.T_SAND, P.T_FAKE, -1)

# 0-based screens inside the original atlas (Content/gui/location_settings.xml)
IMP_ROOM = 44
BASE_GAME_SCREENS = tuple(range(C.BASE_SCREENS)) + (IMP_ROOM,)
NBP_SCREENS = tuple(range(45, 100))            # 46-100: entrance + New Babe+
GHOST_SCREENS = tuple(range(101, 154))         # 102-154: Ghost of the Babe
FOREST_SCREENS = tuple(range(155, 164))        # 156-164: Philosopher's Forest + Hole
FULL_SCREENS = BASE_GAME_SCREENS + NBP_SCREENS + GHOST_SCREENS + FOREST_SCREENS

# babe position on the last base-game screen (screen-local, see RESEARCH.md)
# calibrated so the base-game TAS ending triggers on its last input frame
BASE_GOAL_LOCAL = (347, 100, 410, 144)


def screen_origin_in_atlas(index, cols=C.ATLAS_COLS):
    """Top-left pixel of screen ``index`` (0-based) in the atlas."""
    return (index // cols) * C.SCREEN_TILES_W, (index % cols) * C.SCREEN_TILES_H


def world_y_of_screen(index):
    """World y of the top edge of screen ``index``."""
    return -C.SCREEN_H * index


def wind_code(intensity=0, direction=None):
    """Per-screen wind code used by ``Level.wind`` and the kernel.

    0: no wind, 1: the default oscillating wind; otherwise the coloured
    variant with ``intensity`` 1..16 and ``direction`` None (oscillating),
    True (constant, leftwards) or False (constant, rightwards)."""
    if intensity <= 0:
        return 1
    kind = 0 if direction is None else (1 if direction else 2)
    return 2 + (int(intensity) - 1) + 16 * kind


def _classify(rgba):
    """Map an (H, W, 4) RGBA screen image to raw block codes (slopes unresolved).

    Returns (codes, wind, teleports, unknown) where slopes are marked with -1,
    ``wind`` is the screen's wind code and ``teleports`` is the (left, right)
    pair of 1-based targets (0 = none) as LevelManager.LoadBlocksInterval reads
    them: a pixel in columns 0-29 sets the left link, 30-59 the right one, and
    later pixels (row-major) overwrite earlier ones."""
    h, w = rgba.shape[:2]
    codes = np.zeros((h, w), dtype=np.int8)
    a = rgba[..., 3] == 255
    r = rgba[..., 0].astype(np.int32)
    g = rgba[..., 1].astype(np.int32)
    b = rgba[..., 2].astype(np.int32)
    key = (r << 16) | (g << 8) | b

    def k(rgb):
        return (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]

    known = np.zeros((h, w), bool)
    for rgb, code in BLOCK_COLOURS.items():
        m = a & (key == k(rgb))
        codes[m] = code
        known |= m
    m = a & (key == k(RGB_SLOPE))
    codes[m] = -1
    known |= m
    wind = 0
    intensity = 0
    direction = None
    left = right = 0
    for yy, xx in zip(*np.nonzero(a & ~known)):
        rr, gg, bb = int(r[yy, xx]), int(g[yy, xx]), int(b[yy, xx])
        if gg == 0 and bb == 255:
            if xx >= 30:
                right = rr
            else:
                left = rr
        if gg == 255 and bb == 0:
            if 191 < rr < 208:
                intensity, direction = rr - 191, None
            elif 207 < rr < 224:
                intensity, direction = rr - 207, True
            elif 223 < rr < 240:
                intensity, direction = rr - 223, False
            wind = 1
    if wind:
        wind = wind_code(intensity, direction)
    unknown = set()
    for yy, xx in zip(*np.nonzero(a & ~known)):
        rgb = (int(r[yy, xx]), int(g[yy, xx]), int(b[yy, xx]))
        if not ((rgb[1] == 0 and rgb[2] == 255) or (rgb[1] == 255 and rgb[2] == 0)):
            unknown.add(rgb)
    return codes, wind, (left, right), unknown


def resolve_slopes(codes):
    """Turn slope markers (-1) into oriented slope tiles from their neighbours
    (BaseBlockFactory.GetSlopeType).

    ``codes`` is a grid of rows (top first); for a whole map pass the stacked
    global grid so that neighbours on adjacent screens count, as in the game.
    Columns outside the grid count as solid (LevelManager.IsBlockEdge). A slope
    without a valid pair of neighbours becomes ``T_SLOPE_NONE``, which the
    game creates but never collides with."""
    codes = np.asarray(codes)
    h, w = codes.shape
    solidish = np.isin(codes, _EDGE_CODES)
    pad = np.zeros((h + 2, w + 2), bool)
    pad[1:-1, 1:-1] = solidish
    pad[:, 0] = True
    pad[:, -1] = True
    out = codes.copy()
    ys, xs = np.nonzero(codes == -1)
    for y, x in zip(ys, xs):
        py, px = y + 1, x + 1
        right = pad[py, px + 1]
        left = pad[py, px - 1]
        below = pad[py + 1, px]
        above = pad[py - 1, px]
        if right and below:
            t = P.T_SLOPE_TL
        elif left and below:
            t = P.T_SLOPE_TR
        elif right and above:
            t = P.T_SLOPE_BL
        elif left and above:
            t = P.T_SLOPE_BR
        else:
            t = P.T_SLOPE_NONE
        out[y, x] = t
    return out


class Ending:
    """A babe the king can reach (EndingManager): standing on the ground on the
    camera screen ``screen`` (0-based) with the hitbox centre beyond ``x``
    (``side`` +1: centre.x >= x, -1: centre.x <= x) and centre.y <= ``max_y``."""

    __slots__ = ("name", "screen", "side", "x", "max_y")

    def __init__(self, name, screen, side, x, max_y=1e18):
        self.name = name
        self.screen = int(screen)
        self.side = int(side)
        self.x = float(x)
        self.max_y = float(max_y)

    def as_tuple(self):
        return (self.screen, self.side, self.x, self.max_y)

    def __repr__(self):
        return "Ending(%r, screen=%d)" % (self.name, self.screen + 1)


def _ending_pos(screen, inside_x, inside_y=-36):
    """endingKingStartPosition = GAME_RECT.Center + INSIDE_SCREEN_POSITION
    + (0, -screen * 360)."""
    return 240 + inside_x, 180 + inside_y - screen * C.SCREEN_H


# NormalEnding (screen 43), NewBabePlusEnding (100), OwlEnding (154)
_NX, _ = _ending_pos(42, 96)
_BX, _BY = _ending_pos(99, -96)
_OX, _OY = _ending_pos(153, 0)
ENDING_NORMAL = Ending("normal", 42, +1, _NX)
ENDING_NBP = Ending("new_babe_plus", 99, -1, _BX, _BY + 1)
ENDING_GHOST = Ending("ghost_of_the_babe", 153, +1, _OX, _OY + 1)
ENDINGS = (ENDING_NORMAL, ENDING_NBP, ENDING_GHOST)


class Level:
    """Immutable level description consumed by the physics kernel."""

    def __init__(self, screens, wind, tp_left, tp_right, goal=None, start=None,
                 name="level", area_names=None, endings=(), content="custom"):
        n = len(screens)
        self.n_screens = n
        self.name = name
        self.content = content
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
            top = world_y_of_screen(min(n, C.BASE_SCREENS) - 1)
            goal = (gx0, top + gy0, gx1, top + gy1)
        self.goal = tuple(float(v) for v in goal)
        self.endings = tuple(e for e in endings if e.screen < n)
        self.start = start if start is not None else (C.START_X, C.START_Y)
        self.area_names = area_names or [C.area_of_screen(i) for i in range(n)]
        self.offset_y = C.SCREEN_H * (n - 1)

    # ------------------------------------------------------------------ io
    @classmethod
    def from_atlas(cls, rgba, n_screens=C.BASE_SCREENS, name="atlas", goal=None,
                   start=None, include=None, endings=(), content="custom"):
        """Build from an atlas image array (H, W, 4) in the native layout.

        ``n_screens`` screens (0-based 0..n-1) are loaded. ``include``
        optionally lists the 0-based screen indices to keep: screens outside it
        are loaded empty and side teleports into them are dropped. Slopes are
        oriented on the stacked map, so neighbours on adjacent screens count
        as in the game."""
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
        count = max(max(keep) + 1, n_screens)
        unknown_all = set()
        raw, wind, links = [], [], []
        for i in range(count):
            if i not in keep:
                raw.append(np.zeros((C.SCREEN_TILES_H, C.SCREEN_TILES_W), np.int8))
                wind.append(0)
                links.append((0, 0))
                continue
            codes, w, tele, unknown = _classify(screen_image(i))
            unknown_all.update(unknown)
            raw.append(codes)
            wind.append(w)
            links.append(tele)
        if unknown_all:
            warnings.warn("ignored unknown block colours: %s" % sorted(unknown_all))
        # orient slopes on the stacked map (row 0 = top of the highest screen)
        H = C.SCREEN_TILES_H
        stacked = np.concatenate(raw[::-1], axis=0)
        resolved = resolve_slopes(stacked)
        screens = [resolved[(count - 1 - i) * H:(count - i) * H] for i in range(count)]
        tl, tr = [], []
        for i, (left, right) in enumerate(links):
            out = []
            for t in (left, right):
                if t and (t - 1) not in keep:
                    warnings.warn("screen %d: side teleport to excluded screen %d dropped"
                                  % (i + 1, t))
                    t = 0
                out.append(t - 1 if t else -1)
            tl.append(out[0])
            tr.append(out[1])
        if goal is None:
            gx0, gy0, gx1, gy1 = BASE_GOAL_LOCAL
            top = world_y_of_screen(min(n_screens, C.BASE_SCREENS) - 1)
            goal = (gx0, top + gy0, gx1, top + gy1)
        names = [C.area_of_screen(i) if i in keep else "Excluded" for i in range(count)]
        return cls(screens, wind, tl, tr, goal=goal, start=start, name=name, area_names=names,
                   endings=endings, content=content)

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
                            goal=goal, start=start, content="builtin")

    @classmethod
    def original(cls, game_dir=None, content="full"):
        """Original geometry from the local Steam install (``Content/level.xnb``).

        ``content="full"`` (default) loads the whole world exactly as the game
        does: all 169 atlas screens, i.e. the base game (1-43), the Imp Room
        (45), New Babe+ (46-100), Ghost of the Babe (102-154) and Philosopher's
        Forest with the Hole (156-164), with every side teleport and all three
        endings. ``content="base"`` keeps only screens 1-43, the Imp Room and
        the normal ending."""
        from .xnb import find_level_xnb

        path = find_level_xnb(game_dir)
        if path is None:
            raise FileNotFoundError(
                "Jump King's Content/level.xnb was not found. Install the game "
                "through Steam (or pass game_dir=...).")
        if content == "base":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return cls.from_xnb(path, n_screens=C.BASE_SCREENS, include=BASE_GAME_SCREENS,
                                    endings=(ENDING_NORMAL,), content="base")
        if content != "full":
            raise ValueError("content must be 'full' or 'base'")
        return cls.from_xnb(path, n_screens=C.ATLAS_SCREENS, include=range(C.ATLAS_SCREENS),
                            endings=ENDINGS, content="full")

    @classmethod
    def load(cls, source="auto"):
        """``builtin``, ``original`` (full world), ``original-base``, ``auto``
        (original if installed) or a path."""
        if source == "builtin":
            return cls.builtin()
        if source == "original":
            return cls.original()
        if source in ("original-base", "base"):
            return cls.original(content="base")
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

    def ending(self, name):
        """The ``Ending`` called ``name`` (normal, new_babe_plus, ghost_of_the_babe)."""
        for e in self.endings:
            if e.name == name:
                return e
        raise KeyError("level %r has no ending %r" % (self.name, name))

    def to_atlas(self):
        """Encode back into a 780x585 RGBA atlas (round-trips through from_atlas)."""
        cols = C.ATLAS_COLS
        atlas = np.zeros((cols * C.SCREEN_TILES_H, cols * C.SCREEN_TILES_W, 4), dtype=np.uint8)
        colour = {code: rgb for rgb, code in BLOCK_COLOURS.items()}
        for code in (P.T_SLOPE_TL, P.T_SLOPE_TR, P.T_SLOPE_BL, P.T_SLOPE_BR, P.T_SLOPE_NONE):
            colour[code] = RGB_SLOPE
        for i, sc in enumerate(self.screen_tiles):
            x0, y0 = screen_origin_in_atlas(i)
            view = atlas[y0:y0 + C.SCREEN_TILES_H, x0:x0 + C.SCREEN_TILES_W]
            for code, rgb in colour.items():
                m = sc == code
                view[m, :3] = rgb
                view[m, 3] = 255
            free = list(zip(*np.nonzero(sc == P.T_EMPTY)))
            markers = []
            w = int(self.wind[i])
            if w == 1:
                markers.append(("A", RGB_WIND))
            elif w >= 2:
                k = w - 2
                markers.append(("A", (192 + 16 * (k // 16) + k % 16, 255, 0)))
            tl, tr = int(self.tp_left[i]), int(self.tp_right[i])
            if tl >= 0 and tl == tr:
                markers.append(("L", (tl + 1, 0, 255)))
            else:
                if tl >= 0:
                    markers.append(("L", (tl + 1, 0, 255)))
                if tr >= 0:
                    markers.append(("R", (tr + 1, 0, 255)))
            for side, rgb in markers:
                if side == "L":
                    cand = [p for p in free if p[1] < 30]
                elif side == "R":
                    cand = [p for p in free if p[1] >= 30][::-1]
                else:
                    cand = free
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
    def params(self, splat_time=C.SPLAT_TIME, wind_offset=0.0, wind_scale=1.0,
               legacy_teleport=False):
        p = np.zeros(P.PARAM_SIZE, dtype=np.float64)
        p[P.P_LEGACY_TELEPORT] = 1.0 if legacy_teleport else 0.0
        p[P.P_SPLAT_TIME] = splat_time
        p[P.P_GOAL_X0], p[P.P_GOAL_Y0], p[P.P_GOAL_X1], p[P.P_GOAL_Y1] = self.goal
        p[P.P_WIND_OFFSET] = wind_offset
        p[P.P_N_SCREENS] = self.n_screens
        p[P.P_OFFSET_Y] = self.offset_y
        p[P.P_WIND_SCALE] = wind_scale
        p[P.P_N_ENDINGS] = len(self.endings)
        for k, e in enumerate(self.endings[:P.MAX_ENDINGS]):
            b = P.P_ENDINGS + P.ENDING_STRIDE * k
            p[b:b + P.ENDING_STRIDE] = e.as_tuple()
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

    def teleport_target(self, screen, exit_left):
        """0-based screen a side exit of ``screen`` leads to (-1: none)."""
        tl, tr = int(self.tp_left[screen]), int(self.tp_right[screen])
        if tl >= 0 and tr >= 0:
            return tl if exit_left else tr
        return tl if tl >= 0 else tr

    def screens_with(self, code):
        """1-based screens containing tile ``code``."""
        return [i + 1 for i in range(self.n_screens) if (self.screen_tiles[i] == code).any()]

    def __repr__(self):
        return "Level(%r, screens=%d, wind=%d, teleports=%d)" % (
            self.name, self.n_screens, int((self.wind > 0).sum()),
            int(((self.tp_left >= 0) | (self.tp_right >= 0)).sum()))
