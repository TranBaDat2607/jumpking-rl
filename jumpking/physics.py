"""Deterministic Jump King physics kernel.

The whole player/world state lives in one float64 vector (see the ``S_*``
indices) so that it can be copied, hashed, batched and restored in O(1).
Every arithmetic result that is float32 in the original game is rounded to
float32 with ``f32`` to reproduce its numerics.

The kernel is a port of the game's ``BodyComp`` behaviour pipeline and the
player behaviour tree (``PlayerEntity.MakeBT``), including the DLC blocks
(water, sand, quark), the directional wind variants, the Giant Boots and the
Snake Ring. Collision follows ``LevelManager.CheckCollisionInternal``: only the
camera screen and its two neighbours are tested, and the per-screen results
are merged in the game's order (which matters for slopes on screen seams).

The functions are written in a numba-compatible subset of Python and are JIT
compiled when numba is installed (set ``JUMPKING_NO_NUMBA=1`` to disable).
See docs/RESEARCH.md for the reference behaviour.
"""
import math
import os

import numpy as np

from . import constants as C

# ---------------------------------------------------------------------------
# JIT helper
# ---------------------------------------------------------------------------
NUMBA_ENABLED = False
if os.environ.get("JUMPKING_NO_NUMBA", "0") != "1":
    try:
        import numba  # noqa: F401

        NUMBA_ENABLED = True
    except ImportError:  # pragma: no cover - optional dependency
        NUMBA_ENABLED = False


def _invalidate_stale_cache():
    """numba only tracks this file's timestamp; constants live elsewhere and are
    frozen into the compiled code, so drop the cache when they change."""
    import glob
    import hashlib

    here = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(here, "__pycache__")
    digest = hashlib.sha1()
    for path in (os.path.join(here, "constants.py"), os.path.join(here, "physics.py"),
                 C.OVERRIDE_PATH):
        if os.path.isfile(path):
            with open(path, "rb") as fh:
                digest.update(fh.read())
    stamp = os.path.join(cache_dir, "physics.constants.sha1")
    try:
        with open(stamp, "r") as fh:
            if fh.read().strip() == digest.hexdigest():
                return
    except OSError:
        pass
    # every cached kernel of the package may have inlined the old values
    for f in glob.glob(os.path.join(cache_dir, "*.nb[ic]")):
        try:
            os.remove(f)
        except OSError:
            pass
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(stamp, "w") as fh:
            fh.write(digest.hexdigest())
    except OSError:
        pass


if NUMBA_ENABLED:
    _invalidate_stale_cache()


def _jit(fn):
    if NUMBA_ENABLED:
        return numba.njit(cache=True, nogil=True)(fn)
    return fn


# ---------------------------------------------------------------------------
# Tile codes
# ---------------------------------------------------------------------------
T_EMPTY = 0
T_SOLID = 1
T_SLOPE_TL = 2   # empty corner top-left  -> surface faces up-left
T_SLOPE_TR = 3   # empty corner top-right -> surface faces up-right
T_SLOPE_BL = 4   # empty corner bottom-left
T_SLOPE_BR = 5   # empty corner bottom-right
T_ICE = 6
T_SNOW = 7
T_NOWIND = 8     # non-solid, cancels wind
T_FAKE = 9       # non-solid, only used to orient slopes
T_WATER = 10     # non-solid: halves movement, gravity and jump charging
T_SAND = 11      # non-solid quicksand
T_QUARK = 12     # non-solid: snaps a max-speed fall to the 8 px grid
T_SLOPE_NONE = 13  # slope pixel without a valid orientation (never collides)
N_TILE_TYPES = 14

# collision info bit flags
CI_COLL = 1        # touches a blocking block
CI_BOX = 2         # touches a blocking non-slope block
CI_ICE = 4
CI_SNOW = 8
CI_WATER = 16
CI_SAND = 32
CI_QUARK = 64
CI_NOWIND = 128
CI_FLAGS = 255
CI_SLOPE_SHIFT = 8     # effective slope type (tile code) in bits 8..11
CI_NORMAL_SHIFT = 12   # slope whose normal is reported, bits 12..15

# ---------------------------------------------------------------------------
# State vector layout
# ---------------------------------------------------------------------------
S_X = 0
S_Y = 1
S_VX = 2
S_VY = 3
S_LVX = 4            # velocity at the start of the last body update
S_LVY = 5
S_ON_GROUND = 6
S_KNOCKED = 7
S_ON_ICE = 8         # raw ice flag of the last Y resolution
S_ON_SNOW = 9        # raw snow flag of the last Y resolution
S_WIND_LATCH = 10
S_LAST_SCREEN = 11
S_SCREEN = 12        # camera screen (0-based)
S_JUMP_HELD = 13
S_CAN_JUMP = 14
S_CHARGING = 15      # JumpState is running
S_CHARGE = 16        # JumpState timer, seconds (float32)
S_AIR_CHARGE = 17    # jump leniency pause timer while airborne
S_H0 = 18            # JumpState direction buffer (most recent first)
S_H1 = 19
S_H2 = 20
S_H3 = 21
S_WAS_GROUNDED = 22  # the grounded branch of the player tree ran last tick
S_SPLAT = 23         # FailState is running
S_SPLAT_TIME = 24
S_SPLAT_DONE = 25
S_TICK = 26          # game ticks since start (drives the wind clock)
S_FACING = 27        # -1 / +1
S_AIR_TICKS = 28     # consecutive ticks without ground contact
S_JUMPS = 29
S_FALLS = 30         # screen-down transitions
S_SPLATS = 31
S_GOAL = 32
S_LAST_EVENTS = 33
S_MAX_HEIGHT = 34    # best (lowest) world y of the hitbox bottom seen so far
S_LAST_POWER = 35    # power of the last jump
S_WIND_OFFSET = 36   # seconds added to the wind clock (part of the state)
S_IN_WATER = 37      # WaterBlockBehaviour.IsPlayerOnBlock
S_IN_SAND = 38       # SandBlockBehaviour.IsPlayerOnBlock
S_BOOTS = 39         # wearing the Giant Boots
S_RING = 40          # Snake Ring enabled
S_BODY_OFF = 41      # BodyComp disabled (Giant Boots landing thud)
S_THUD = 42          # Giant Boots landing sequence: running child index
S_THUD_T = 43        # its pause timer
S_AIR_STAGE = 44     # airborne sequence index (0: jump leniency, 1: past it)
S_ENDING = 45        # ending reached: 1 normal, 2 New Babe+, 3 Ghost of the Babe
STATE_SIZE = 46

# events (bit flags returned by tick)
EV_JUMP = 1
EV_LAND = 2
EV_SPLAT = 4
EV_BUMP_WALL = 8
EV_BUMP_CEIL = 16
EV_SCREEN_UP = 32
EV_SCREEN_DOWN = 64
EV_GOAL = 128
EV_TELEPORT = 256
EV_SLOPE = 512
EV_GETUP = 1024
EV_WATER = 2048      # entered or left water
EV_SAND = 4096       # inside sand this tick
EV_TOUCH = 8192      # the hitbox overlaps the planning rectangle P_TOUCH

# level parameter vector layout
P_SPLAT_TIME = 0
P_GOAL_X0 = 1        # legacy rectangle goal (used when P_N_ENDINGS == 0)
P_GOAL_Y0 = 2
P_GOAL_X1 = 3
P_GOAL_Y1 = 4
P_WIND_OFFSET = 5    # default wind clock offset copied into new states
P_N_SCREENS = 6
P_OFFSET_Y = 7       # world y -> grid y offset (= 360 * (n_screens - 1))
P_WIND_SCALE = 8     # 1.0 = original
P_N_ENDINGS = 9      # number of babe endings (EndingManager)
P_ENDINGS = 10       # per ending: screen (0-based), side (+1/-1), x, max centre y
ENDING_STRIDE = 4
MAX_ENDINGS = 3
# 1 = side teleports keep the fractional y (game builds before the 2023
# BodyComp behaviour rewrite, e.g. the 2019 community TAS routes); 0 = the
# current game, which truncates y to an integer (HandlePlayerTeleportBehaviour)
P_LEGACY_TELEPORT = P_ENDINGS + ENDING_STRIDE * MAX_ENDINGS
# a rectangle (x, y, w, h) that raises EV_TOUCH while the hitbox overlaps it
# (w = 0: off). A planning aid with no effect on the simulation: searches use
# it to find jumps that touch an item in mid-air
P_TOUCH = P_LEGACY_TELEPORT + 1
PARAM_SIZE = P_TOUCH + 4

# module-level float32 constants (numba freezes globals as constants)
GRAVITY = C.GRAVITY
JUMP = C.JUMP
SPEED = C.SPEED
MAX_FALL = C.MAX_FALL
JUMP_TIME = C.JUMP_TIME
WALK_SPEED = C.WALK_SPEED
BOUNCE = C.BOUNCE
ICE_FRICTION = C.ICE_FRICTION
AIR_CHARGE_LIMIT = C.AIR_CHARGE_LIMIT
DT = C.DT
SNOW_MIN_POWER = C.SNOW_MIN_POWER
SNOW_BOOST_POWER = C.SNOW_BOOST_POWER
WIND_PHASE_RATE = C.WIND_PHASE_RATE
WIND_STRENGTH = C.WIND_STRENGTH
WIND_INTENSITY_STEP = C.WIND_INTENSITY_STEP
TARGET_ELAPSED_SECONDS = C.TARGET_ELAPSED_SECONDS
SAND_FALL_SPEED = C.SAND_FALL_SPEED
SAND_MOVE_MULTIPLIER = C.SAND_MOVE_MULTIPLIER
SAND_JUMP_MULTIPLIER = C.SAND_JUMP_MULTIPLIER
WATER_MULTIPLIER = C.WATER_MULTIPLIER
BOOTS_FREEZE_TIME = C.BOOTS_FREEZE_TIME
BOOTS_SHAKE_TIME = C.BOOTS_SHAKE_TIME
ICE_WALK_ACCEL = float(np.float32(C.ICE_FRICTION * np.float32(2.0)))
PW = C.PLAYER_W
PH = C.PLAYER_H
CAP_LEFT = float(np.float32(-PW / 2.0))   # (0f - width) / 2f
CAP_RIGHT = float(480 - PW // 2)          # 480 - width / 2 (int division)


def _f32py(v):
    return float(np.float32(v))


def _normalize_py(x, y):
    """MonoGame Vector2.Normalize in float32."""
    x = np.float32(x)
    y = np.float32(y)
    s = np.float32(x * x) + np.float32(y * y)
    inv = np.float32(1.0) / np.float32(math.sqrt(float(np.float32(s))))
    return float(np.float32(x * inv)), float(np.float32(y * inv))


# SlopeBlock.GetNormal (normalised once) and, for MakeComposite, the tangent
# of the normal normalised a second time: Perpendicular(n) = (n.y, -n.x).
_N_TL = _normalize_py(-1.0, -1.0)
_N_TR = _normalize_py(1.0, -1.0)
_N_BL = _normalize_py(-1.0, 1.0)
_N_BR = _normalize_py(1.0, 1.0)
NTL_X, NTL_Y = _N_TL
NTR_X, NTR_Y = _N_TR
NBL_X, NBL_Y = _N_BL
NBR_X, NBR_Y = _N_BR
_T_TL = _normalize_py(*_N_TL)
_T_TR = _normalize_py(*_N_TR)
_T_BL = _normalize_py(*_N_BL)
_T_BR = _normalize_py(*_N_BR)
TTL_X, TTL_Y = _T_TL[1], -_T_TL[0]
TTR_X, TTR_Y = _T_TR[1], -_T_TR[0]
TBL_X, TBL_Y = _T_BL[1], -_T_BL[0]
TBR_X, TBR_Y = _T_BR[1], -_T_BR[0]
UNIT = abs(NTL_X)


@_jit
def f32(v):
    return float(np.float32(v))


@_jit
def _sign(v):
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


@_jit
def _trunc(v):
    """C# (int) cast of a float: truncation toward zero."""
    return int(v)


@_jit
def _move_towards(current, target, step):
    """ErikMath.MoveTowards in float32."""
    diff = f32(target - current)
    if diff < -step:
        diff = -step
    if diff > step:
        diff = step
    return f32(current + diff)


@_jit
def _approach(value, target, step):
    return _move_towards(value, target, step)


@_jit
def _slope_normal(code):
    if code == T_SLOPE_TL:
        return NTL_X, NTL_Y
    if code == T_SLOPE_TR:
        return NTR_X, NTR_Y
    if code == T_SLOPE_BL:
        return NBL_X, NBL_Y
    if code == T_SLOPE_BR:
        return NBR_X, NBR_Y
    return 0.0, 0.0


@_jit
def _slope_tangent(code):
    if code == T_SLOPE_TL:
        return TTL_X, TTL_Y
    if code == T_SLOPE_TR:
        return TTR_X, TTR_Y
    if code == T_SLOPE_BL:
        return TBL_X, TBL_Y
    if code == T_SLOPE_BR:
        return TBR_X, TBR_Y
    return 0.0, 0.0


@_jit
def _dot_normal(vx, vy, code):
    nx, ny = _slope_normal(code)
    return f32(f32(vx * nx) + f32(vy * ny))


@_jit
def _composite(vx, vy, code):
    """VectorMath.MakeComposite(v, normal).i: the tangential component."""
    px, py = _slope_tangent(code)
    d = f32(f32(vx * px) + f32(vy * py))
    return f32(d * px), f32(d * py)


# ---------------------------------------------------------------------------
# Slope triangles (SlopeBlock.Intersects, an exact port of the line tests)
# ---------------------------------------------------------------------------
@_jit
def _in_span(start, end, p):
    """ErikMath.IsInSpan: strictly between the two ends."""
    if start > end:
        t = start
        start = end
        end = t
    return p > start and p < end


@_jit
def _tdiv(a, b):
    """C# integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        return -q
    return q


@_jit
def _line_k_m(x0, y0, x1, y1):
    if x0 > x1:
        t = x0
        x0 = x1
        x1 = t
        t = y0
        y0 = y1
        y1 = t
    if x0 == x1:
        return True, 0.0, 0.0
    k = float(_tdiv(y1 - y0, x1 - x0))
    m = f32(float(y0) - f32(k * x0))
    return False, k, m


@_jit
def _lines_cross(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
    """LineUtil.Intersects(left=a, right=b)."""
    av, ak, am = _line_k_m(ax0, ay0, ax1, ay1)
    bv, bk, bm = _line_k_m(bx0, by0, bx1, by1)
    if av and bv:
        return False
    if av or bv:
        if av:
            nx0, nx1, nk, nm = bx0, bx1, bk, bm
            vx, vy0, vy1 = ax0, ay0, ay1
        else:
            nx0, nx1, nk, nm = ax0, ax1, ak, am
            vx, vy0, vy1 = bx0, by0, by1
        if not _in_span(nx0, nx1, vx):
            return False
        yy = f32(f32(nk * vx) + nm)
        return _in_span(vy0, vy1, yy)
    if ak == bk:
        return False   # parallel: the game divides by zero (no crossing)
    xx = f32(f32(bm - am) / f32(ak - bk))
    return _in_span(ax0, ax1, xx) and _in_span(bx0, bx1, xx)


@_jit
def _rect_contains(hx, hy, px, py):
    return hx <= px and px < hx + PW and hy <= py and py < hy + PH


@_jit
def _tri_edge_hits(x0, y0, x1, y1, hx, hy):
    if _lines_cross(x0, y0, x1, y1, hx, hy, hx + PW, hy):
        return True
    if _lines_cross(x0, y0, x1, y1, hx + PW, hy, hx + PW, hy + PH):
        return True
    if _lines_cross(x0, y0, x1, y1, hx + PW, hy + PH, hx, hy + PH):
        return True
    if _lines_cross(x0, y0, x1, y1, hx, hy + PH, hx, hy):
        return True
    return False


@_jit
def _slope_hits(code, cx, cy, hx, hy):
    """Does the player hitbox at integer (hx, hy) collide with the slope cell
    whose top-left corner is (cx, cy)? Port of SlopeBlock.Intersects (the
    BottomLeft and BottomRight triangles are identical in the game)."""
    if not (cx < hx + PW and hx < cx + 8 and cy < hy + PH and hy < cy + 8):
        return False
    if code == T_SLOPE_TL:
        px, py, qx, qy, rx, ry = cx + 8, cy, cx + 8, cy + 8, cx, cy + 8
    elif code == T_SLOPE_TR:
        px, py, qx, qy, rx, ry = cx, cy, cx + 8, cy + 8, cx, cy + 8
    else:
        px, py, qx, qy, rx, ry = cx, cy, cx + 8, cy, cx, cy + 8
    if _tri_edge_hits(px, py, qx, qy, hx, hy):
        return True
    if _tri_edge_hits(qx, qy, rx, ry, hx, hy):
        return True
    if _tri_edge_hits(rx, ry, px, py, hx, hy):
        return True
    # vertices, then edge midpoints (Line.middle, integer halves)
    if _rect_contains(hx, hy, px, py) or _rect_contains(hx, hy, qx, qy):
        return True
    if _rect_contains(hx, hy, _tdiv(px + qx, 2), _tdiv(py + qy, 2)):
        return True
    if _rect_contains(hx, hy, qx, qy) or _rect_contains(hx, hy, rx, ry):
        return True
    if _rect_contains(hx, hy, _tdiv(qx + rx, 2), _tdiv(qy + ry, 2)):
        return True
    if _rect_contains(hx, hy, rx, ry) or _rect_contains(hx, hy, px, py):
        return True
    if _rect_contains(hx, hy, _tdiv(rx + px, 2), _tdiv(ry + py, 2)):
        return True
    return False


# ---------------------------------------------------------------------------
# Collision queries
# ---------------------------------------------------------------------------
@_jit
def _try_rows(tiles, offy, ra, rb, hx, hy, floor_below):
    """LevelScreen.TryCollision restricted to grid rows [ra, rb].

    Returns the info bits: flags of every touched block, the effective slope
    type (bits 8..11) and the slope whose normal is reported (bits 12..15)."""
    nrows = tiles.shape[0]
    ncols = tiles.shape[1]
    gy = hy + offy
    r0 = gy // 8
    r1 = (gy + PH - 1) // 8
    if r0 < ra:
        r0 = ra
    if r1 > rb:
        r1 = rb
    c0 = hx // 8
    c1 = (hx + PW - 1) // 8
    if c0 < 0:
        c0 = 0
    if c1 > ncols - 1:
        c1 = ncols - 1
    centre = hx + PW // 2
    flags = 0
    closest = -1
    closest_slope = 0
    box_row = 1 << 30
    slope_row = 1 << 30
    upper_slope = 0
    for r in range(r0, r1 + 1):
        if r < 0:
            continue
        if r >= nrows:
            if floor_below:
                # below the bottom of the map: solid floor
                flags |= CI_COLL | CI_BOX
                if r < box_row:
                    box_row = r
            continue
        cy = r * 8 - offy
        for c in range(c0, c1 + 1):
            t = int(tiles[r, c])
            if t == T_EMPTY or t == T_FAKE or t == T_SLOPE_NONE:
                continue
            if t == T_SOLID or t == T_ICE or t == T_SNOW:
                flags |= CI_COLL | CI_BOX
                if t == T_ICE:
                    flags |= CI_ICE
                elif t == T_SNOW:
                    flags |= CI_SNOW
                if r < box_row:
                    box_row = r
            elif t == T_NOWIND:
                flags |= CI_NOWIND
            elif t == T_WATER:
                flags |= CI_WATER
            elif t == T_SAND:
                flags |= CI_SAND
            elif t == T_QUARK:
                if hy + PH <= cy + 8:
                    flags |= CI_QUARK
            elif _slope_hits(t, c * 8, cy, hx, hy):
                flags |= CI_COLL
                if r < slope_row:
                    slope_row = r
                    upper_slope = t
                dist = abs(c * 8 + 4 - centre)
                if closest < 0 or dist <= closest:
                    closest = dist
                    closest_slope = t
    slope = closest_slope
    if box_row < (1 << 30) and upper_slope != 0 and box_row <= slope_row and \
            (upper_slope == T_SLOPE_TL or upper_slope == T_SLOPE_TR):
        slope = 0
    return flags | (slope << CI_SLOPE_SHIFT) | (closest_slope << CI_NORMAL_SHIFT)


@_jit
def _screen_rows(nscr, scr):
    ra = (nscr - 1 - scr) * 45
    return ra, ra + 44


@_jit
def _try_screen(tiles, offy, nscr, scr, hx, hy):
    ra, rb = _screen_rows(nscr, scr)
    if scr == 0:
        rb = 1 << 30
    return _try_rows(tiles, offy, ra, rb, hx, hy, scr == 0)


@_jit
def _infos(tiles, offy, nscr, cam, hx, hy):
    """TryCollision of the screens cam-1, cam, cam+1 (0 for screens that do
    not exist or that the hitbox's rows do not reach)."""
    gy = hy + offy
    r0 = gy // 8
    r1 = (gy + PH - 1) // 8
    a = 0
    b = 0
    c = 0
    for k in range(3):
        scr = cam - 1 + k
        if scr < 0 or scr >= nscr:
            continue
        ra = (nscr - 1 - scr) * 45
        rb = ra + 44
        if scr == 0:
            rb = 1 << 30
        if r1 < ra or r0 > rb:
            continue
        info = _try_rows(tiles, offy, ra, rb, hx, hy, scr == 0)
        if k == 0:
            a = info
        elif k == 1:
            b = info
        else:
            c = info
    return a, b, c


@_jit
def _merge(nscr, cam, below, here, above):
    """LevelManager.CheckCollisionInternal: merge the camera screen's info
    with its neighbours (cam-1 first, then cam+1)."""
    hit = (here & CI_COLL) != 0
    agg = here
    for k in range(2):
        i = cam - 1 if k == 0 else cam + 1
        if i < 0 or i >= nscr:
            continue
        info3 = below if k == 0 else above
        if (info3 & CI_COLL) == 0:
            if info3 & CI_SAND:
                agg |= CI_SAND
            continue
        if not hit:
            # the neighbour becomes the baseline; the camera screen's blocks
            # are added except water and quark
            agg = info3 | (here & CI_FLAGS & ~(CI_WATER | CI_QUARK))
        else:
            agg |= info3 & CI_FLAGS
            st = (info3 >> CI_SLOPE_SHIFT) & 15
            if st != 0:
                agg = (agg & ~(15 << CI_SLOPE_SHIFT)) | (st << CI_SLOPE_SHIFT)
        hit = True
    return agg


@_jit
def check_collision(tiles, offy, nscr, cam, hx, hy):
    """LevelManager.CheckCollision for the hitbox at integer (hx, hy) with the
    camera on screen ``cam``: info bits merged as in CheckCollisionInternal."""
    a, b, c = _infos(tiles, offy, nscr, cam, hx, hy)
    return _merge(nscr, cam, a, b, c)


@_jit
def _any_flag(tiles, offy, nscr, cam, hx, hy, flag):
    """Is ``flag`` set for any of the three screens around the camera
    (LevelManager.GetCollisionInfo / IsInWater)."""
    a, b, c = _infos(tiles, offy, nscr, cam, hx, hy)
    return ((a | b | c) & flag) != 0


@_jit
def _in_wind(tiles, offy, nscr, cam, hx, hy):
    """GetCollisionInfo(...).IsInWind() for a windy camera screen."""
    a, b, c = _infos(tiles, offy, nscr, cam, hx, hy)
    if b & CI_NOWIND:
        return False
    if cam == 0 and (c & CI_NOWIND):
        return False
    return True


@_jit
def collide(tiles, offy, hx, hy):
    """Collision info for the hitbox at integer (hx, hy) over the whole map
    (one big screen). Used by tools; the kernel uses ``check_collision``."""
    return _try_rows(tiles, offy, -(1 << 30), 1 << 30, hx, hy, True)


@_jit
def _slope_of(info):
    """Effective slope code of a collision (0 = flat box contact)."""
    return (info >> CI_SLOPE_SHIFT) & 15


@_jit
def _normal_of(info):
    return (info >> CI_NORMAL_SHIFT) & 15


@_jit
def in_nowind(tiles, offy, hx, hy):
    return (collide(tiles, offy, hx, hy) & CI_NOWIND) != 0


@_jit
def wind_velocity(tick, offset, scale, code=1):
    """Port of WindManager.CurrentVelocityRaw.

    ``code`` is the per-screen wind code of ``Level.wind``: 1 is the default
    oscillating wind; 2 + (intensity - 1) + 16 * kind encodes the coloured
    variants (kind 0 oscillating, 1 constant leftwards, 2 constant rightwards).
    """
    seconds = tick * TARGET_ELAPSED_SECONDS + offset
    # TimeSpan.FromSeconds rounds to whole milliseconds (.NET Framework)
    ms = math.floor(seconds * 1000.0 + 0.5)
    t = f32(ms / 1000.0)
    phase = f32(t * WIND_PHASE_RATE)
    wave = f32(math.sin(phase))
    if f32(math.cos(phase)) > 0:
        wave = f32(f32(wave * 2.0) + 1.0)
    else:
        wave = f32(f32(wave * 2.0) - 1.0)
    if wave > 1.0:
        wave = 1.0
    if wave < -1.0:
        wave = -1.0
    if code >= 2:
        k = code - 2
        kind = k // 16
        strength = f32(WIND_INTENSITY_STEP * (k % 16 + 1))
        if kind == 1:
            return f32(-strength * scale)
        if kind == 2:
            return f32(strength * scale)
        return f32(f32(wave * strength) * scale)
    return f32(f32(wave * WIND_STRENGTH) * scale)


@_jit
def screen_of_y(hy, n_screens):
    cy = hy + PH // 2
    s = -int(math.floor(cy / 360.0))
    if s < 0:
        s = 0
    if s > n_screens - 1:
        s = n_screens - 1
    return s


@_jit
def _camera_target(hy):
    """-(int)Math.Floor((float)centre.Y / 360f)"""
    cy = hy + PH // 2
    return -int(math.floor(f32(cy / 360.0)))


@_jit
def _camera_set(cam, num, nscr, params):
    """Camera.UpdateCameraWithVelocity once the move is allowed."""
    cam = num
    if cam > nscr - 1:
        cam = nscr - 1
    n_end = int(params[P_N_ENDINGS])
    for k in range(n_end):
        if cam == int(params[P_ENDINGS + ENDING_STRIDE * k]) + 1:
            cam -= 1
    if cam < 0:
        cam = 0
    return cam


@_jit
def check_endings(s, params):
    """EndingManager.CheckWin on the current state; returns the 1-based
    ending reached or 0."""
    if s[S_ON_GROUND] == 0:
        return 0
    cam = int(s[S_SCREEN])
    cx = _trunc(s[S_X]) + PW // 2
    cy = _trunc(s[S_Y]) + PH // 2
    n_end = int(params[P_N_ENDINGS])
    for k in range(n_end):
        b = P_ENDINGS + ENDING_STRIDE * k
        if cam != int(params[b]):
            continue
        side = params[b + 1]
        if side > 0 and cx < params[b + 2]:
            continue
        if side < 0 and cx > params[b + 2]:
            continue
        if cy > params[b + 3]:
            continue
        return k + 1
    return 0


# ---------------------------------------------------------------------------
# The tick
# ---------------------------------------------------------------------------
@_jit
def tick(s, d, jump, tiles, wind_en, tp_left, tp_right, params):
    """Advance the game by one 60 Hz tick.

    s      : state vector (modified in place)
    d      : horizontal input (-1, 0, 1)
    jump   : jump button held (0/1)
    returns: event bit flags
    """
    ev = 0
    offy = int(params[P_OFFSET_Y])
    nscr = int(params[P_N_SCREENS])
    if s[S_GOAL] > 0:
        s[S_LAST_EVENTS] = 0
        s[S_TICK] += 1
        return 0

    # GameLoop runs before the entities: the win check sees last tick's state
    if params[P_N_ENDINGS] > 0:
        k = check_endings(s, params)
        if k > 0:
            s[S_GOAL] = 1
            s[S_ENDING] = k
            s[S_TICK] += 1
            s[S_LAST_EVENTS] = EV_GOAL
            return EV_GOAL

    x = s[S_X]
    y = s[S_Y]
    vx = s[S_VX]
    vy = s[S_VY]
    lvx = s[S_LVX]
    lvy = s[S_LVY]
    on_ground = s[S_ON_GROUND] > 0
    knocked = s[S_KNOCKED] > 0
    on_ice = s[S_ON_ICE] > 0
    on_snow = s[S_ON_SNOW] > 0
    in_water = s[S_IN_WATER] > 0
    in_sand = s[S_IN_SAND] > 0
    ring = s[S_RING] > 0
    boots = s[S_BOOTS] > 0
    cam = int(s[S_SCREEN])
    prev_cam = cam
    was_in_water = in_water

    # ------------------------------------------------------------ BodyComp
    if s[S_BODY_OFF] == 0:
        hx = _trunc(x)
        hy = _trunc(y)
        # CacheCollisionStateBehaviour (one query serves the sand and water
        # flags, the wind's no-wind test and the early exit: same position)
        ia, ib, ic = _infos(tiles, offy, nscr, cam, hx, hy)
        in_sand = ((ia | ib | ic) & CI_SAND) != 0
        in_water = ((ia | ib | ic) & CI_WATER) != 0
        # WindVelocityUpdateBehaviour
        wcode = int(wind_en[cam])
        if wcode == 0:
            s[S_WIND_LATCH] = 0
        elif on_ground or s[S_LAST_SCREEN] > cam:
            s[S_WIND_LATCH] = 1
        snow_eff = on_snow and not ring
        if (not snow_eff) and s[S_WIND_LATCH] > 0 and wcode != 0:
            if not ((ib & CI_NOWIND) or (cam == 0 and (ic & CI_NOWIND))):
                vx = f32(vx + wind_velocity(s[S_TICK], s[S_WIND_OFFSET],
                                            params[P_WIND_SCALE], wcode))
        # CacheLastScreenBehaviour
        s[S_LAST_SCREEN] = cam
        # CacheVelocityBehaviour
        lvx = vx
        lvy = vy
        # CollisionCheckEarlyExitBehaviour
        if _merge(nscr, cam, ia, ib, ic) & CI_COLL:
            x = f32(x + vx)
            y = f32(y + vy)
        else:
            # UpdateXPositionFromVelocity (sand x0.25, water x0.5)
            mx = vx
            if in_sand:
                mx = f32(mx * SAND_MOVE_MULTIPLIER)
            if in_water:
                mx = f32(mx * WATER_MULTIPLIER)
            x = f32(x + mx)

            # CapPositionBehaviour / HandlePlayerTeleportBehaviour
            capped = False
            tl = int(tp_left[cam])
            tr = int(tp_right[cam])
            if tl < 0 and tr < 0:
                ox = x
                if x < CAP_LEFT:
                    x = CAP_LEFT
                if x > CAP_RIGHT:
                    x = CAP_RIGHT
                capped = x != ox
            else:
                cxp = _trunc(x) + PW // 2
                if cxp < 0 or cxp > 480:
                    left = cxp < 0
                    if left:
                        x = f32(x + 480.0)
                    else:
                        x = f32(x - 480.0)
                    if tl >= 0 and tr >= 0:
                        tgt = tl if left else tr
                    elif tl >= 0:
                        tgt = tl
                    else:
                        tgt = tr
                    src = _trunc(f32(f32(-f32(y - 360.0)) / 360.0))
                    offset = 360 * (tgt - src)
                    ccy = _trunc(y) + PH // 2 - offset
                    if params[P_LEGACY_TELEPORT] > 0:
                        y = f32(y - offset)
                    else:
                        y = float(_trunc(y) - offset)
                    num = -int(math.floor(f32(ccy / 360.0)))
                    if num != cam:
                        cam = _camera_set(cam, num, nscr, params)
                    ev |= EV_TELEPORT

            # ResolveXCollisionBehaviour
            touched_x_slope = False
            info = check_collision(tiles, offy, nscr, cam, _trunc(x), _trunc(y))
            if (info & CI_COLL) or ((info & CI_SAND) and not in_sand):
                x = float(_trunc(x))
                step = 1 if vx > 0 else -1
                nxt = info
                it = 0
                while True:
                    info = nxt
                    x -= step
                    nxt = check_collision(tiles, offy, nscr, cam, _trunc(x), _trunc(y))
                    it += 1
                    if not ((nxt & CI_COLL) or ((nxt & CI_SAND) and not in_sand)) or it > 4096:
                        break
                sl = _slope_of(info)
                if sl == 0:
                    vx = f32(vx * -BOUNCE)
                    knocked = True
                    ev |= EV_BUMP_WALL
                else:
                    touched_x_slope = True
                    ev |= EV_SLOPE
                    if on_ground and abs(vx) <= WALK_SPEED and vy > 0:
                        vx = 0.0
                    elif _dot_normal(lvx, lvy, _normal_of(info)) < 0:
                        vx, vy = _composite(vx, vy, _normal_of(info))

            # UpdateYPositionFromVelocity (sand: x0.5 upwards, grounded
            # +1 px probe outside sand; water x0.5)
            my = vy
            if in_sand and vy <= 0:
                my = f32(my * SAND_JUMP_MULTIPLIER)
            if (not in_sand) and on_ground and vy > 0:
                y = f32(y + 1.0)
            if in_water:
                my = f32(my * WATER_MULTIPLIER)
            y = f32(y + my)

            # ResolveYCollisionBehaviour
            step_y = 1 if vy > 0 else -1
            touch_y = False
            on_snow = False
            info = check_collision(tiles, offy, nscr, cam, _trunc(x), _trunc(y))
            final_info = info        # the query at the body's final position
            sand_up = (not in_sand) and vy < 0
            if (info & CI_COLL) or ((info & CI_SAND) and sand_up):
                touch_y = True
                on_ice = (info & CI_ICE) != 0
                y = float(_trunc(y))
                nxt = info
                it = 0
                while True:
                    info = nxt
                    y -= step_y
                    nxt = check_collision(tiles, offy, nscr, cam, _trunc(x), _trunc(y))
                    it += 1
                    if not ((nxt & CI_COLL) or ((nxt & CI_SAND) and sand_up)) or it > 4096:
                        break
                final_info = nxt
                sl = _slope_of(info)
                nc = _normal_of(info)
                if step_y > 0:
                    sloped = sl != 0
                    on_ground = True
                    if sloped:
                        on_ground = False
                        if _dot_normal(lvx, lvy, nc) < 0:
                            vx, vy = _composite(vx, vy, nc)
                            knocked = True
                            ev |= EV_SLOPE
                        else:
                            sloped = False
                    if not sloped:
                        knocked = False
                        if info & CI_SNOW:
                            on_snow = True
                else:
                    ev |= EV_BUMP_CEIL
                    if sl == 0:
                        vx = f32(vx * BOUNCE)
                    elif _dot_normal(lvx, lvy, nc) < 0:
                        cvx, cvy = _composite(vx, vy, nc)
                        vy = cvy
                        vx = f32(vx * BOUNCE)
                if sl == 0 or vy <= 0 or on_ground:
                    vy = 0.0
            else:
                if (info & CI_QUARK) and vy == MAX_FALL:
                    if capped:
                        vx = 0.0
                    y = float(int(math.ceil(f32(y / 8.0))) * 8)
                    final_info = check_collision(tiles, offy, nscr, cam, _trunc(x), _trunc(y))
                on_ground = False
                on_ice = False

            # ApplyGravityBehaviour (water halves gravity)
            g = GRAVITY
            if in_water:
                g = f32(g * WATER_MULTIPLIER)
            vy = f32(vy + g)
            if vy > MAX_FALL:
                vy = MAX_FALL

            # ExecuteBlockBehaviours: ice friction, then sand
            ice_eff = on_ground if ring else on_ice
            if ice_eff:
                vx = _move_towards(vx, 0.0, ICE_FRICTION)
            in_sand = (final_info & CI_SAND) != 0
            if in_sand:
                if vy > SAND_FALL_SPEED:
                    vy = SAND_FALL_SPEED
                knocked = False

            # GuardtowerSoulBugFixBehaviour
            if touched_x_slope and (not touch_y) and step_y > 0 and not knocked:
                vx = f32(vx * -BOUNCE)
                knocked = True

    # ------------------------------------------------- input + behaviour tree
    if jump and s[S_JUMP_HELD] == 0:
        s[S_CAN_JUMP] = 1
    if not jump:
        s[S_CAN_JUMP] = 0
    s[S_JUMP_HELD] = 1 if jump else 0
    ice_eff = on_ground if ring else on_ice
    snow_eff = on_snow and not ring
    ground = on_ground or in_sand          # the IsOnGround node
    consecutive = s[S_WAS_GROUNDED] > 0
    run_jump = False
    jump_in_air = False
    if ground:
        # ---- FailState
        fs = 2   # 0 running, 1 success, 2 failure
        if s[S_SPLAT] > 0:
            if not consecutive:
                # ResumeRun: Start() again and restart the timer
                if not ice_eff:
                    vx = 0.0
                s[S_SPLATS] += 1
                ev |= EV_SPLAT
                s[S_SPLAT_TIME] = 0.0
            fs = 0
        elif lvy == MAX_FALL:
            s[S_SPLAT] = 1
            s[S_SPLAT_TIME] = 0.0
            s[S_SPLAT_DONE] = 0
            s[S_CHARGING] = 0
            s[S_CHARGE] = 0.0
            if not ice_eff:
                vx = 0.0
            s[S_SPLATS] += 1
            ev |= EV_SPLAT
            fs = 0
        if fs == 0:
            if not ice_eff:
                vx = 0.0
            if s[S_SPLAT_DONE] == 0 and on_ground:
                s[S_SPLAT_TIME] = f32(s[S_SPLAT_TIME] + DT)
                if s[S_SPLAT_TIME] >= params[P_SPLAT_TIME]:
                    s[S_SPLAT_DONE] = 1
            if s[S_SPLAT_DONE] > 0 and (jump or d != 0):
                s[S_SPLAT] = 0
                s[S_SPLAT_TIME] = 0.0
                s[S_SPLAT_DONE] = 0
                fs = 1
                ev |= EV_GETUP
        if fs == 2:
            run_jump = True
    else:
        # airborne: the jump keeps charging for the leniency pause (2 ticks)
        if s[S_AIR_STAGE] == 0:
            if s[S_CHARGING] > 0:
                s[S_AIR_CHARGE] = f32(s[S_AIR_CHARGE] + DT)
                if s[S_AIR_CHARGE] >= AIR_CHARGE_LIMIT:
                    s[S_AIR_STAGE] = 1
                run_jump = True
                jump_in_air = True
            else:
                s[S_AIR_STAGE] = 1
        elif s[S_CHARGING] > 0:
            # JumpState is no longer run: it restarts on landing
            s[S_CHARGING] = 0
            s[S_CHARGE] = 0.0

    js = 2
    if run_jump:
        # ---- JumpState
        if s[S_CHARGING] == 0:
            # OnNewRun: timer and direction buffer reset
            s[S_CHARGE] = 0.0
            s[S_H0] = 0
            s[S_H1] = 0
            s[S_H2] = 0
            s[S_H3] = 0
        s[S_H3] = s[S_H2]
        s[S_H2] = s[S_H1]
        s[S_H1] = s[S_H0]
        s[S_H0] = d
        go = True
        if s[S_CHARGE] == 0.0:
            # TryConsumeJump
            go = s[S_CAN_JUMP] > 0
            s[S_CAN_JUMP] = 0
        if not go:
            js = 2
            s[S_CHARGING] = 0
        else:
            if s[S_CHARGE] == 0.0 and jump:
                if not ice_eff:
                    vx = 0.0
            if (not ice_eff) and on_ground:
                vx = 0.0
            mult = WATER_MULTIPLIER if in_water else 1.0
            s[S_CHARGE] = f32(s[S_CHARGE] + f32(DT * mult))
            if s[S_CHARGE] >= JUMP_TIME or not jump:
                power = f32(s[S_CHARGE] / JUMP_TIME)
                if power > 1.0:
                    power = 1.0
                direction = d
                if direction == 0:
                    if s[S_H0] != 0:
                        direction = int(s[S_H0])
                    elif s[S_H1] != 0:
                        direction = int(s[S_H1])
                    elif s[S_H2] != 0:
                        direction = int(s[S_H2])
                    elif s[S_H3] != 0:
                        direction = int(s[S_H3])
                s[S_CHARGING] = 0
                js = 1
                do_jump = True
                if snow_eff and power < SNOW_BOOST_POWER:
                    if not (power > SNOW_MIN_POWER):
                        do_jump = False
                    else:
                        power = SNOW_BOOST_POWER
                if do_jump:
                    vy = f32(JUMP * power)
                    vx = f32(vx + f32(direction * SPEED))
                    if vx < -SPEED:
                        vx = -SPEED
                    if vx > SPEED:
                        vx = SPEED
                    if direction != 0:
                        s[S_FACING] = direction
                    s[S_CHARGE] = 0.0
                    s[S_H0] = 0
                    s[S_H1] = 0
                    s[S_H2] = 0
                    s[S_H3] = 0
                    s[S_JUMPS] += 1
                    s[S_LAST_POWER] = power
                    ev |= EV_JUMP
            else:
                js = 0
                s[S_CHARGING] = 1
        if js == 1 and not jump_in_air:
            # EnableComponent(body) after a jump
            s[S_BODY_OFF] = 0
    if ground and run_jump and js == 2:
        # ---- Walk
        if not (vy < 0 and in_sand):
            num = f32(d * WALK_SPEED)
            if boots:
                num = 0.0
            if snow_eff:
                vx = 0.0
            elif ice_eff:
                if num != 0 and (_sign(num) != _sign(vx) or abs(num) > abs(vx)):
                    vx = _move_towards(vx, num, ICE_WALK_ACCEL)
            else:
                vx = num
            if d != 0:
                s[S_FACING] = d

    if ground:
        # ---- Giant Boots landing thud (runs in parallel with the controls)
        if not consecutive:
            s[S_THUD] = 0
        stage = int(s[S_THUD])
        if stage == 0:
            if boots:
                s[S_BODY_OFF] = 1
                s[S_THUD_T] = 0.0
                stage = 1
        if stage == 1:
            s[S_THUD_T] = f32(s[S_THUD_T] + DT)
            if s[S_THUD_T] >= BOOTS_FREEZE_TIME:
                s[S_BODY_OFF] = 0
                s[S_THUD_T] = 0.0
                stage = 2
        if stage == 2:
            s[S_THUD_T] = f32(s[S_THUD_T] + DT)
            if s[S_THUD_T] >= BOOTS_SHAKE_TIME:
                stage = 3
        s[S_THUD] = stage
        s[S_AIR_STAGE] = 0
        s[S_AIR_CHARGE] = 0.0
    s[S_WAS_GROUNDED] = 1 if ground else 0

    # ------------------------------------------------------ CameraFollowComp
    hy = _trunc(y)
    if not in_sand:
        num = _camera_target(hy)
        if num != cam:
            if on_ground:
                cam = _camera_set(cam, num, nscr, params)
            elif vy != 0:
                if _sign(num - cam) == _sign(-vy) and not (vy < 0 and abs(vy) < 3):
                    cam = _camera_set(cam, num, nscr, params)
    if cam > prev_cam and (ev & EV_TELEPORT) == 0:
        ev |= EV_SCREEN_UP
    if cam < prev_cam and (ev & EV_TELEPORT) == 0:
        ev |= EV_SCREEN_DOWN
        s[S_FALLS] += 1
    if in_water != was_in_water:
        ev |= EV_WATER
    if in_sand:
        ev |= EV_SAND

    # landing event
    if on_ground:
        if s[S_AIR_TICKS] > 0:
            ev |= EV_LAND
        s[S_AIR_TICKS] = 0
    else:
        s[S_AIR_TICKS] += 1

    # progress / legacy rectangle goal
    bottom = hy + PH
    if bottom < s[S_MAX_HEIGHT]:
        s[S_MAX_HEIGHT] = bottom
    hx = _trunc(x)
    if params[P_N_ENDINGS] == 0:
        if (hx < params[P_GOAL_X1] and hx + PW > params[P_GOAL_X0]
                and hy < params[P_GOAL_Y1] and hy + PH > params[P_GOAL_Y0]):
            s[S_GOAL] = 1
            s[S_ENDING] = 1
            ev |= EV_GOAL

    s[S_X] = x
    s[S_Y] = y
    s[S_VX] = vx
    s[S_VY] = vy
    s[S_LVX] = lvx
    s[S_LVY] = lvy
    s[S_ON_GROUND] = 1 if on_ground else 0
    s[S_KNOCKED] = 1 if knocked else 0
    s[S_ON_ICE] = 1 if on_ice else 0
    s[S_ON_SNOW] = 1 if on_snow else 0
    s[S_IN_WATER] = 1 if in_water else 0
    s[S_IN_SAND] = 1 if in_sand else 0
    s[S_SCREEN] = cam
    s[S_TICK] += 1
    if params[P_TOUCH + 2] > 0:
        hx = _trunc(x)
        hy = _trunc(y)
        tx = params[P_TOUCH]
        ty = params[P_TOUCH + 1]
        if (tx < hx + PW and hx < tx + params[P_TOUCH + 2]
                and ty < hy + PH and hy < ty + params[P_TOUCH + 3]):
            ev |= EV_TOUCH
    s[S_LAST_EVENTS] = ev
    return ev


@_jit
def run_inputs(s, dirs, jumps, tiles, wind_en, tp_left, tp_right, params):
    """Run a sequence of inputs; returns OR of all events."""
    ev = 0
    for i in range(dirs.shape[0]):
        ev |= tick(s, int(dirs[i]), int(jumps[i]), tiles, wind_en, tp_left, tp_right, params)
    return ev


@_jit
def _grounded(s):
    """On the ground, or sinking in quicksand: the IsOnGround node counts sand
    as ground, so a jump can be charged there."""
    return s[S_ON_GROUND] > 0 or (s[S_IN_SAND] > 0 and s[S_VY] >= 0)


@_jit
def is_settled(s):
    """Standing still on solid ground (or sinking in quicksand) and ready for
    the next command."""
    if s[S_GOAL] > 0:
        return True
    if not _grounded(s) or s[S_CHARGING] > 0 or s[S_KNOCKED] > 0:
        return False
    if s[S_BODY_OFF] > 0:
        return False
    if s[S_SPLAT] > 0:
        return s[S_SPLAT_DONE] > 0
    if (s[S_ON_ICE] > 0 or s[S_RING] > 0) and abs(s[S_VX]) > 0.05:
        return False
    return True


@_jit
def is_landed(s):
    """Back on the ground and ready for a command, possibly still sliding on
    ice (macros in ``landed`` mode stop here so momentum can be kept)."""
    if s[S_GOAL] > 0:
        return True
    if not _grounded(s) or s[S_CHARGING] > 0 or s[S_KNOCKED] > 0 or s[S_BODY_OFF] > 0:
        return False
    if s[S_SPLAT] > 0:
        return s[S_SPLAT_DONE] > 0
    return True


@_jit
def settle(s, tiles, wind_en, tp_left, tp_right, params, max_ticks, landed=False):
    """Release all inputs and run until the player is settled (or, with
    ``landed``, until he is back on the ground)."""
    ev = 0
    n = 0
    while n < max_ticks:
        ev |= tick(s, 0, 0, tiles, wind_en, tp_left, tp_right, params)
        n += 1
        if is_landed(s) if landed else is_settled(s):
            break
    return ev, n


@_jit
def macro_jump(s, direction, charge, tiles, wind_en, tp_left, tp_right, params, max_ticks,
               landed=False):
    """Charge a jump for ``charge`` ticks (2..36, power = charge/36) with the
    given direction held, release, and wait until landed.

    Returns (events, ticks used)."""
    ev = 0
    n = 0
    # make sure the button is released first (new press needed)
    if s[S_JUMP_HELD] > 0:
        ev |= tick(s, direction, 0, tiles, wind_en, tp_left, tp_right, params)
        n += 1
    # stand up from a splat (consumes one input tick)
    if s[S_SPLAT] > 0 and s[S_SPLAT_DONE] > 0:
        ev |= tick(s, direction, 0, tiles, wind_en, tp_left, tp_right, params)
        n += 1
    if charge < 2:
        charge = 2
    if charge > 36:
        charge = 36
    # hold until charge-1 increments are accumulated; the release tick adds
    # the last one, so power = charge / 36 (charge 36 auto-releases). In
    # water a tick adds half an increment: the same power takes twice as long
    target = (charge - 1) * DT - 1e-6
    if s[S_IN_WATER] > 0:
        target = (charge - WATER_MULTIPLIER) * DT - 1e-6
    started = False
    waited = 0
    while n < max_ticks:
        e = tick(s, direction, 1, tiles, wind_en, tp_left, tp_right, params)
        ev |= e
        n += 1
        if e & EV_JUMP:
            break
        if s[S_CHARGING] > 0:
            started = True
            if s[S_CHARGE] >= target:
                break
        else:
            waited += 1
            if started or waited > 8:
                # charge cancelled (walked off a ledge, landed hard, ...)
                break
    if (ev & EV_JUMP) == 0 and s[S_CHARGING] > 0:
        ev |= tick(s, direction, 0, tiles, wind_en, tp_left, tp_right, params)
        n += 1
    e2, n2 = settle(s, tiles, wind_en, tp_left, tp_right, params, max_ticks - n, landed)
    return ev | e2, n + n2


@_jit
def macro_walk(s, direction, ticks, tiles, wind_en, tp_left, tp_right, params, max_ticks,
               landed=False):
    """Walk for ``ticks`` ticks then wait until settled."""
    ev = 0
    n = 0
    if s[S_SPLAT] > 0 and s[S_SPLAT_DONE] > 0:
        ev |= tick(s, direction, 0, tiles, wind_en, tp_left, tp_right, params)
        n += 1
    while n < ticks and n < max_ticks:
        ev |= tick(s, direction, 0, tiles, wind_en, tp_left, tp_right, params)
        n += 1
    e2, n2 = settle(s, tiles, wind_en, tp_left, tp_right, params, max_ticks - n, landed)
    return ev | e2, n + n2


def new_state(x=C.START_X, y=C.START_Y, screen=None, n_screens=1, wind_offset=0.0):
    s = np.zeros(STATE_SIZE, dtype=np.float64)
    s[S_X] = x
    s[S_Y] = y
    s[S_VY] = 0.26  # as in a fresh save state (resting on the floor)
    s[S_FACING] = 1
    if screen is None:
        screen = max(0, min(n_screens - 1, -int(math.floor((int(y) + PH // 2) / 360.0))))
    s[S_SCREEN] = screen
    s[S_LAST_SCREEN] = screen
    s[S_MAX_HEIGHT] = int(y) + PH
    s[S_WIND_OFFSET] = wind_offset
    return s
