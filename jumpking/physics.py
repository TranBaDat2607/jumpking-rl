"""Deterministic Jump King physics kernel.

The whole player/world state lives in one float64 vector (see the ``S_*``
indices) so that it can be copied, hashed, batched and restored in O(1).
Every arithmetic result that is float32 in the original game is rounded to
float32 with ``f32`` to reproduce its numerics.

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
    for f in glob.glob(os.path.join(cache_dir, "physics.*.nb[ic]")):
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
N_TILE_TYPES = 10

# collision info bit flags
CI_COLL = 1
CI_BOX = 2
CI_ICE = 4
CI_SNOW = 8
CI_SLOPE_SHIFT = 4     # slope tile code stored in bits 4..7

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
S_ON_ICE = 8
S_ON_SNOW = 9
S_WIND_LATCH = 10
S_LAST_SCREEN = 11
S_SCREEN = 12        # camera screen (0-based)
S_JUMP_HELD = 13
S_CAN_JUMP = 14
S_CHARGING = 15
S_CHARGE = 16        # seconds (float32)
S_AIR_CHARGE = 17
S_H0 = 18            # direction history (most recent first)
S_H1 = 19
S_H2 = 20
S_H3 = 21
S_WAS_GROUNDED = 22
S_SPLAT = 23
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
STATE_SIZE = 37

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

# level parameter vector layout
P_SPLAT_TIME = 0
P_GOAL_X0 = 1
P_GOAL_Y0 = 2
P_GOAL_X1 = 3
P_GOAL_Y1 = 4
P_WIND_OFFSET = 5    # default wind clock offset copied into new states
P_N_SCREENS = 6
P_OFFSET_Y = 7       # world y -> grid y offset (= 360 * (n_screens - 1))
P_WIND_SCALE = 8     # 1.0 = original
PARAM_SIZE = 9

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
TARGET_ELAPSED_SECONDS = C.TARGET_ELAPSED_SECONDS
PW = C.PLAYER_W
PH = C.PLAYER_H
UNIT = float(np.float32(0.7071067811865475))


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
def _approach(value, target, step):
    if value < target:
        return min(target, f32(value + step))
    return max(target, f32(value - step))


@_jit
def _slope_normal(code):
    if code == T_SLOPE_TL:
        return -UNIT, -UNIT
    if code == T_SLOPE_TR:
        return UNIT, -UNIT
    if code == T_SLOPE_BL:
        return -UNIT, UNIT
    if code == T_SLOPE_BR:
        return UNIT, UNIT
    return 0.0, 0.0


@_jit
def _slope_hits(code, cx, cy, hx, hy):
    """Does the integer rect [hx,hx+PW)x[hy,hy+PH) overlap the solid half of a
    slope cell whose top-left corner is (cx, cy)?"""
    a0 = max(hx, cx) - cx
    a1 = min(hx + PW, cx + 8) - cx
    b0 = max(hy, cy) - cy
    b1 = min(hy + PH, cy + 8) - cy
    if a1 <= a0 or b1 <= b0:
        return False
    if code == T_SLOPE_TL:
        return a1 + b1 > 8
    if code == T_SLOPE_TR:
        return b1 - a0 > 0
    if code == T_SLOPE_BL:
        return a1 - b0 > 0
    return a0 + b0 < 8


@_jit
def collide(tiles, offy, hx, hy):
    """Collision info bit field for the player hitbox at integer (hx, hy)."""
    nrows = tiles.shape[0]
    ncols = tiles.shape[1]
    c0 = hx // 8
    c1 = (hx + PW - 1) // 8
    gy = hy + offy
    r0 = gy // 8
    r1 = (gy + PH - 1) // 8
    flags = 0
    slope = 0
    for r in range(r0, r1 + 1):
        if r < 0:
            continue
        if r >= nrows:
            # below the bottom of the map: solid floor
            flags |= CI_COLL | CI_BOX
            continue
        for c in range(c0, c1 + 1):
            if c < 0 or c >= ncols:
                continue
            t = tiles[r, c]
            if t == T_EMPTY or t == T_NOWIND or t == T_FAKE:
                continue
            if t == T_SOLID:
                flags |= CI_COLL | CI_BOX
            elif t == T_ICE:
                flags |= CI_COLL | CI_BOX | CI_ICE
            elif t == T_SNOW:
                flags |= CI_COLL | CI_BOX | CI_SNOW
            else:
                if _slope_hits(t, c * 8, r * 8 - offy, hx, gy - offy):
                    flags |= CI_COLL
                    if slope == 0:
                        slope = t
    return flags | (slope << CI_SLOPE_SHIFT)


@_jit
def _slope_of(info):
    """Effective slope code of a collision (boxes take precedence)."""
    if info & CI_BOX:
        return 0
    return (info >> CI_SLOPE_SHIFT) & 15


@_jit
def in_nowind(tiles, offy, hx, hy):
    nrows = tiles.shape[0]
    ncols = tiles.shape[1]
    gy = hy + offy
    for r in range(gy // 8, (gy + PH - 1) // 8 + 1):
        if r < 0 or r >= nrows:
            continue
        for c in range(hx // 8, (hx + PW - 1) // 8 + 1):
            if c < 0 or c >= ncols:
                continue
            if tiles[r, c] == T_NOWIND:
                return True
    return False


@_jit
def wind_velocity(tick, offset, scale):
    """Port of WindManager.CurrentVelocityRaw for the default intensity."""
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
def tick(s, d, jump, tiles, wind_en, tp_left, tp_right, params):
    """Advance the game by one 60 Hz tick.

    s      : state vector (modified in place)
    d      : horizontal input (-1, 0, 1)
    jump   : jump button held (0/1)
    returns: event bit flags
    """
    ev = 0
    offy = int(params[P_OFFSET_Y])
    n_screens = int(params[P_N_SCREENS])
    if s[S_GOAL] > 0:
        s[S_LAST_EVENTS] = 0
        s[S_TICK] += 1
        return 0

    x = s[S_X]
    y = s[S_Y]
    vx = s[S_VX]
    vy = s[S_VY]
    on_ground = s[S_ON_GROUND] > 0
    knocked = s[S_KNOCKED] > 0
    on_ice = s[S_ON_ICE] > 0
    on_snow = s[S_ON_SNOW] > 0
    screen = int(s[S_SCREEN])
    prev_screen = screen

    # ------------------------------------------------------------------ body
    lvx = vx
    lvy = vy

    # wind
    if wind_en[screen] == 0:
        s[S_WIND_LATCH] = 0
    elif on_ground or s[S_LAST_SCREEN] > screen:
        s[S_WIND_LATCH] = 1
    if (not on_snow) and s[S_WIND_LATCH] > 0 and wind_en[screen] != 0:
        if not in_nowind(tiles, offy, int(x), int(y)):
            vx = f32(vx + wind_velocity(s[S_TICK], s[S_WIND_OFFSET], params[P_WIND_SCALE]))
    s[S_LAST_SCREEN] = screen

    # x movement
    x = f32(x + vx)

    # cap / side teleport
    tl = tp_left[screen]
    tr = tp_right[screen]
    if tl < 0 and tr < 0:
        if x < -PW / 2.0:
            x = -PW / 2.0
        if x > 480 - PW // 2:
            x = float(480 - PW // 2)
    else:
        cxp = int(x) + PW // 2
        if cxp < 0 or cxp > 480:
            left = cxp < 0
            if left:
                tgt = tl if tl >= 0 else tr
                x = f32(x + 480.0)
            else:
                tgt = tr if tr >= 0 else tl
                x = f32(x - 480.0)
            src = int(-(y - 360.0) / 360.0)
            offset = 360 * (tgt - src)
            y = float(int(y) - offset)
            screen = screen_of_y(int(y), n_screens)
            ev |= EV_TELEPORT

    # resolve x
    touched_x_slope = False
    info = collide(tiles, offy, int(x), int(y))
    if info & CI_COLL:
        x = float(int(x))
        step = 1 if vx > 0 else -1
        nxt = info
        it = 0
        while True:
            info = nxt
            x -= step
            nxt = collide(tiles, offy, int(x), int(y))
            it += 1
            if (nxt & CI_COLL) == 0 or it > 4096:
                break
        sl = _slope_of(info)
        if sl == 0:
            vx = f32(vx * -BOUNCE)
            knocked = True
            ev |= EV_BUMP_WALL
        else:
            touched_x_slope = True
            ev |= EV_SLOPE
            nx, ny = _slope_normal(sl)
            if on_ground and abs(vx) <= WALK_SPEED and vy > 0:
                vx = 0.0
            elif f32(f32(lvx * nx) + f32(lvy * ny)) < 0:
                dd = f32(f32(vx * nx) + f32(vy * ny))
                vx = f32(vx - f32(dd * nx))
                vy = f32(vy - f32(dd * ny))

    # y movement. While grounded the native SandBlock behaviour adds one pixel
    # to downward motion (outside sand), so a standing king re-touches the
    # floor every tick instead of hovering above it.
    dy = vy
    if on_ground and vy > 0:
        dy = f32(vy + 1.0)
    y = f32(y + dy)

    # resolve y
    step_y = 1 if vy > 0 else -1
    touch_y = False
    on_snow = False
    info = collide(tiles, offy, int(x), int(y))
    if info & CI_COLL:
        touch_y = True
        on_ice = (info & CI_ICE) != 0
        y = float(int(y))
        nxt = info
        it = 0
        while True:
            info = nxt
            y -= step_y
            nxt = collide(tiles, offy, int(x), int(y))
            it += 1
            if (nxt & CI_COLL) == 0 or it > 4096:
                break
        sl = _slope_of(info)
        if step_y > 0:
            on_ground = True
            if sl != 0:
                on_ground = False
                nx, ny = _slope_normal(sl)
                if f32(f32(lvx * nx) + f32(lvy * ny)) < 0:
                    dd = f32(f32(vx * nx) + f32(vy * ny))
                    vx = f32(vx - f32(dd * nx))
                    vy = f32(vy - f32(dd * ny))
                    knocked = True
                    ev |= EV_SLOPE
                else:
                    sl = 0
            if sl == 0:
                knocked = False
                if info & CI_SNOW:
                    on_snow = True
        else:
            ev |= EV_BUMP_CEIL
            if sl == 0:
                vx = f32(vx * BOUNCE)
            else:
                nx, ny = _slope_normal(sl)
                if f32(f32(lvx * nx) + f32(lvy * ny)) < 0:
                    dd = f32(f32(vx * nx) + f32(vy * ny))
                    vy = f32(vy - f32(dd * ny))
                    vx = f32(vx * BOUNCE)
        if sl == 0 or vy <= 0 or on_ground:
            vy = 0.0
    else:
        on_ground = False
        on_ice = False

    # guard tower soul bug fix
    if touched_x_slope and (not touch_y) and step_y > 0 and not knocked:
        vx = f32(vx * -BOUNCE)
        knocked = True

    # gravity
    vy = f32(vy + GRAVITY)
    if vy > MAX_FALL:
        vy = MAX_FALL

    # materials: ice friction
    if on_ice:
        vx = _approach(vx, 0.0, ICE_FRICTION)

    # -------------------------------------------------------------- controls
    if jump and s[S_JUMP_HELD] == 0:
        s[S_CAN_JUMP] = 1
    if not jump:
        s[S_CAN_JUMP] = 0
    s[S_JUMP_HELD] = 1 if jump else 0
    ground = on_ground
    if ground and s[S_WAS_GROUNDED] == 0 and lvy == MAX_FALL:
        s[S_SPLAT] = 1
        s[S_SPLAT_TIME] = 0.0
        s[S_SPLAT_DONE] = 0
        s[S_CHARGING] = 0
        s[S_CHARGE] = 0.0
        s[S_SPLATS] += 1
        ev |= EV_SPLAT
    s[S_WAS_GROUNDED] = 1 if ground else 0

    controls_done = False
    if s[S_SPLAT] > 0:
        controls_done = True
        if not on_ice:
            vx = 0.0
        if s[S_SPLAT_DONE] == 0 and on_ground:
            s[S_SPLAT_TIME] = f32(s[S_SPLAT_TIME] + DT)
            if s[S_SPLAT_TIME] >= params[P_SPLAT_TIME]:
                s[S_SPLAT_DONE] = 1
        if s[S_SPLAT_DONE] > 0 and (jump or d != 0):
            s[S_SPLAT] = 0
            s[S_SPLAT_TIME] = 0.0
            ev |= EV_GETUP
    if not controls_done:
        if not ground:
            if s[S_CHARGING] == 0:
                controls_done = True
            else:
                s[S_AIR_CHARGE] = f32(s[S_AIR_CHARGE] + DT)
                if s[S_AIR_CHARGE] > AIR_CHARGE_LIMIT:
                    s[S_CHARGING] = 0
                    s[S_CHARGE] = 0.0
                    controls_done = True
        else:
            s[S_AIR_CHARGE] = 0.0
    if not controls_done:
        s[S_H3] = s[S_H2]
        s[S_H2] = s[S_H1]
        s[S_H1] = s[S_H0]
        s[S_H0] = d
        if s[S_CHARGING] > 0 or s[S_CAN_JUMP] > 0:
            controls_done = True
            if s[S_CHARGING] == 0:
                s[S_CAN_JUMP] = 0
                s[S_CHARGE] = 0.0
                s[S_CHARGING] = 1
                if not on_ice:
                    vx = 0.0
            if (not on_ice) and on_ground:
                vx = 0.0
            s[S_CHARGE] = f32(s[S_CHARGE] + DT)
            if s[S_CHARGE] >= JUMP_TIME or not jump:
                power = f32(s[S_CHARGE] / JUMP_TIME)
                if power > 1.0:
                    power = 1.0
                direction = 0
                if s[S_H0] != 0:
                    direction = int(s[S_H0])
                elif s[S_H1] != 0:
                    direction = int(s[S_H1])
                elif s[S_H2] != 0:
                    direction = int(s[S_H2])
                elif s[S_H3] != 0:
                    direction = int(s[S_H3])
                s[S_CHARGING] = 0
                s[S_CHARGE] = 0.0
                s[S_H0] = 0
                s[S_H1] = 0
                s[S_H2] = 0
                s[S_H3] = 0
                do_jump = True
                if on_snow and power < SNOW_BOOST_POWER:
                    if power <= SNOW_MIN_POWER:
                        do_jump = False
                    else:
                        power = SNOW_BOOST_POWER
                if do_jump:
                    vy = f32(JUMP * power)
                    vx = f32(vx + f32(direction * SPEED))
                    if vx > SPEED:
                        vx = SPEED
                    if vx < -SPEED:
                        vx = -SPEED
                    if direction != 0:
                        s[S_FACING] = direction
                    s[S_JUMPS] += 1
                    s[S_LAST_POWER] = power
                    ev |= EV_JUMP
    if not controls_done and ground:
        target = d * WALK_SPEED
        if on_snow:
            vx = 0.0
        elif on_ice:
            if target != 0 and (_sign(target) != _sign(vx) or abs(target) > abs(vx)):
                vx = _approach(vx, target, f32(ICE_FRICTION * 2.0))
        else:
            vx = target
        if d != 0:
            s[S_FACING] = d

    # ---------------------------------------------------------------- camera
    hy = int(y)
    target_screen = -int(math.floor((hy + PH // 2) / 360.0))
    if target_screen != screen and vy != 0:
        if _sign(target_screen - screen) == _sign(-vy) and not (vy < 0 and abs(vy) < 3):
            if target_screen < 0:
                target_screen = 0
            if target_screen > n_screens - 1:
                target_screen = n_screens - 1
            screen = target_screen
    if screen > prev_screen and (ev & EV_TELEPORT) == 0:
        ev |= EV_SCREEN_UP
    if screen < prev_screen and (ev & EV_TELEPORT) == 0:
        ev |= EV_SCREEN_DOWN
        s[S_FALLS] += 1

    # landing event
    if on_ground:
        if s[S_AIR_TICKS] > 0:
            ev |= EV_LAND
        s[S_AIR_TICKS] = 0
    else:
        s[S_AIR_TICKS] += 1

    # progress / goal
    bottom = hy + PH
    if bottom < s[S_MAX_HEIGHT]:
        s[S_MAX_HEIGHT] = bottom
    hx = int(x)
    if (hx < params[P_GOAL_X1] and hx + PW > params[P_GOAL_X0]
            and hy < params[P_GOAL_Y1] and hy + PH > params[P_GOAL_Y0]):
        s[S_GOAL] = 1
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
    s[S_SCREEN] = screen
    s[S_TICK] += 1
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
def is_settled(s):
    """Standing still on solid ground and ready for the next command."""
    if s[S_GOAL] > 0:
        return True
    if s[S_ON_GROUND] == 0 or s[S_CHARGING] > 0 or s[S_KNOCKED] > 0:
        return False
    if s[S_SPLAT] > 0:
        return s[S_SPLAT_DONE] > 0
    if s[S_ON_ICE] > 0 and abs(s[S_VX]) > 0.05:
        return False
    return True


@_jit
def settle(s, tiles, wind_en, tp_left, tp_right, params, max_ticks):
    """Release all inputs and run until the player is settled."""
    ev = 0
    n = 0
    while n < max_ticks:
        ev |= tick(s, 0, 0, tiles, wind_en, tp_left, tp_right, params)
        n += 1
        if is_settled(s):
            break
    return ev, n


@_jit
def macro_jump(s, direction, charge, tiles, wind_en, tp_left, tp_right, params, max_ticks):
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
    # the last one, so power = charge / 36 (charge 36 auto-releases)
    target = (charge - 1) * DT - 1e-6
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
    e2, n2 = settle(s, tiles, wind_en, tp_left, tp_right, params, max_ticks - n)
    return ev | e2, n + n2


@_jit
def macro_walk(s, direction, ticks, tiles, wind_en, tp_left, tp_right, params, max_ticks):
    """Walk for ``ticks`` ticks then wait until settled."""
    ev = 0
    n = 0
    if s[S_SPLAT] > 0 and s[S_SPLAT_DONE] > 0:
        ev |= tick(s, direction, 0, tiles, wind_en, tp_left, tp_right, params)
        n += 1
    while n < ticks and n < max_ticks:
        ev |= tick(s, direction, 0, tiles, wind_en, tp_left, tp_right, params)
        n += 1
    e2, n2 = settle(s, tiles, wind_en, tp_left, tp_right, params, max_ticks - n)
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
