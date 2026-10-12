"""Researched Jump King constants (see docs/RESEARCH.md for sources).

All physics values are float32 in the original game; they are stored here as
Python floats that are exactly representable as float32.
"""
import json
import os

import numpy as np


def _f32(v):
    return float(np.float32(v))


# --- timing ---------------------------------------------------------------
FPS = 60
# The .NET Framework build ticks every 17 ms (TargetElapsedTime); this clock
# drives the wind and the in-game timer. Physics callbacks always receive a
# fixed 1/60 s delta (BehaviourContext.FrameDelta).
TARGET_ELAPSED_SECONDS = 0.017
DT = _f32(1.0 / 60.0)

# --- screen / tiles -------------------------------------------------------
SCREEN_W = 480
SCREEN_H = 360
TILE = 8
SCREEN_TILES_W = 60
SCREEN_TILES_H = 45
ATLAS_COLS = 13                              # 780 x 585 atlas -> 13 x 13 screens
ATLAS_SCREENS = ATLAS_COLS * ATLAS_COLS

# --- player ---------------------------------------------------------------
PLAYER_W = 18
PLAYER_H = 26

GRAVITY = _f32(0.2571429)
JUMP = _f32(-8.742858)
SPEED = _f32(3.5)
MAX_FALL = _f32(10.0)
JUMP_TIME = _f32(0.6)
WALK_SPEED = _f32(1.5)
BOUNCE = _f32(0.5)
ICE_FRICTION = _f32(0.1)
AIR_CHARGE_LIMIT = _f32(1.0 / 30.0)
SPLAT_TIME = _f32(0.5)    # PlayerValues.get_SPLAT_TIME in JumpKing.exe

# snow: weak jumps are suppressed / boosted
SNOW_MIN_POWER = _f32(0.2)
SNOW_BOOST_POWER = _f32(0.3)

# --- wind -----------------------------------------------------------------
WIND_PHASE_RATE = _f32(0.48124886)   # rad / s
WIND_STRENGTH = _f32(0.1)            # px / tick^2 at full wave

# --- start position (screen 1, from the JumpKingTAS base-game route) --------
START_X = 231.0
START_Y = 302.0

# number of base game screens
BASE_SCREENS = 43

AREAS = (
    # name, first screen (1-based), last screen (1-based)
    ("Redcrown Woods", 1, 5),
    ("Colossal Drain", 6, 10),
    ("False King's Keep", 11, 14),
    ("Bargainburg", 15, 19),
    ("Great Frontier", 20, 25),
    ("Windswept Bluff", 26, 26),
    ("Stormwall Pass", 27, 32),
    ("Chapel Perilous", 33, 36),
    ("Blue Ruin", 37, 39),
    ("The Tower", 40, 43),
)


def area_of_screen(screen_index):
    """Area name for a 0-based screen index."""
    n = screen_index + 1
    for name, a, b in AREAS:
        if a <= n <= b:
            return name
    return "Unknown"


# --- optional overrides --------------------------------------------------
# scripts/extract_constants.py --write reads the exact values from your own
# JumpKing.exe and stores them here; JUMPKING_CONSTANTS can point elsewhere.
OVERRIDE_PATH = os.environ.get("JUMPKING_CONSTANTS") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "constants_override.json")
OVERRIDES = {}
if os.path.isfile(OVERRIDE_PATH):
    with open(OVERRIDE_PATH, "r", encoding="utf-8") as _fh:
        OVERRIDES = json.load(_fh)
    for _k, _v in OVERRIDES.items():
        if _k in globals() and isinstance(globals()[_k], float):
            globals()[_k] = _f32(_v)
