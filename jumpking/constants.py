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

# --- DLC blocks (JumpKing.PlayerValues, Sand/WaterBlockBehaviour) ----------
SAND_FALL_SPEED = _f32(0.75)        # max sinking speed inside sand
SAND_MOVE_MULTIPLIER = _f32(0.25)   # horizontal movement inside sand
SAND_JUMP_MULTIPLIER = _f32(0.5)    # upward movement inside sand
WATER_MULTIPLIER = _f32(0.5)        # movement, gravity and jump charge in water

# --- items ----------------------------------------------------------------
# Giant Boots: on landing the body freezes for a 0.05 s pause, then a 0.15 s
# screen shake follows (PlayerEntity.MakeBT); walking is disabled (Walk).
BOOTS_FREEZE_TIME = _f32(0.05)
BOOTS_SHAKE_TIME = _f32(0.15)

# --- wind -----------------------------------------------------------------
WIND_PHASE_RATE = _f32(0.48124886)   # rad / s
WIND_STRENGTH = _f32(0.1)            # px / tick^2 at full wave
WIND_INTENSITY_STEP = _f32(0.0125)   # per intensity level of (R, 255, 0) wind

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

# The DLC areas of the shared 169-screen world (Content/gui/location_settings.xml;
# the section names of the community TAS routes for the screens in between).
AREAS_SECRET = (
    ("Imp Room", 45, 45),
    ("New Babe+ Entrance", 46, 46),
)
AREAS_NBP = (
    ("Brightcrown Woods", 47, 52),
    ("Colossal Dungeon", 53, 59),
    ("False King's Castle", 60, 63),
    ("Underburg", 64, 70),
    ("Lost Frontier", 71, 77),
    ("Hidden Kingdom", 78, 83),
    ("Black Sanctum", 84, 89),
    ("Deep Ruin", 90, 94),
    ("The Dark Tower", 95, 100),
)
AREAS_GHOST = (
    ("Bog", 102, 108),
    ("Moulding Manor", 109, 116),
    ("Bugstalk", 117, 123),
    ("House of Nine Lives", 124, 130),
    ("The Phantom Tower", 131, 139),
    ("Halted Ruin", 140, 147),
    ("Tower of Antumbra", 148, 154),
    ("Philosopher's Forest", 156, 160),
    ("Hole", 161, 164),
)
ALL_AREAS = AREAS + AREAS_SECRET + AREAS_NBP + AREAS_GHOST


def area_of_screen(screen_index):
    """Area name for a 0-based screen index."""
    n = screen_index + 1
    for name, a, b in ALL_AREAS:
        if a <= n <= b:
            return name
    return "Empty"


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
