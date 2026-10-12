# Jump King base game: researched mechanics

These notes record what the clone reproduces and where each value comes from.
Everything is per game tick. The game runs a fixed 60 Hz update
(`TargetElapsedTime` = 166667 × 100 ns ≈ 0.0166667 s). Coordinates are in
pixels with Y pointing down.

## Sources

| Ref | Source | Used for |
| --- | --- | --- |
| S1 | `qequ/jump-king-parabole` (mod; constants read from `JumpKing.PlayerValues`) | gravity, jump velocity, speed, max fall, charge time, hitbox 18×26 |
| S2 | `anyalink99/jump-king-mods`: `screen-solver/src/NativePlayer.cs`, `NativeResolution.cs`, `NativeWorld.cs`, `jk-runtime/src/Simulation/*` (a parity-tested port of the native body pipeline and controller) | update order, collision resolution, bounce, slopes, ice, snow, wind latch, splat, charge, direction buffer, camera, teleports |
| S3 | `mega-gameplay-expansion/docs/no-walk-off.md` (same repo) | walk speed ±1.5, ice friction 0.1 (and ×2 while walking) |
| S4 | `jk-runtime/src/Simulation/NativeWind.cs` | wind waveform and strength |
| S5 | Official workshop docs (`TeamNexile/jk-workshop-docs`), JumpKingPlus hitbox docs | hitbox atlas format, block colours, slope rules, snow minimum charge |
| S6 | `ShootMe/JumpKingTAS` (`LevelFiles/NewGame.tas`) | start position (231, 302), screen list per area, input timing (35 held frames + release = full jump) |
| S7 | Jump King wiki / PSNProfiles / Namu wiki | area names and screen counts (43 screens total) |

## Constants

| Name | Value | Source |
| --- | --- | --- |
| Hitbox | 18 × 26 px | S1 |
| `GRAVITY` | 0.2571429 px/tick² | S1 |
| `JUMP` | −8.742858 px/tick (scaled by charge power) | S1 |
| `SPEED` (jump horizontal) | 3.5 px/tick | S1 |
| `MAX_FALL` | 10 px/tick | S1 |
| `JUMP_TIME` | 0.6 s (36 ticks) | S1, S6 |
| `WALK_SPEED` | 1.5 px/tick | S2, S3 |
| `BOUNCE` | 0.5 | S2 (comment: "native walls multiply Vx by −BOUNCE (0.5)") |
| `ICE_FRICTION` | 0.1 px/tick² | S3 |
| Wind | `0.1 × wave(t)` px/tick² added to vx | S4 |
| `SPLAT_TIME` | **not found**; the clone uses 0.25 s of grounded time (configurable) | estimate |
| Screen | 480 × 360 px = 60 × 45 tiles of 8 px | S5 |

## Per-tick order

1. **Body update** (`BodyComp` behaviour list):
   1. `last_velocity = velocity`.
   2. **Wind**: if the screen has no wind, clear the wind latch. Otherwise set
      the latch when the player is on the ground or has just fallen in from a
      higher screen (`last_screen > screen`). If latched, not standing on snow and
      not inside a no-wind block, `vx += wind(t)`. Then `last_screen = screen`.
   3. **X move**: `x += vx`.
   4. **Cap / teleport**: screens without side teleports clamp
      `x ∈ [−w/2, 480 − w/2]`. Screens with teleports wrap the player to the linked
      screen once the hitbox centre leaves `[0, 480]`.
   5. **Resolve X**: if the hitbox overlaps solid, truncate `x` to an integer and
      step back one pixel at a time against the motion until clear. A flat wall
      gives `vx *= −0.5` and the knocked state. A slope gives a slide: the
      velocity is projected onto the slope surface when `last_velocity · n < 0`,
      or `vx = 0` when walking slowly into it on the ground.
   6. **Y move**: `y += vy`.
   7. **Resolve Y** works the same way, with `step = +1` when `vy > 0` and −1
      otherwise.
      - **Floor**: sets `on_ground`, clears knocked, and records ice and snow.
      - **Sloped floor**: does not count as ground; the velocity is projected
        onto the slope and knocked is set.
      - **Ceiling**: `vx *= 0.5`.
      - `vy = 0` unless the player is sliding on a slope.
      - **No collision**: `on_ground = false` and the ice flag clears.
   8. **Guard-tower fix**: if the player touched a slope in X, had no Y collision,
      is falling and is not knocked, then `vx *= −0.5` and knocked is set.
   9. **Gravity**: `vy = min(vy + GRAVITY, MAX_FALL)`.
   10. **Materials**: on ice, `vx` approaches 0 by `ICE_FRICTION`.
2. **Controls** (player state machine), described in its own section below.
3. **Camera**: `target = −floor(centre_y / 360)`. The camera moves only when the
   player moves towards the target, and not while rising slower than 3 px/tick.

The hitbox is `Rectangle((int)x, (int)y, 18, 26)`, which truncates toward zero.
Because of this, a player standing still alternates between grounded and
airborne. Gravity carries the player 0.26 px and then 0.77 px into the air, and
the floor is hit again on the third tick. The controller expects this pattern:
it tolerates up to 2 airborne ticks (`air_charge > 1/30 s` cancels a charge).

## Controls

- **Jump input**: a new press sets `can_jump`. Releasing clears it, and so
  does starting a charge. A press held in the air is buffered until landing.
- **Splat**: landing (`ground && !was_grounded`) with `last_velocity.y == MAX_FALL`
  sets splat and cancels any charge. While splatted, `vx = 0` (except on ice).
  The splat timer accumulates only while grounded. Once it reaches `SPLAT_TIME`,
  any jump or direction input stands the player up.
- **In the air**: no control. An active charge survives 2 airborne ticks, so
  it is not lost to the standing flicker.
- **Direction history**: the last 4 grounded direction inputs. The jump uses
  the most recent non-zero one, so a direction pressed up to 3 ticks before
  release still counts.
- **Charging**: starts on the tick `can_jump` is set. Each tick it sets
  `vx = 0` (unless on ice) and adds `charge += dt`.
- **Release**: happens when `charge ≥ 0.6` or jump is released.
  - `power = min(1, charge / 0.6)`.
  - **On snow**: if `power ≤ 0.2`, no jump happens. If `power < 0.3`, it is
    raised to 0.3.
  - `vy = JUMP × power`; `vx = clamp(vx + dir × 3.5, ±3.5)`.

  Holding for *n* ticks and releasing on tick *n + 1* gives
  `power = (n + 1) / 36`. Holding 35 ticks gives a full jump, as in the TAS.
- **Walking**: grounded and not charging.
  - Normal ground: `vx = dir × 1.5`.
  - Snow: `vx = 0`.
  - Ice: `vx` approaches `dir × 1.5` by 0.2 per tick, and only when
    accelerating or reversing.

## Level format (native and clone)

- **Atlas**: `Content/level.xnb` is an uncompressed XNB5 `Texture2D` (Color). The
  atlas is 780 × 585 px, a grid of 13 × 13 screens, each screen 60 × 45 px with
  one pixel per 8 px tile.
  - The atlas is column-major: screen *i* (0-based) has its top-left corner at
    `(i // 13 × 60, i % 13 × 45)`.
  - World Y of screen *i* is `[−360 i, −360 i + 360)`.
- **Block colours** (base-game subset):

  | RGB | Block | Notes |
  | --- | --- | --- |
  | (0,0,0) | solid | |
  | (255,0,0) | slope | Orientation comes from its neighbours: one solid or fake on the left or right, and one above or below. |
  | (128,128,128) | fake | Non-solid; only counts as a slope neighbour. |
  | (0,255,255) | ice | |
  | (255,255,0) | snow | |
  | (0,255,0) | wind | Screen-level flag. |
  | (255,255,255) | no-wind zone | |
  | (R,0,255) | teleport | Side link to screen R. |

  Water, sand, quark, low gravity, warp and other blocks are DLC or mod content.
  The clone ignores them.
- **Slope orientation** (empty corner is named):

  | Solid neighbours | Orientation | Surface normal |
  | --- | --- | --- |
  | right and below | TopLeft | (−1, −1)/√2 |
  | left and below | TopRight | (1, −1)/√2 |
  | right and above | BottomLeft | (−1, 1)/√2 |
  | left and above | BottomRight | (1, 1)/√2 |

## Base game layout

There are 43 screens.

| Area | Screens |
| --- | --- |
| Redcrown Woods | 1–5 |
| Colossal Drain | 6–10 |
| False King's Keep | 11–14 |
| Bargainburg | 15–19 |
| Great Frontier | 20–25 |
| Windswept Bluff (wind) | 26 |
| Stormwall Pass (wind) | 27–32 |
| Chapel Perilous | 33–36 |
| Blue Ruin | 37–39 |
| The Tower | 40–43 |

The babe waits at the top of screen 43 (around x 375, y 113), and reaching her
ends the game. The player starts at (231, 302) on screen 1.

## Excluded

The clone leaves out all New Babe+ and Ghost of the Babe content, plus:

- **Blocks**: water, sand, quark.
- **Items**: Giant Boots, Snake Ring and the other items.
- **Presentation**: NPCs, ravens and cosmetics.
