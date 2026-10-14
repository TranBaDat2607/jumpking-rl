# Jump King: researched mechanics (base game, New Babe+, Ghost of the Babe)

These notes record what the clone reproduces and where each value comes from.
Everything is per game tick. The game runs a fixed 60 Hz update (physics
`FrameDelta` = 1/60 s; the game clock `TargetElapsedTime` is 17 ms and drives
the wind and the in-game timer). Coordinates are in pixels with Y pointing down.

The authoritative source is the game itself: `JumpKing.exe` of the Steam build
(app 1061090), decompiled with ILSpy (S8), plus the content files the game
loads at runtime (S10). Earlier mod sources (S1-S5) agree with it wherever they
overlap. Three real-game speed-runs (S6, S9) are replayed tick for tick as the
final check.

## Sources

| Ref | Source | Used for |
| --- | --- | --- |
| S1 | `qequ/jump-king-parabole` (mod; constants read from `JumpKing.PlayerValues`) | gravity, jump velocity, speed, max fall, charge time, hitbox 18×26 |
| S2 | `anyalink99/jump-king-mods`: `screen-solver/src/NativePlayer.cs`, `NativeResolution.cs`, `NativeWorld.cs`, `jk-runtime/src/Simulation/*` (a parity-tested port of the native body pipeline and controller) | first version of the update order, collision resolution, slopes, wind latch, splat, charge, direction buffer, camera, teleports |
| S3 | `mega-gameplay-expansion/docs/no-walk-off.md` (same repo) | walk speed ±1.5, ice friction 0.1 (and ×2 while walking) |
| S4 | `jk-runtime/src/Simulation/NativeWind.cs` | wind waveform and strength |
| S5 | Official workshop docs (`TeamNexile/jk-workshop-docs`), JumpKingPlus hitbox docs | hitbox atlas format, block colours |
| S6 | `ShootMe/JumpKingTAS` `LevelFiles/NewGame.tas` | start position (231, 302), input timing, base-game parity |
| S7 | Jump King wiki / PSNProfiles / Namu wiki | area names and screen counts |
| S8 | `JumpKing.exe` (Steam build), decompiled with ILSpy 8.2 (`ilspycmd -p`, and `--ilcode` where the decompiled C# needed checking). Classes cited below by name: `BodyComp`, `JumpKing.BodyCompBehaviours.*`, `JumpKing.BlockBehaviours.*`, `LevelManager`, `LevelScreen`, `BaseBlockFactory`, `SlopeBlock`, `LineUtil`, `Camera`, `PlayerEntity`, `JumpState`, `FailState`, `Walk`, `IsOnGround`, `WindManager`, `EndingManager`, `NormalEnding`, `NewBabePlusEnding`, `OwlEnding`, `NBPDecisionState` | every DLC mechanic, the exact collision merge, camera rules, endings |
| S9 | `ShootMe/JumpKingTAS` `LevelFiles/NewBabe.tas` and `NewGhost.tas` | New Babe+ and Ghost of the Babe parity |
| S10 | The game's content files: `Content/level.xnb`, `Content/gui/location_settings.xml`, `Content/props/**` (achievement hitboxes, hidden walls, world items, NPC and merchant settings, raven paths), `Content/king/skin_settings.xml` | the 169-screen world, area names, item and NPC data (read at runtime, never committed) |

## Constants

| Name | Value | Source |
| --- | --- | --- |
| Hitbox | 18 × 26 px | S1, S8 `PlayerValues` |
| `GRAVITY` | 0.2571429 px/tick² | S1 |
| `JUMP` | −8.742858 px/tick (scaled by charge power) | S1 |
| `SPEED` (jump horizontal) | 3.5 px/tick | S1 |
| `MAX_FALL` | 10 px/tick | S1 |
| `JUMP_TIME` | 0.6 s (36 ticks) | S1, S6 |
| `WALK_SPEED` | 1.5 px/tick | S2, S3 |
| `BOUNCE` | 0.5 | S2, S8 |
| `ICE_FRICTION` | 0.1 px/tick² | S3, S8 |
| Wind | `0.1 × wave(t)` px/tick² added to vx | S4, S8 `WindManager` |
| Wind variants | `0.0125 × intensity`, see below | S8 `WindManager`, `LevelManager.LoadBlocksInterval` |
| `SPLAT_TIME` | 0.5 s of grounded time | S8 `PlayerValues.get_SPLAT_TIME` |
| `SAND_FALL_SPEED` | 0.75 px/tick | S8 `PlayerValues`, `SandBlockBehaviour` |
| `SAND_MOVE_MULTIPLIER` | 0.25 | S8 |
| `SAND_JUMP_MULTIPLIER` | 0.5 | S8 |
| Water multiplier | 0.5 (movement, gravity, jump charge) | S8 `WaterBlockBehaviour.GetWaterMultiplier` |
| Giant Boots landing pause | 0.05 s body freeze, then 0.15 s shake | S8 `PlayerEntity.MakeBT` |
| Screen | 480 × 360 px = 60 × 45 tiles of 8 px | S5 |

## Per-tick order

A frame runs, in order (S8 `JumpGame.Update`, `EntityManager.Update`):

1. **GameLoop** (behaviour tree): the achievement clock ticks, then
   `EndingManager.CheckWin` tests the state left by the *previous* frame
   (see Endings). A win stops the frame.
2. **Entities**, in reverse creation order: the run-wide managers, then the
   player, then props, world items, merchants, NPCs and ravens (those see the
   player's new position).
3. The **player** entity updates its components: `BodyComp` (physics),
   `InputComponent`, the behaviour tree (controls) and `CameraFollowComp`.

### BodyComp

The body runs a list of behaviours (S8 `BodyComp.SetupBaseBehaviours`):

1. **Cache collision state**: the sand flag is "the hitbox overlaps sand" and
   the water flag is "the hitbox overlaps water", both over the camera screen
   and its two neighbours.
2. **Wind**: if the camera screen has no wind, clear the wind latch. Otherwise
   set the latch when the player is on the ground or has just fallen in from a
   higher screen (`last_screen > screen`). If latched, not on snow (see the
   Snake Ring) and not inside a no-wind block of the camera screen,
   `vx += wind(t)`. Then `last_screen = screen`.
3. `last_velocity = velocity`.
4. **Early exit**: if the hitbox already overlaps a blocking block,
   `position += velocity` and the rest of the body update is skipped (no
   gravity).
5. **X move**: `x += vx`, scaled by 0.25 inside sand and by 0.5 in water.
6. **Cap / teleport**: screens without side teleports clamp
   `x ∈ [−9, 471]`. Screens with teleports move the player by ±480 px once the
   hitbox centre leaves `[0, 480]` and shift y to the linked screen:
   `y = (int)y − 360 × (target − (int)((360 − y) / 360))`, then the camera
   jumps to the new screen. A pixel in the left half of the screen's atlas
   cell sets the left exit's link, one in the right half the right exit's; a
   screen with a single link uses it for both exits.
7. **Resolve X**: if the hitbox overlaps a blocking block (or sand, when the
   player was not already in sand), truncate `x` to an integer and step back one
   pixel at a time against the motion until clear. A flat wall gives
   `vx *= −0.5` and the knocked state. A slope gives `vx = 0` when walking slowly
   into it on the ground, otherwise the velocity is replaced by its tangential
   component (`MakeComposite(v, n).i`) when `last_velocity · n < 0`.
8. **Y move**: `y += vy`. Upward motion inside sand is halved; outside sand a
   grounded king moving down gets `y += 1` first (the probe that keeps him
   grounded); in water the move is halved.
9. **Resolve Y** works the same way, with `step = +1` when `vy > 0` and −1
   otherwise (sand blocks only upward motion from outside).
   - **Floor**: sets `on_ground`, clears knocked, and records ice and snow.
   - **Sloped floor**: does not count as ground; with `last_velocity · n < 0`
     the velocity becomes tangential and knocked is set, otherwise the contact
     behaves as a floor except that `on_ground` stays false.
   - **Ceiling**: `vx *= 0.5`; a sloped ceiling also sets `vy` to the tangential
     component.
   - `vy = 0` unless the contact is a slope with `vy > 0` and no ground.
   - **No collision**: `on_ground = false`, the ice flag clears, and a fall at
     `MAX_FALL` through a quark block snaps `y` up to the 8 px grid
     (`ceil(y / 8) × 8`; `vx = 0` if the x position was capped this tick).
10. **Gravity**: `vy = min(vy + g, MAX_FALL)` with `g` halved in water.
11. **Blocks**: on ice (or on any ground with the Snake Ring) `vx` approaches 0 by
    `ICE_FRICTION`. Inside sand `vy = min(vy, 0.75)` and knocked clears.
12. **Guard-tower fix**: if the player touched a slope in X, had no Y collision,
    is falling and is not knocked, then `vx *= −0.5` and knocked is set.

All arithmetic is float32 (`f32` in the kernel). The hitbox is
`Rectangle((int)x, (int)y, 18, 26)`, which truncates toward zero: above screen 1
(negative y) a fractional y rounds the hitbox *down*.

### Collision queries

`LevelManager.CheckCollision` tests only the camera screen and its two
neighbours, each with `LevelScreen.TryCollision`, then merges the results
(`CheckCollisionInternal`): if the camera screen collides, colliding
neighbours add their blocks and may override the slope *type* (but not the
slope normal); if it does not, the first colliding neighbour becomes the
baseline and the camera screen adds its blocks except water and quark. The
clone reproduces this merge exactly (`physics.check_collision`), including
the odd cases on screen seams.

Within a screen, the slope cell closest in x to the hitbox centre gives the
slope type and normal. If a box is at least as high as the highest touched
slope and that slope faces upwards, the contact counts as a flat box contact
(the normal stays set).

Slope cells collide through `SlopeBlock.Intersects`: triangle-edge against
hitbox-edge crossings (`LineUtil`, strict spans, integer slopes), triangle
vertices and edge midpoints inside the hitbox. The clone ports this test
exactly. The game's `BottomLeft` triangle has the same corners as
`BottomRight` (an upstream quirk the port keeps); `TopLeft`/`TopRight` match the
old closed-form test.

Quark blocks register only while the hitbox bottom is above the block's
bottom. Water, sand, quark and no-wind blocks never block.

### Camera

`Camera.UpdateCameraWithVelocity` with target `−floor(centre.y / 360)`:

- On the ground (not in sand) the camera always follows (forced update).
- In the air it moves only towards the player's motion, and not while rising
  slower than 3 px/tick.
- Inside sand it does not move.
- It never goes past the last screen, and it never shows the screen right above
  a babe's screen (44, 101, 155): it stays one screen lower.

## Controls

The player's behaviour tree (S8 `PlayerEntity.MakeBT`) runs one of two
branches each tick.

**Grounded branch** (`IsOnGround`: on the ground *or* inside sand), in
parallel:

- *FailState* (splat). Whenever it is run fresh it checks
  `last_velocity.y == MAX_FALL`; a hit starts a splat (counted as a fall) and
  cancels any charge. While splatted, `vx = 0` (except on ice); the timer
  accumulates only while on the ground; once it reaches `SPLAT_TIME`, any
  jump or direction input stands the player up (that tick does nothing else).
  A splat interrupted by an airborne tick restarts its timer and counts as a
  new fall on landing.
- otherwise *JumpState*: a new press sets `can_jump`; releasing clears it.
  Whenever the jump state starts fresh it clears its timer and its 4-entry
  direction buffer. Charging starts on the tick `can_jump` is consumed: `vx = 0`
  (unless on ice), `timer += dt` (`dt × 0.5` in water). Release happens at
  `timer ≥ 0.6` or when jump is released: `power = min(1, timer / 0.6)`; on snow
  `power ≤ 0.2` does nothing and `power < 0.3` becomes 0.3; `vy = JUMP × power`;
  `vx = clamp(vx + dir × 3.5, ±3.5)` with `dir` the current input or, if 0, the
  most recent non-zero buffered input of the charge.
- otherwise *Walk*: nothing when moving up inside sand; on snow `vx = 0`; on
  ice `vx` approaches `dir × 1.5` by 0.2 (only when accelerating or reversing);
  otherwise `vx = dir × 1.5`; with the Giant Boots the target is 0.
- *Giant Boots landing*: with the boots on, the first grounded tick disables
  the body for a 0.05 s pause (the next two physics updates are skipped), then
  a 0.15 s screen shake runs; it restarts after every airborne tick. A jump
  re-enables the body at once.

**Airborne branch**: a charge that was running keeps running for the 1/30 s
leniency pause (2 ticks, so a jump can still be released just after walking
off a ledge); after that the jump state is no longer run and restarts on
landing.

Holding for *n* ticks and releasing on tick *n + 1* gives `power = (n + 1) / 36`.

## Items with physical effects

| Item | Effect | Source |
| --- | --- | --- |
| Giant Boots | walking disabled; landing freeze (above) | S8 `Walk`, `PlayerEntity.MakeBT` |
| Snake Ring (the "ice ring") | ice behaviour on any ground (friction, ice walking, no `vx = 0` when charging or splatting); snow has no effect (no weak-jump rule, walking works, wind applies) | S8 `IceBlockBehaviour.IsPlayerOnBlock`, `SnowBlockBehaviour.IsPlayerOnBlock` |

All other items (crowns, shoes, cap, gnome hat, tunic, yellow shoes, owl cape,
rings, coins, fragments, mushrooms, the bug note) are cosmetic or currency.

## Level format (native and clone)

- **Atlas**: `Content/level.xnb` is an uncompressed XNB5 `Texture2D` (Color). The
  atlas is 780 × 585 px, a grid of 13 × 13 screens, each screen 60 × 45 px with
  one pixel per 8 px tile. All 169 cells are loaded (`LevelTexture`).
  - The atlas is column-major: screen *i* (0-based) has its top-left corner at
    `(i // 13 × 60, i % 13 × 45)`.
  - World Y of screen *i* is `[−360 i, −360 i + 360)`; the screens of all three
    stories are stacked into one world.
- **Block colours** (`BaseBlockFactory`, `LevelManager.LoadBlocksInterval`):

  | RGB | Block | Notes |
  | --- | --- | --- |
  | (0,0,0) | solid | |
  | (255,0,0) | slope | Oriented by its neighbours (below). |
  | (128,128,128) | fake | Non-solid; only counts as a slope neighbour. |
  | (0,255,255) | ice | |
  | (255,255,0) | snow | |
  | (0,170,170) | water | Non-solid. |
  | (255,106,0) | sand | Non-solid quicksand; counts as solid for slope orientation. |
  | (182,255,0) | quark | Non-solid; snaps max-speed falls to the 8 px grid. |
  | (255,255,255) | no-wind zone | Non-solid. |
  | (0,255,0) | wind | Screen-level flag. |
  | (R,255,0), R 192..239 | wind variant | R 192-207: oscillating with intensity R−191; 208-223: constant leftwards, R−207; 224-239: constant rightwards, R−223; force `0.0125 × intensity`. Not used by the official world. |
  | (R,0,255) | teleport | Side link to screen R (left/right half of the cell = left/right exit). |

- **Slope orientation** (`BaseBlockFactory.GetSlopeType`; empty corner named):

  | Solid neighbours | Orientation | Surface normal |
  | --- | --- | --- |
  | right and below | TopLeft | (−1, −1)/√2 |
  | left and below | TopRight | (1, −1)/√2 |
  | right and above | BottomLeft | (−1, 1)/√2 |
  | left and above | BottomRight | (1, 1)/√2 |

  Neighbours are read on the stacked world, so the screen above or below
  counts, and columns outside the screen count as solid
  (`LevelManager.IsBlockEdge`). Solid, slope, ice, snow, sand and fake orient
  slopes. A slope with no valid pair becomes a block with no orientation
  that never collides (one exists, on screen 112).

## The world

| Story | Screens | Areas |
| --- | --- | --- |
| Base game | 1–43 | Redcrown Woods 1–5, Colossal Drain 6–10, False King's Keep 11–14, Bargainburg 15–19, Great Frontier 20–25, Windswept Bluff 26 (wind), Stormwall Pass 27–32 (wind), Chapel Perilous 33–36, Blue Ruin 37–39, The Tower 40–43 |
| Secret | 45, 46 | the Imp Room (45), the New Babe+ entrance (46) |
| New Babe+ | 47–100 | Brightcrown Woods 47–52, Colossal Dungeon 53–59, False King's Castle 60–63, Underburg 64–70 (wind), Lost Frontier 71–77, Hidden Kingdom 78–83, Black Sanctum 84–89, Deep Ruin 90–94 (water), The Dark Tower 95–100 |
| Ghost of the Babe | 102–154 | Bog 102–108 (water, snow), Moulding Manor 109–116, Bugstalk 117–123 (ice), House of Nine Lives 124–130, The Phantom Tower 131–139, Halted Ruin 140–147 (water), Tower of Antumbra 148–154 (sand) |
| Philosopher's Forest | 156–164 | the passage from Redcrown Woods (156), the forest (157–160), the Hole (164 down to 161) |

Side teleports: 3 ↔ 156, 18 ↔ 45, 46 ↔ 47, 160 ↔ 164, 161 ↔ 108.

Routes (the order the community TAS routes take, `jumpking/routes.py`):

- **Normal ending**: screens 1–43, the babe on screen 43.
- **New Babe+**: Bargainburg 4 (18) → Imp Room (45) → 46 → 47–100, the babe
  on screen 100. A new New Babe+ game starts on screen 1; if the player owns
  the crown, the game first offers to sacrifice it and start in the Imp Room
  at (171, −15570) (`NBPDecisionState.TeleportToImpRoom`).
- **Ghost of the Babe**: Redcrown Woods 3 → 156 → Philosopher's Forest
  157–160 → the top of the Hole (164) → down to 161 → the Bog (108) → down to
  102 → up 103–154, the babe on screen 154.

### Endings

`EndingManager.CheckWin` checks all three babes every frame, before the
entities update. The king must be on the ground with the camera on the babe's
screen; `P = (240, 180) + INSIDE_SCREEN_POSITION + (0, −360 × screen)`:

| Ending | Screen | Condition on the hitbox centre |
| --- | --- | --- |
| Normal (`NormalEnding`) | 43 | `x ≥ 336` |
| New Babe+ (`NewBabePlusEnding`) | 100 | `x ≤ 144` and `y ≤ −35 495` |
| Ghost of the Babe (`OwlEnding`) | 154 | `x ≥ 240` and `y ≤ −54 935` |

## Runs, items, NPCs and achievements (the campaign)

`jumpking/campaign.py` reproduces the game around the physics; the logic is
ported from the classes named below (S8) and fed with the content files of
the install (S10). The behaviour-tree library those entities are built from
(`BehaviorTree.*`) is ported literally in `jumpking/bt.py`, because its
semantics set the timing. A node that did not run on the previous tick
restarts, sequences and running-selectors resume at their running child, and
timers accumulate the 1/60 s frame delta in float32.

### Frame order

`GameLoop` first advances the achievement clock (`AchievementManager.Update`,
which also checks the outfit achievements every frame), then the win check.
The entities then update in reverse creation order: items spawned during the
run, the `FullRunManager`, the player, the illusory-wall and achievement
triggers, the world items, the merchants, the NPCs and the birds
(`IntroState.InitializeEntities`, `PropManager.Load`, `EntityManager.Update`).

### Saves: what a new game resets

- **Permanent** (`SavesPerma`, `campaign.Profile`): the inventory, the equip
  options and worn skins, story flags (`CompletedNormalGame`, `CompletedNBP`,
  `CompletedGhost`, `StartedNBP`, `StartedGhost`), lifetime statistics (jumps,
  falls = splats, frames, attempts, wins) with the snapshot taken at the start
  of the attempt, and the Steam achievements.
- **Per run** (`CombinedSaveFile`, `campaign.RunState`): NPC states, merchant
  states, the birds' perches, the world items still lying around, and
  `FullRunSave` (whether the Giant Boots / Snake Ring were worn since screen 1).

### New games

The title menu offers New Game, New Babe+ and Ghost of the Babe. All three
start at (231, 302) on screen 1. New Game and Ghost of the Babe move the white
raven to perch 17 on screen 48 (`NBPDecisionState.MoveWhiteBird`). New Babe+
with the crown in the inventory asks whether to sacrifice it. Sacrificing
removes the crown and starts in the Imp Room at (171, −15570); keeping it
starts on screen 1. Wearing the New Babe+ crown starts in the Imp Room for free.

### Items and skins

World items (`Content/props/worlditems/worlditems.xml`) are picked up when the
hitbox touches the item's sprite rectangle (anchored at its bottom centre).
Each pickup adds one to the inventory and puts the item on if it is
wearable. Skins (`Content/king/skin_settings.xml`) occupy layers: Hat (crowns,
cap, gnome hat), Boots (shoes, Giant Boots, yellow shoes), Shirt (tunic),
SnakeRing and Cape. Putting one on takes off whatever shares its layer. The
Giant Boots and the shop's yellow shoes are only displayed in the world; they
are sold.

| Item | Where |
| --- | --- |
| Gold ring | dropped by the raven at its last perch (screen 32), once per run |
| Ruby | dropped by the white raven at its last perch (screen 49) |
| Silver coins (10) | New Babe+: screens 52, 53, 54, 56, 60, 68, 81, 91, 93, 94 |
| Mushrooms (9) | Philosopher's Forest: screens 157–160 |
| Ghost fragments (4) | screens 108, 111, 113, 115 |
| Gnome hat | screen 160 |
| Cap | screen 6 |
| Babe ghost | screen 108 (picked up anywhere on screen 106) |
| Crown / New Babe+ crown / owl crown | the three endings |
| Owl cape | the Ghost of the Babe ending, if the red bird was chased off all its perches |

### NPCs (`OldManEntity`, `OldManComp`, `OldManState`)

An NPC talks when the camera is on its home screen and the king touches its
trigger box. It then waits 0.5 s, checks the box again, and waits the rest of
its `talk_delay`. It takes a quote: first a story-flag quote, then the intro,
then the most recent quote of its pool, and once the pool is empty its
looping lines, if it has any. It says the quote line by line, one character
per `talking_speed` (default 0.05 s) with a 2.2 s pause after each line, then
it is busy for 30 s. The pool grows when the camera reaches a screen beyond
the best one seen this run that has quotes for that NPC. Talk achievements
compare the run's talk count with a fixed number when a conversation ends:
old man 10 and 40, hermit 10, Bargainburg merchant 4, the outsider cat 1,
the gargoyles 1.

### Merchants (`MerchantEntity`, `MerchantComp`)

| Merchant | Screen | Sells | Price (fixed when the run starts) | Achievements |
| --- | --- | --- | --- | --- |
| merchant | 15 | Shoes | gold rings owned + 1 | ≥ 1: `PURCHASE_SHOES`, ≥ 10: `PURCHASE_FOR_10` |
| merchant_nbp | 71 | Giant Boots | silver owned + 10 | ≥ 10: `BUY_GIANTBOOTS` |
| snake | 88 | Snake Ring | rubies owned + 1 | ≥ 1: `BUY_ICERING` |
| owl_mold_man | 109 | Yellow shoes | ghost fragments owned + 4 | ≥ 4: `PURCHASE_YELLOW_SHOES` |
| owl_shroom_guy | 160 | Tunic | mushrooms owned + 9 | ≥ 9: `PURCHASE_SCHOLAR_TUNIC` |

On first contact a merchant introduces itself. If the king still touches it
afterwards and can pay, it sells: it takes the price, gives the item and puts
it on.

### Birds (`RavenEntity`, `RavenComp`, `RavenFlee`)

Each bird sits on a perch from its `.ravset` path. When the king's hitbox
centre comes within the perch's trigger distance (40 px by default), the bird
flies off. It flies up (or down and sideways when its next perch is on a lower
screen), accelerating every second frame, until it leaves the camera's view.
It then sits on its next perch. After the last perch it is gone.

| Bird | Perches | Reward |
| --- | --- | --- |
| raven | 14, screens 3 → 32 | gold ring at the last perch; `CHASE_BIRD_PAST_SNOWSTORM` on reaching it; `WIN_TOUCH_CROW_2` if it fled exactly twice when the normal babe is reached |
| white raven | 52, screens 1 → 18, then 48 → 88 → 49 | ruby at the last perch |
| red bird (tsuchinoko) | 17, Philosopher's Forest → screen 147 | the owl cape at the Ghost of the Babe ending (`GET_OWLCAPE`) |
| fly | 39, screens 117–119 | `CHASE_AWAY_BUG` when it leaves its last perch |

### Achievements

The 48 codes are the game's `AchievementCode` enum; `jumpking/achievements.py`
lists each with its store name and exact rule (`jumpking achievements` prints
them). Highlights of the rules:

- Falls are splats. The 100th and 1000th splat are checked with equality.
  Jumps are checked with `≥ 1000`, `≥ 10000` and `≥ 20000` (the enum calls the
  last one `JUMP_100000`).
- Win checks run at the moment of victory, on the run's statistics (frames
  since the run began × 17 ms; splats; jumps). The outfit achievements are
  checked every frame.
- The Giant Leaps and Absolute Will achievements need the item worn on
  screen 1 and never taken off since (`FullRunManager`). A New Babe+ run that
  starts in the Imp Room only counts after the king walks back to screen 1.
- Hitbox achievements: Chapel Perilous (screen 33), the Imp Room (45), the
  nose (screen 71, a 13 × 8 box), Deep Ruin's sea (top 160 px of screen 90),
  and illusory walls on screens 6, 7, 8 and 22.

Store names and descriptions come from Steam's community stats page and
GameFAQs' achievement list. The order of the latter matches the enum, which
pins the code ↔ name mapping (SteamDB, which lists the same data, rejects
automated requests).

## Verification against the installed game

With the game installed from Steam (app 1061090), I checked the clone against
the real files.

- **Constants.** `PlayerValues` in `JumpKing.exe` defines every value as a
  multiple of `FPS_SCALE = 0.5`:

  | Constant | Value |
  | --- | --- |
  | `WALK_SPEED` | 3 × 0.5 |
  | `SPEED` | 7 × 0.5 |
  | `MAX_FALL` | 20 × 0.5 |
  | `ICE_FRICTION` | 0.2 × 0.5 |
  | `BOUNCE` | 0.5 |
  | `JUMP_TIME` | 0.6 |
  | `SPLAT_TIME` | 0.5 |
  | hitbox | 18 × 26 |

  `GRAVITY` and `JUMP` are derived from a jump height. `MAX_SLIDE` exists but
  nothing in the game uses it.
- **TAS parity.** All three community routes replay tick for tick on the
  whole original world, visit every section's screen and reach their babe on
  the exact frame of the recorded in-game timer
  (`scripts/validate_original.py --download`, `tests/test_original.py`):

  | Route | Sections | Ending | Timer |
  | --- | --- | --- | --- |
  | `NewGame.tas` | 43 / 43 | normal | 3:44.859 |
  | `NewBabe.tas` | 56 / 56 | New Babe+ | 5:20.637 |
  | `NewGhost.tas` | 71 / 71 | Ghost of the Babe | 5:35.138 |

  The same three routes also replay inside the full campaign (all NPCs,
  birds and triggers running) and unlock what a player of those runs gets,
  e.g. Fool me Twice... (the speed-run scares the raven exactly twice) and
  Nobody's Perfect (the Ghost route has exactly two falls)
  (`tests/test_campaign.py`).

  Details of the TAS tool and of the 2019-2020 game builds the routes were
  recorded on:
  - the `@` line lasts one frame;
  - the `X` reset leaves the run clock at −2 ticks for `NewGame.tas` and at 0
    for the two DLC routes;
  - in those builds a side teleport kept the fractional y. Today's
    `HandlePlayerTeleportBehaviour` truncates it (`conv.i4` in the IL), which
    the clone does by default; `legacy_teleport=True` restores the old
    behaviour. Only `NewGhost.tas` crosses a teleport at a fractional y
    (screen 3 → 156); with today's truncation its hops land one tick earlier
    and the route fails at Philosopher's Forest.
