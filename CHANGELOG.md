# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **A playthrough to all 48 achievements.** `scripts/achievement_run.py`
  plays from a new profile to every achievement with button inputs only, run
  after run from new games, and saves it
  (`jumpking/data/playthrough_all48.json`); `tests/test_playthrough.py`
  replays it. Speed-run inputs are referenced (`["tas", key, first, last]`,
  `planner.tas_frames`), not copied.
- `jumpking.planner`: `Planner` (BFS routes to goals, played frame by frame in
  the campaign; `tas` and `repeat` actions) and `replay`.
- `solver.bfs(..., cell=(px, v))`: coarse deduplication for landed-mode
  searches; it finds Blue Ruin's ice-momentum route in a few hundred nodes.
- `JumpKingEnv(landed_macros=True)`, `examples/go_explore.py --landed`
  (landed macros, cells keyed by speed): Go-Explore now beats the original
  game's normal ending (18 min) instead of stalling on screen 38.
- `tas.macro_to_inputs(..., landed=True)`.
- `physics.P_TOUCH` / `EV_TOUCH`: an optional planning rectangle that raises
  an event while the hitbox overlaps it, and `solver.bfs(..., touch=rect)`:
  items are picked up in mid-air, so a search can now aim for a jump that
  passes through one (some mushrooms sit in alcoves you cannot stand in).
- `Planner.stays(goal)`: a landing only reaches a screen if the king, left
  alone, comes to rest there (with the Snake Ring and the Giant Boots' landing
  freeze some landings slide straight off the ledge).

### Fixed
- Macro jumps in water: `charge` now means power `charge / 36` there too
  (twice as many ticks of charging); it used to cap water jumps at half power,
  which made Deep Ruin's screen 90 impossible for the macros.
- Macros end in quicksand: `is_settled` / `is_landed` count sinking in sand
  as ground, as the game's `IsOnGround` node does (a jump can be charged
  there). Before, every macro into Halted Ruin's sand timed out.

## [0.3.0] - 2026-10-13

### Added
- **The whole world.** `Level.original()` loads all 169 screens of the game's
  atlas: the base game, the Imp Room, New Babe+, Ghost of the Babe,
  Philosopher's Forest and the Hole, with every side teleport and all three
  babes (`level.endings`). `content="base"` keeps the old base-game view.
- **DLC physics**, ported from the game's decompiled code: water, sand
  (quicksand), quark blocks, coloured wind variants, the Giant Boots and the
  Snake Ring.
- **The campaign** (`jumpking.campaign`): new games in the three modes,
  world items, inventory and outfits, NPC dialogue, merchants, birds,
  achievement hitboxes and illusory walls, statistics, the endings' rewards
  and a persistent `Profile`. All 48 Steam achievements are implemented
  (`jumpking.achievements`, `jumpking achievements`).
- `jumpking.bt` (the game's behaviour-tree library), `jumpking.content` (the
  game's content files, read from your install), `jumpking.routes` (the three
  stories' routes) and `jumpking.verify` (TAS parity).
- Gymnasium: the `ending` option and the `route` reward, four new vector
  features (water, sand, boots, ring), `JumpKingCampaign-v0` and
  `JumpKingCampaignMacro-v0`.
- `jumpking play --campaign`, `jumpking achievements`.
- Parity: the New Babe+ and Ghost of the Babe TAS routes reproduce tick for
  tick and win on the exact recorded timer, like the base-game route.

### Changed
- The physics kernel follows the game's behaviour pipeline exactly. This
  covers the per-screen collision merge, the slope triangle test, tangential
  slides, the early exit when starting inside a wall, the camera's forced
  follow on the ground, the win check at the start of the frame, the jump
  direction buffer reset, and the splat check on every grounded tick.
- Slopes are oriented on the stacked world with screen edges counting as
  solid; a slope without orientation never collides (it used to become solid).
- The normal ending uses the game's condition (`centre x ≥ 336` on screen 43,
  on the ground) instead of a calibrated rectangle; the built-in map keeps its
  rectangle goal.
- Side teleports truncate y to an integer, as today's game does
  (`legacy_teleport=True` restores the behaviour of the 2019 builds).
- `dnfile` is now a runtime dependency (it reads the game's dialogue text).
- The state vector has 46 slots and the parameter vector 23.

## [0.2.0] - 2026-10-13

### Added
- Original-map support: the base game is read from your own Steam install
  (screens 1–43 plus the hidden Imp Room). New Babe+, Ghost of the Babe and
  Philosopher's Forest are excluded.
- `tests/test_original.py`: replays the community base-game TAS (3:44.859)
  tick for tick on the real map and checks constants against `JumpKing.exe`.
- `scripts/extract_constants.py`: reads `PlayerValues` from the game's IL.
- Natively batched simulator (`BatchSim`), `JumpKingVectorEnv`, and an SB3
  `VecEnv` adapter (~640k steps/s).

### Changed
- `SPLAT_TIME` is now 0.5 s, taken from the game executable.
- The default level is `auto`: the original map if installed, otherwise the
  built-in map.
- The babe trigger is calibrated from the TAS ending.

## [0.1.0] - 2026-10-13

### Added
- Physics kernel (numba JIT with a pure-Python fallback), `Game` API and macro
  actions.
- Native hitbox-atlas and `level.xnb` reader, with Steam install discovery.
- Built-in 43-screen map, generated and verified by the physics engine.
- Gymnasium environments, a pygame viewer, `.tas` replays, a BFS solver, and
  PPO, evaluation and Go-Explore scripts.
