# Plan

## Goals

1. **Faithful mechanics.** Port the native per-tick pipeline exactly as listed in
   `RESEARCH.md`, using float32 arithmetic like the original.
2. **Deterministic headless core.** The simulation has no rendering or wall-clock
   dependency. The whole state is a small array that can be snapshotted and
   restored in O(1), which suits search, Go-Explore and resets mid-level.
3. **Fast.** The core tick is written in a numba-compatible style and JIT
   compiled when numba is available, with a pure-Python fallback.
4. **Level data.**
   - The built-in 43-screen base-game-style map is stored in the native
     780 × 585 hitbox atlas format (PNG).
   - Optionally, the original map can be loaded from the user's own Steam
     install (`Content/level.xnb`), which gives exact original geometry.
5. **RL interface.**
   - Gymnasium env with:
     - frame-level actions, or macro actions (direction × charge ticks, plus walk);
     - vector, tile-grid, RGB or dict observations;
     - shaped rewards with pluggable reward functions;
     - curriculum starts on any screen;
     - `get_state` / `set_state`.
   - Works with Gymnasium vector envs.
   - Training (PPO) and evaluation scripts.
6. **Human play and replays.**
   - pygame front end with keyboard play and debug overlay.
   - Replays in the JumpKingTAS `.tas` text format.
7. **Verification.**
   - Unit tests for each mechanic.
   - A determinism test.
   - A reachability solver (BFS over landings) that proves every built-in screen
     can be completed and produces expert demonstrations.

## Layout

```
jumpking/
  constants.py      researched constants (float32)
  physics.py        tick function (numba / python)
  level.py          Level: tile grid, slopes, wind, teleports, atlas IO
  xnb.py            native level.xnb reader + Steam install discovery
  game.py           Game: high-level API (step, snapshot, macro actions)
  mapgen.py         generator for the built-in map (area themes) + validation
  solver.py         landing-graph BFS solver
  render.py         numpy software renderer (rgb observations)
  play.py           pygame human play / replay viewer
  tas.py            .tas replay format
  envs/             Gymnasium environments + reward functions
  data/level.png    built-in map atlas
scripts/            train_ppo.py, evaluate.py, benchmark.py, build_map.py
tests/
```

## Status

All goals are implemented and committed.

- **Physics port**:
  - per-tick pipeline, float32 numerics;
  - grounded +1 px probe;
  - wind latch and waveform on the 17 ms clock;
  - slopes, ice, snow, splat, camera and teleports.
- **Levels**:
  - native atlas IO;
  - `level.xnb` reader with Steam discovery;
  - built-in 43-screen map, generated and verified by the engine.
- **RL interface**:
  - `JumpKingEnv` (macro or frame actions; vector, grid, rgb or dict
    observations; rewards; curriculum; state access);
  - natively batched `JumpKingVectorEnv` with an SB3 adapter;
  - PPO training and evaluation scripts;
  - Go-Explore example.
- **Tools**:
  - pygame play and replay viewer;
  - `.tas` recordings;
  - BFS solver;
  - CLI;
  - original-map parity validator (replays the base-game TAS);
  - constant extractor for the user's own `JumpKing.exe`.
- **Tests**: pytest suite of 55 tests.

Open items need the original game files, which are currently not installed on
this machine:

- Run `scripts/validate_original.py --download` for tick-level parity on the
  real map.
- Run `scripts/extract_constants.py --write` to replace the estimated
  `SPLAT_TIME`.
