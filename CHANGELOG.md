# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

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
