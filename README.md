<div align="center">

# jumpking-rl

**A tick-exact *Jump King* simulator for reinforcement learning: the base game, New Babe+, Ghost of the Babe and all 48 Steam achievements.**

Headless, deterministic and about 45,000× faster than real time, with
Gymnasium environments, batched simulation and O(1) save/restore. It is
**verified frame-for-frame against the original game** on all three stories.

[![CI](https://github.com/TranBaDat2607/jumpking-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/TranBaDat2607/jumpking-rl/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.x-0081A5)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-111%20passing-brightgreen)](tests)

<img src="docs/images/demo.gif" width="480" alt="The king climbing Redcrown Woods">

[Quick start](#quick-start) •
[The whole game](#the-whole-game-three-stories-and-48-achievements) •
[Environments](#environments) •
[Fidelity](#fidelity-verified-against-the-real-game) •
[Benchmarks](#performance) •
[Docs](docs/RL_GUIDE.md)

</div>

---

*Jump King* is a notoriously hard platformer. You charge a jump, release it,
and live with the consequences: one bad jump can undo ten minutes of
progress. That makes it a natural **hard-exploration benchmark**, but the real
game cannot be run headless, reset to a chosen state or stepped
deterministically.

**jumpking-rl** reimplements the game tick by tick, in float32 exactly like
the original, and wraps it in a research-friendly API. The physics and the
game logic are ported from the game's own (decompiled) code.

## Highlights

- **Verified fidelity on all three stories.** The community speed-runs of the
  base game, New Babe+ and Ghost of the Babe replay tick for tick on the real
  world: every screen section is reproduced and each run reaches its babe on
  the exact frame of the recorded timer (3:44.859, 5:20.637, 5:35.138).
- **The whole world.** All 169 screens of the game's map in one world: the
  base game, the Imp Room, New Babe+, Ghost of the Babe, Philosopher's
  Forest and the Hole. Every side passage and all three babes are included,
  and so are the DLC mechanics: water, quicksand, quark blocks, wind with
  no-wind zones, and the Giant Boots and the Snake Ring.
- **The whole game.** `Campaign` runs everything around the physics. NPCs and
  merchants have their real dialogue timing and prices. The four birds flee
  along their paths, and items, the inventory, outfits, new games in each
  mode, the endings' rewards and a persistent profile all work as in the
  game. **All 48 Steam achievements can be unlocked.**
- **Fast.**

  | Workload | Rate |
  | --- | --- |
  | Game ticks, one core | 2.75 M/s |
  | Batched macro steps, 512 kings, all cores | ~440 k/s |
  | Full campaign (NPCs, birds, items), one core | ~15 k frames/s |

- **The real map from your own copy.** The original level and content are read
  at runtime from your Steam install. No game assets are distributed. Without
  the game, a built-in 43-screen map is used instead; it is verified solvable.
- **Gymnasium-native.**
  - Macro actions (a whole jump per step) or per-frame button presses.
  - Vector, tile-grid, RGB or dict observations.
  - A choice of ending, route-aware rewards and curriculum starts.
  - A campaign env with achievement rewards.
  - A batched `VectorEnv` and a stable-baselines3 adapter.
- **O(1) state save/restore.** The physics is a 46-float vector, ideal for
  Go-Explore, MCTS, search and demonstrations. The whole campaign can be
  snapshotted too.
- **Human-playable.** A pygame front end, including the full campaign with
  dialogue and achievement pop-ups, plus `.tas` recordings compatible with the
  real game's TAS tool.

## Quick start

```bash
git clone https://github.com/TranBaDat2607/jumpking-rl
cd jumpking-rl
pip install -e ".[dev]"            # add ".[train]" for torch + stable-baselines3

jumpking play                      # play it yourself: arrows + Space
jumpking play --campaign new_babe_plus --profile save.json   # the whole game
jumpking achievements              # the 48 achievements and how to get them
jumpking info                      # is your Steam copy detected? is numba on?
```

```python
import gymnasium as gym

env = gym.make("jumpking.envs:JumpKing-v0", ending="ghost_of_the_babe", reward="route")
obs, info = env.reset(seed=0)
for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
print(info["screen"], info["route_progress"])
```

Low-level control is available too:

```python
from jumpking import Game

g = Game()                                  # the original world if installed, else built-in
g.step(direction=1, jump=False)             # one 60 Hz tick of button input
events, ticks = g.jump(direction=-1, charge=36)   # full-power jump left, waits for landing
g.wear(giant_boots=True)                    # item physics
snap = g.snapshot(); g.restore(snap)        # bit-exact save / restore
```

## The whole game: three stories and 48 achievements

With the game installed, `jumpking.campaign.Campaign` plays Jump King frame by
frame like the real thing:

| Part | What is simulated |
| --- | --- |
| Stories | New Game, New Babe+ (with the crown sacrifice that starts you in the Imp Room) and Ghost of the Babe, in one 169-screen world with three babes |
| Items | gold rings, rubies, silver coins, mushrooms, ghost fragments, the gnome hat, the cap; pick-ups, inventory, wearable outfits with their layers; the Giant Boots (no walking, landing thud) and the Snake Ring (icy floors, no snow) |
| NPCs | the old man, the hermit, the imp, the gnome, the archaeologist, the stone men, the troll king, the fairies, the cats, the bog man, the gargoyles: dialogue pools that grow as you reach new screens, timed by the game's own text |
| Merchants | the shoe merchant (gold rings), the Giant Boots merchant (silver), the snake (Snake Ring for a ruby), the mould man (yellow shoes) and the mushroom scholar (tunic), with the game's per-run prices |
| Birds | the raven, the white raven, the red bird and the fly, fleeing along their perches and dropping their treasure |
| Secrets | illusory walls, the Imp Room, the hidden nose, the sea of Deep Ruin, Chapel Perilous |
| Progress | run statistics (time, jumps, falls), lifetime statistics, story flags, the endings' crowns and the owl cape, a persistent profile (JSON) |

All 48 achievements are implemented with the game's exact rules
(`jumpking achievements` lists them). `tests/test_campaign.py` unlocks every
one of them on a single profile over a series of runs, and replays the three
real speed-runs inside the campaign. That replay unlocks what a player of
those runs really gets; for example, the base-game run scares the raven
exactly twice (*Fool me Twice...*) and the Ghost of the Babe run has exactly
two falls (*Nobody's Perfect*).

```python
from jumpking.campaign import Campaign

c = Campaign()                      # needs the game installed
c.new_game("normal")
for _ in range(60 * 20):
    c.step(0, False)                # stand next to the old man and listen
print(c.entity("old_man").talk.bb["line"], c.unlocked)
```

## Environments

| id | action space | observation |
| --- | --- | --- |
| `JumpKing-v0` | macro, `Discrete(113)`: 3 directions × 35 charge levels + 8 walks | vector, `float32 (26,)` |
| `JumpKingGrid-v0` | macro | tile grid, `uint8 (3, 90, 60)` |
| `JumpKingFrame-v0` | per-frame buttons, `Discrete(6)` | vector |
| `JumpKingPixels-v0` | per-frame buttons | RGB, `uint8 (90, 120, 3)` |
| `JumpKingCampaign-v0` | per-frame buttons, the whole game | vector + run stats + outfit + 48 achievement bits |
| `JumpKingCampaignMacro-v0` | macro, the whole game | same |

- **Endings**: `ending="normal" | "new_babe_plus" | "ghost_of_the_babe" | "any"`.
  Reaching any babe ends the episode, as in the game.
- **Rewards**: `height` (potential-based), `max_height`, `screen`, `route`
  (progress along the chosen ending's route), `sparse`, or your own
  callable. Optional goal bonus and step penalty. The campaign envs reward
  achievements and the ending.
- **Curriculum**: `start="random_screen"`, `"imp_room"`, a list of screens,
  or `reset(options={"screen": k})` / `{"state": snapshot}`.
- **Info dict**: full state, event flags (jump, land, splat, bump, screen
  up/down, goal, water, sand), the ending reached, route progress and the
  game ticks consumed.
- **Batched**: `jumpking.envs.vector.JumpKingVectorEnv(num_envs=512)` and
  `SB3VecEnv` advance every king in one multi-threaded numba call.
  `jumpking.batch.BatchSim` exposes the raw batched simulator for genetic
  algorithms, ES or MCTS.

Every option is documented in **[docs/RL_GUIDE.md](docs/RL_GUIDE.md)**.

## Fidelity: verified against the real game

The physics kernel is a port of the game's own body pipeline (`BodyComp`
behaviours) and player behaviour tree, decompiled from `JumpKing.exe`. It
covers:

- wind, X move, side teleports and X collision with pixel back-off;
- Y move with the ground probe, then Y collision;
- gravity and the block behaviours (ice, snow, water, sand, quark);
- controls and camera;
- collision merged per screen exactly like the game, slope triangles tested
  like the game, and float32 arithmetic throughout.

Every rule is sourced in **[docs/RESEARCH.md](docs/RESEARCH.md)**.

With the game installed, `tests/test_original.py` runs these checks:

| Check | Result |
| --- | --- |
| Base-game TAS ([ShootMe/JumpKingTAS](https://github.com/ShootMe/JumpKingTAS) `NewGame.tas`) on the real world | 43 / 43 sections; babe on the exact recorded frame, 3:44.859 |
| New Babe+ TAS (`NewBabe.tas`): wind, no-wind zones, water, ice, snow | 56 / 56 sections; babe on the exact recorded frame, 5:20.637 |
| Ghost of the Babe TAS (`NewGhost.tas`): water, sand, quark, the Hole | 71 / 71 sections; babe on the exact recorded frame, 5:35.138 |
| `PlayerValues` read from your `JumpKing.exe` | walk 1.5, jump 3.5, max fall 10, bounce 0.5, charge 0.6 s, splat 0.5 s, hitbox 18×26 |

The routes were recorded in 2019-2020. One detail of those builds differs
from today's game: side teleports used to keep the fractional y coordinate.
It is available as `legacy_teleport=True`, and only the Ghost route depends
on it. You can run the checks by hand:

```bash
python scripts/validate_original.py --download     # tick-level parity report, all three routes
python scripts/extract_constants.py                # constants from your JumpKing.exe
```

**Scope.** Everything that affects play is in scope. Sound, graphics,
cutscene animation, the workshop and mods are not.

## Performance

Measured on an Intel Core i7 (12 threads), Windows 11, Python 3.13, on the
original world:

| Workload | Throughput |
| --- | --- |
| Raw game ticks, one core | **2.75 M ticks/s** (≈45,000× real time) |
| Macro actions (full jumps), one core | 53 k/s |
| `JumpKing-v0`, single env | 33 k steps/s |
| `JumpKingVectorEnv`, 512 kings, all cores | **440 k steps/s** |
| `Campaign` (all NPCs, birds and items), one core | 15 k frames/s |

Run `jumpking benchmark` to measure your own machine.

## Baselines

| Agent | Map | Result |
| --- | --- | --- |
| Random macro actions | original / built-in | screens 1–2 |
| PPO, 14 s of training (v0.2.0) | original (base game) | mean return 0.01 → 0.18; best height screen 3 |
| Go-Explore (`examples/go_explore.py`), route-scored cells | original world, normal ending | reaches screen 38 (Blue Ruin) in 10 min, then its archive stalls; under v0.2 physics the same search beat the game in 132 s |
| Go-Explore | built-in | beats the map in 46 s |

The PPO row only shows that the training pipeline works. Solving Jump King
with RL alone remains an open challenge.

```bash
python scripts/train_ppo.py --timesteps 5e6 --n-envs 64     # PPO on the batched env, TensorBoard in runs/
python scripts/evaluate.py --model runs/<name>/final_model.zip --episodes 20 --save-tas best.tas
python examples/go_explore.py --ending new_babe_plus --start imp_room --save-tas route.tas
jumpking replay route.tas                                   # watch any recording
```

## Maps

| Map | How to use | Notes |
| --- | --- | --- |
| **Original** | `level="original"` (default when installed) | The whole 169-screen world read from `Steam/steamapps/common/Jump King/Content/level.xnb`. Requires owning the game. `level="original-base"` keeps only the base game and the Imp Room. |
| **Built-in** | `level="builtin"` (fallback) | Original 43-screen layout following the base game's areas. Generated by the physics engine itself, so solvable by construction; the verified route is stored. |
| **Custom** | `level="path/to/level.png"` | Any map in the native 780×585 hitbox-atlas format, including water, sand, quark and coloured wind blocks. |

<details>
<summary>Built-in map overview</summary>

![built-in map](docs/images/map.png)

</details>

## CLI

| Command | Purpose |
| --- | --- |
| `jumpking play` | keyboard play (R restart, F1 debug, F2 record `.tas`, F5/F9 save/load) |
| `jumpking play --campaign MODE [--profile save.json]` | the whole game: dialogue, birds, items, achievements (B/N toggle the Giant Boots / Snake Ring) |
| `jumpking achievements [--profile save.json]` | the 48 achievements, their rules and your progress |
| `jumpking demo` | watch the built-in solution |
| `jumpking replay file.tas` | play back any recording, including the real game's TAS files |
| `jumpking solve --screen N` | BFS solver over landing states |
| `jumpking preview` | render the map to PNG |
| `jumpking benchmark` | measure simulation speed |
| `jumpking info` | environment and install diagnostics |

## Project layout

```
jumpking/
  physics.py      tick kernel and macros (numba JIT, pure-Python fallback)
  level.py        native hitbox atlas, slopes, wind, teleports, endings
  routes.py       the three stories' routes, starts and route progress
  campaign.py     the whole game: runs, items, NPCs, merchants, birds, achievements
  achievements.py the 48 achievements and their rules
  content.py      reader for the game's content files (NPCs, items, ...) from your install
  bt.py           port of the game's behaviour-tree library
  verify.py       tick-level parity against the community TAS routes
  xnb.py          level.xnb reader + Steam discovery
  envs/           Gymnasium envs (incl. the campaign) + batched VectorEnv / SB3 adapter
  game.py         high-level API
  batch.py        batched simulator
  solver.py       BFS over landings
  mapgen.py       physics-driven map generator
  render.py       numpy renderer
  play.py         pygame viewer (and campaign mode)
  tas.py          recordings
  cli.py          `jumpking` command
scripts/          train_ppo, evaluate, validate_original, extract_constants, build_map
examples/         go_explore
tests/            111 tests: physics and DLC mechanics, IO, envs, batching, map solvability,
                  original-game parity on all three stories, all 48 achievements
docs/             RESEARCH.md (mechanics + sources), RL_GUIDE.md (API reference)
```

## Contributing

Bug reports, fidelity findings and new baselines are very welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md). Physics changes must keep the original-game
parity tests passing.

## Citation

If this project helps your research, please cite it. GitHub's "Cite this
repository" button uses [CITATION.cff](CITATION.cff).

```bibtex
@software{jumpking_rl,
  author  = {Tran, Ba Dat},
  title   = {jumpking-rl: a tick-exact Jump King simulator for reinforcement learning},
  year    = {2026},
  url     = {https://github.com/TranBaDat2607/jumpking-rl},
  version = {0.3.0}
}
```

## Acknowledgements

Mechanics research builds on the work of the Jump King community:

- [ShootMe/JumpKingTAS](https://github.com/ShootMe/JumpKingTAS): the base-game, New Babe+ and Ghost of the Babe TAS routes used for parity testing.
- [anyalink99/jump-king-mods](https://github.com/anyalink99/jump-king-mods): the parity-tested description of the native physics pipeline.
- [qequ/jump-king-parabole](https://github.com/qequ/jump-king-parabole): physics constants.
- [TeamNexile/jk-workshop-docs](https://github.com/TeamNexile/jk-workshop-docs) and [JumpKingPlus](https://jumpkingplus.github.io/workshop/docs/modding/hitboxes/): the level format.
- [ILSpy](https://github.com/icsharpcode/ILSpy): used to read the game's own code from your copy.

No code from these projects is included. See [docs/RESEARCH.md](docs/RESEARCH.md)
for exactly what was learned from each.

## License and disclaimer

The code, documentation and built-in map are released under the
[MIT License](LICENSE).

*Jump King* is a trademark of **Nexile**. This is an independent, fan-made
research project, not affiliated with or endorsed by Nexile or UKIYO
Publishing. It ships **no game assets**: the map, the content files and the
executable are only read at runtime from a copy you own. Please
[buy the game](https://store.steampowered.com/app/1061090/Jump_King/); it's
great.
