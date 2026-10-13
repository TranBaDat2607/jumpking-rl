# RL guide

## `JumpKingEnv` arguments

| argument | default | meaning |
| --- | --- | --- |
| `level` | `"auto"` | `"auto"` (original if installed, else built-in), `"original"` (the whole 169-screen world), `"original-base"` (screens 1–43 + the Imp Room), `"builtin"`, a `.png`/`.xnb` path, or a `Level` |
| `ending` | `"normal"` | which babe counts as success: `"normal"` (screen 43), `"new_babe_plus"` (100), `"ghost_of_the_babe"` (154) or `"any"`. Reaching any babe ends the episode. |
| `action_mode` | `"macro"` | `"macro"` or `"frame"` |
| `obs_mode` | `"vector"` | `"vector"`, `"grid"`, `"rgb"` or `"dict"` |
| `reward` | `"height"` | `"height"`, `"max_height"`, `"screen"`, `"route"`, `"sparse"`, or `f(prev_info, info, events) -> float` |
| `goal_bonus` | `10.0` | added once when the babe of `ending` is reached |
| `step_penalty` | `0.0` | subtracted every step |
| `max_episode_steps` | 2000 (macro) or 20 min of ticks (frame) | truncation limit |
| `frame_skip` | `4` | ticks per frame-mode action |
| `charges` | `2..36` | jump charge ticks offered in macro mode (power = charge / 36) |
| `walk_ticks` | `(4, 8, 16, 32)` | walk durations offered in macro mode |
| `start` | `"start"` | `"start"`, `"imp_room"` (New Babe+'s crown-sacrifice start), `"random_screen"`, a screen index, or a list of indices |
| `max_start_screen` | last screen | upper bound for `"random_screen"` |
| `wind_phase` | `"random"` | `"random"`, `"zero"`, or a number of seconds added to the wind clock |
| `splat_time` | `0.5` | splat recovery time in grounded seconds (the game's value) |
| `image_size` | `(90, 120)` | `rgb` observation size |
| `macro_timeout` | `1200` | maximum ticks per macro action |
| `giant_boots`, `snake_ring` | `False` | wear the items (their physics: no walking and a landing freeze; icy floors and no snow) |
| `render_mode` | `None` | `"rgb_array"` or `"human"` |

## Actions

**Macro mode**: `Discrete(3 * len(charges) + 2 * len(walk_ticks))`.

- Index `i < 3 * len(charges)` is a jump with direction
  `(-1, 0, 1)[i // len(charges)]` and charge `charges[i % len(charges)]`.
- The remaining indices are walks with direction `(-1, 1)` and the given
  number of ticks.
- `env.action_meaning(a)` and `env.macro_index(kind, direction, value)`
  convert between indices and meanings.
- A jump presses jump with the direction held, keeps it until the charge is
  reached, releases, then runs until the king has landed and stands still.
- `info["ticks_used"]` is the real game time the action consumed.

**Frame mode**: `Discrete(6)`, with the buttons held for `frame_skip` ticks:

| index | buttons |
| --- | --- |
| 0 | noop |
| 1 | left |
| 2 | right |
| 3 | jump |
| 4 | jump + left |
| 5 | jump + right |

These are exactly the real game's inputs (charge by holding jump, release to
jump), so policies trained here use the same control interface as the game.

## Observations

### `vector`

26 float32 values, in this order (`jumpking.envs.VECTOR_FEATURES`):

| # | feature | scale |
| --- | --- | --- |
| 0 | `x` | x / 480 |
| 1 | `y_in_screen` | 0 = top of the screen, 1 = bottom |
| 2, 3 | `vx`, `vy` | / 10 |
| 4 | `screen` | screen / (n − 1) |
| 5–8 | `on_ground`, `knocked`, `on_ice`, `on_snow` | 0 / 1 |
| 9, 10 | `charging`, `charge` | charge / 0.6 s |
| 11 | `splat` | 0 / 1 |
| 12, 13 | `wind`, `wind_active` | wind force / 0.1, latch |
| 14 | `facing` | −1 / 1 |
| 15, 16 | `height`, `max_height` | / total map height |
| 17 | `jump_held` | 0 / 1 |
| 18 | `x_tile_frac` | x mod 8 / 8 |
| 19 | `screen_has_wind` | 0 / 1 |
| 20, 21 | `wind_phase_sin`, `wind_phase_cos` | 0 on windless screens |
| 22, 23 | `in_water`, `in_sand` | 0 / 1 |
| 24, 25 | `giant_boots`, `snake_ring` | 0 / 1 |

### `grid`

`uint8 (3, 90, 60)` holding the current screen (bottom half) and the screen
above it (top half), one cell per 8 px tile.

| channel | contents |
| --- | --- |
| 0 | blocking terrain (255) |
| 1 | ice 255, water 208, snow 160, sand 128, slope 96, quark 64, no-wind 48 |
| 2 | the player's hitbox (255) |

SB3's `CnnPolicy` accepts it directly.

### `rgb` and `dict`

- `rgb`: the rendered frame resized to `image_size`.
- `dict`: `{"vector": ..., "grid": ...}`, for `MultiInputPolicy`.

## `info`

Every step's info dict holds:

- position and velocity: `x`, `y`, `vx`, `vy`;
- location: `screen` (0-based), `area`;
- progress: `height` (px above the bottom of screen 1), `max_height`;
- state flags: `on_ground`, `knocked`, `on_ice`, `on_snow`, `in_water`,
  `in_sand`, `charging`, `charge`, `splat`, `giant_boots`, `snake_ring`;
- `wind`;
- counters: `ticks`, `jumps`, `falls`, `splats`;
- `goal` (any babe reached), `ending` (1 normal, 2 New Babe+, 3 Ghost of the
  Babe), `success` (the babe of `ending`);
- `route_progress` (screens of the ending's route reached in order) and
  `route_steps` (new ones this step);
- `events` (bit flags from `jumpking.physics`):

  | flag | meaning |
  | --- | --- |
  | `EV_JUMP` | jumped |
  | `EV_LAND` | landed |
  | `EV_SPLAT` | splatted |
  | `EV_BUMP_WALL` | hit a wall |
  | `EV_BUMP_CEIL` | hit a ceiling |
  | `EV_SCREEN_UP` | moved up a screen |
  | `EV_SCREEN_DOWN` | fell down a screen |
  | `EV_GOAL` | reached the babe |
  | `EV_TELEPORT` | went through a side teleport |
  | `EV_SLOPE` | touched a slope |
  | `EV_GETUP` | stood up after a splat |
  | `EV_WATER` | entered or left water |
  | `EV_SAND` | inside sand |

- `ticks_used`.

## State access

`env.get_state()` returns the complete simulation state: a float64 vector
whose layout is given by the `S_*` indices in `jumpking/physics.py`.
`env.set_state(s)` restores it exactly. Uses:

- **Go-Explore and archive methods.** `examples/go_explore.py` (cells scored
  by route progress, `--ending` picks the babe) solves the built-in map in
  about a minute.
- **MCTS and other planners.**
- **Resets in the middle of a level.**
- **Collecting demonstrations.** `jumpking.solver.bfs` finds exact action
  sequences between states.
- **Hindsight and replay buffers.**

The wind depends on the tick counter and the wind-clock offset. Both are part
of the state vector, so restoring a state is bit-exact, including the wind
timing.

## Reward design

- **`height`** is potential-based: the episode return equals the net height
  gained divided by 360. Falling is penalised exactly as much as climbing is
  rewarded, so it cannot be exploited by oscillating.
- **`max_height`** rewards only new records. It pairs well with exploration
  bonuses because it never punishes failed attempts.
- **`screen`** is coarse and close to how humans measure progress.
- **`route`** pays +1 for each new screen along the route of `ending`
  (`jumpking.routes`). The New Babe+ and Ghost of the Babe routes go
  sideways through teleports and down the Hole and the Bog, so height alone
  is a misleading signal there: Redcrown Woods' side passage leads straight
  into Philosopher's Forest, the highest screens of the atlas.
- **Custom rewards** can use `prev_info`, `info` and `events`, for example
  `-0.01 if events & EV_SPLAT else 0`.

## Batched simulation

**`jumpking.envs.vector.JumpKingVectorEnv(num_envs=N, ...)`** is a
`gymnasium.vector.VectorEnv` with macro actions and vector observations.

- It takes the same reward, curriculum and wind options as `JumpKingEnv`.
- One step advances all N kings in a single multi-threaded numba call:
  about 440k steps/s with N = 512.
- It also takes `ending`, `giant_boots` and `snake_ring`, and the `route`
  reward.
- Finished sub-envs are reset in the same step. Their last observation is in
  `infos["final_obs"]`.
- `get_states()` / `set_states()` expose the `(N, STATE_SIZE)` state array.

**`jumpking.envs.vector.SB3VecEnv`** is the same environment as a
stable-baselines3 `VecEnv`. `scripts/train_ppo.py` uses it by default.

**`jumpking.batch.BatchSim(level, n)`** is the raw backend:

- `macro(kinds, dirs, vals)` applies one macro action per king.
- `frames(dirs, jumps, repeat)` holds buttons on every king.
- `heights`, `screens` and `done` are vectorized queries.

It suits population methods.

## Performance tips

- Prefer macro actions with vector or grid observations for learning. Use
  frame mode for research on low-level control.
- Prefer `JumpKingVectorEnv` / `SB3VecEnv` for vector observations. With a
  simulator this fast, inter-process communication costs more than the
  simulation itself, so async/subprocess vector envs are slower: about 17k
  steps/s with 8 processes. Use them only for pixel or grid observations, where
  rendering dominates.
- The first import JIT-compiles the kernel (a few seconds). numba caches the
  result in `jumpking/__pycache__`.
- Curriculum: train with `start="random_screen"` and raise
  `max_start_screen` over time.
- Use `wind_phase="random"` (the default) so policies do not overfit a single
  wind timing.

## Recordings

- `jumpking.tas` reads and writes the JumpKingTAS text format, so `.tas` files
  from the real game's TAS tool can be replayed (use `level="original"`).
- `tas.macro_to_inputs` expands macro actions into exact per-tick inputs.
- `scripts/evaluate.py --save-tas` exports the best episode.
- `jumpking replay file.tas` shows a recording in the viewer.

## The whole game: `JumpKingCampaignEnv`

`JumpKingCampaign-v0` (frame actions) and `JumpKingCampaignMacro-v0` (macro
actions) play the complete game with `jumpking.campaign.Campaign`: three
stories in one world, NPCs and their dialogue, merchants, the four birds,
world items, wearable items, endings, statistics and all 48 Steam
achievements. They need a local Jump King install; the game's content files
are read from it at runtime.

| argument | default | meaning |
| --- | --- | --- |
| `mode` | `"normal"` | the story of a new game: `"normal"`, `"new_babe_plus"`, `"ghost_of_the_babe"` |
| `action_mode` | `"frame"` | `"frame"` (`Discrete(6)` × `frame_skip`) or `"macro"` |
| `achievement_bonus` | `1.0` | per newly unlocked achievement |
| `ending_bonus` | `10.0` | for reaching the babe of `ending` (default: the mode's) |
| `route_reward` | `0.0` | per new screen along the route (shaping) |
| `keep_profile` | `True` | keep inventory, items and achievements between episodes, like the game's permanent save |
| `sacrifice_crown` | `True` | New Babe+ with the crown: start in the Imp Room |

- **Episodes** are runs: `reset()` starts a new game, `reset(options=...)`
  takes `mode`, `profile` (a `Profile` or `"new"`), `equip` (items to put on),
  `sacrifice_crown` and `state` (a `Campaign.snapshot()`).
- **Observations**: the 26 vector features, run statistics, the worn items and
  the 48-bit achievement mask (`jumpking.envs.campaign_env.CAMPAIGN_FEATURES`).
- **info**: the physics info plus `ended`, `ending`, `success`,
  `new_achievements`, `achievements` (count), `run` statistics and route
  progress.
- **Speed**: about 15k frames/s; NPCs, merchants and birds update every frame,
  as in the game.

Several achievements need more than one run (the crown for Double King, ten
gold rings for Big Spender, the Giant Boots for the Giant Leaps runs), so keep
the profile between episodes. `Campaign` can also be used directly:

```python
from jumpking.campaign import Campaign, Profile

c = Campaign(profile=Profile.load("save.json"))   # or Campaign() for a new profile
c.new_game("ghost_of_the_babe")
c.step(direction=1, jump=False)                     # one frame
c.jump(direction=-1, charge=36)                     # a macro, frame by frame
c.equip("GiantBoots")                               # put on an owned item
print(c.unlocked, c.run_stats(), c.profile.inventory)
snap = c.snapshot(); c.restore(snap)                # the whole game state
c.profile.save("save.json")
```

`jumpking achievements [--profile save.json]` lists the 48 achievements with
their exact unlock rules, and `jumpking play --campaign MODE --profile
save.json` plays the campaign with the keyboard.
