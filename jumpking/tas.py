"""Input recordings in the JumpKingTAS text format.

Each line is ``frames,ACTIONS`` (e.g. ``35,L,J`` = hold left + jump for 35
ticks; a bare number = no input). ``#`` starts a comment, ``@x,y,dir`` places
the king at (x/100, y/100), ``X`` resets statistics. Menu-only actions
(U, D, P, C) are ignored. The clone additionally understands a
``#!state v0,v1,...`` comment carrying the complete simulation state (tick
counter, velocities, ...) for bit-exact replays; other tools ignore it. This makes recordings from the real game's TAS tool
(e.g. ``LevelFiles/NewGame.tas``) replayable in the clone.
"""
import numpy as np

from . import physics as P


class Segment:
    __slots__ = ("frames", "direction", "jump", "state", "reset", "line")

    def __init__(self, frames=0, direction=0, jump=False, state=None, reset=False, line=0):
        self.frames = frames
        self.direction = direction
        self.jump = jump
        self.state = state
        self.reset = reset
        self.line = line


def parse(text):
    segments = []
    for n, raw in enumerate(text.splitlines(), 1):
        if raw.startswith("#!state"):
            vals = np.array([float(v) for v in raw[len("#!state"):].replace(",", " ").split()])
            segments.append(Segment(state=vals, line=n))
            continue
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("***"):
            continue
        if line.startswith("@"):
            parts = [p.strip() for p in line[1:].split(",")]
            x, y = int(parts[0]) / 100.0, int(parts[1]) / 100.0
            facing = int(parts[2]) if len(parts) > 2 else 1
            segments.append(Segment(state=(x, y, 1 if facing == 1 else -1), line=n))
            continue
        parts = [p.strip().upper() for p in line.split(",")]
        try:
            frames = int(parts[0])
        except ValueError:
            continue
        acts = set(parts[1:])
        d = (1 if "R" in acts else 0) - (1 if "L" in acts else 0)
        segments.append(Segment(frames, d, "J" in acts, reset="X" in acts, line=n))
    return segments


def load(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return parse(fh.read())


def to_inputs(segments, skip_until_state=True):
    """Flatten to [(direction, jump)] per tick, plus the start state.

    With ``skip_until_state`` everything before the first ``@`` line (menus,
    intro) is dropped."""
    start = None
    inputs = []
    started = not skip_until_state
    for seg in segments:
        if seg.state is not None:
            if start is None or (isinstance(seg.state, np.ndarray) and not inputs):
                start = seg.state
            started = True
            continue
        if not started:
            continue
        inputs.extend([(seg.direction, 1 if seg.jump else 0)] * seg.frames)
    return start, inputs


def replay(game, segments):
    """Run a parsed recording on ``game``. Returns per-tick (x, y, screen)."""
    start, inputs = to_inputs(segments)
    if isinstance(start, np.ndarray):
        game.restore(start)
    elif start is not None:
        x, y, facing = start
        game.reset(x, y)
        game.state[P.S_FACING] = facing
    trace = np.zeros((len(inputs), 3), np.float64)
    for i, (d, j) in enumerate(inputs):
        game.step(d, j)
        trace[i] = (game.x, game.y, game.screen)
    return trace


def compress(inputs):
    """Run-length encode [(direction, jump)] into Segments."""
    segs = []
    for d, j in inputs:
        if segs and segs[-1].direction == d and segs[-1].jump == bool(j):
            segs[-1].frames += 1
        else:
            segs.append(Segment(1, d, bool(j)))
    return segs


def dumps(segments, start=None, header=None, state=None):
    lines = []
    if header:
        lines += ["#" + h for h in header.splitlines()]
    if state is not None:
        lines.append("#!state " + ",".join(repr(float(v)) for v in state))
        if start is None:
            start = (state[P.S_X], state[P.S_Y], state[P.S_FACING])
    if start is not None:
        x, y, facing = start
        lines.append("@%d,%d,%d" % (round(x * 100), round(y * 100), 1 if facing >= 0 else 0))
    for s in segments:
        acts = []
        if s.direction < 0:
            acts.append("L")
        if s.direction > 0:
            acts.append("R")
        if s.jump:
            acts.append("J")
        lines.append(("%4d" % s.frames) + ("," + ",".join(acts) if acts else ""))
    return "\n".join(lines) + "\n"


def save(path, inputs, start=None, header=None, state=None):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dumps(compress(inputs), start, header, state))


def macro_to_inputs(game, actions):
    """Expand macro actions into the exact per-tick inputs they produce
    (recorded by stepping a copy of ``game``)."""
    from .game import Game

    g = Game(game.level)
    g.params[:] = game.params
    g.restore(game.snapshot())
    inputs = []

    def press(d, j):
        g.step(d, j)
        inputs.append((d, j))

    def settle():
        for _ in range(60 * 30):
            if P.is_settled(g.state):
                break
            press(0, 0)

    for kind, d, v in actions:
        if kind == "walk":
            n = 0
            if g.state[P.S_SPLAT] > 0 and g.state[P.S_SPLAT_DONE] > 0:
                press(d, 0)
                n += 1
            while n < v:
                press(d, 0)
                n += 1
            press(0, 0)
            settle()
        else:
            if g.state[P.S_JUMP_HELD] > 0:
                press(d, 0)
            if g.state[P.S_SPLAT] > 0 and g.state[P.S_SPLAT_DONE] > 0:
                press(d, 0)
            target = (max(2, min(36, v)) - 1) * P.DT - 1e-6
            jumped = False
            started = False
            waited = 0
            for _ in range(60 * 30):
                ev = g.step(d, 1)
                inputs.append((d, 1))
                if ev & P.EV_JUMP:
                    jumped = True
                    break
                if g.state[P.S_CHARGING] > 0:
                    started = True
                    if g.state[P.S_CHARGE] >= target:
                        break
                else:
                    waited += 1
                    if started or waited > 8:
                        break
            if not jumped and g.state[P.S_CHARGING] > 0:
                press(d, 0)
            press(0, 0)
            settle()
    return inputs
