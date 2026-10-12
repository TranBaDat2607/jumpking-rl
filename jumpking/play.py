"""Interactive pygame front end: play, watch replays, inspect levels.

Controls
--------
Left / Right (or A / D)   walk, choose jump direction
Space (or Up / W)         hold to charge a jump, release to jump
R                         restart from the beginning
F1                        toggle the debug overlay
F2                        start / stop recording (.tas, saved in ./replays)
F5 / F9                   save / load a practice state
PageUp / PageDown         practice: move to the next / previous screen
Esc                       quit
"""
import os
import time

import numpy as np

from . import constants as C
from . import physics as P
from .game import Game
from .render import Renderer


def _require_pygame():
    try:
        import pygame  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pygame is required for the viewer: pip install pygame") from exc
    import pygame

    return pygame


class Viewer:
    def __init__(self, game, scale=2, hz=C.FPS, title="Jump King clone"):
        pygame = _require_pygame()
        self.pg = pygame
        self.game = game
        self.renderer = Renderer(game.level)
        self.scale = scale
        self.hz = hz
        pygame.init()
        self.screen = pygame.display.set_mode((C.SCREEN_W * scale, C.SCREEN_H * scale))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 14)
        self.big = pygame.font.SysFont("georgia,serif", 36, bold=True)
        self.debug = True
        self.trail = []

    def draw(self, extra=None):
        pg = self.pg
        frame = self.renderer.frame(self.game.state, trail=self.trail[-240:])
        surf = pg.surfarray.make_surface(frame.swapaxes(0, 1))
        surf = pg.transform.scale(surf, self.screen.get_size())
        self.screen.blit(surf, (0, 0))
        info = self.game.info()
        if self.debug:
            lines = [
                "%s  (screen %d/%d)" % (info["area"], info["screen"] + 1, self.game.level.n_screens),
                "pos %.2f, %.2f   vel %.3f, %.3f" % (info["x"], info["y"], info["vx"], info["vy"]),
                "height %d px  best %d px" % (info["height"], info["max_height"]),
                "ground %d  knocked %d  ice %d  snow %d  splat %d" % (
                    info["on_ground"], info["knocked"], info["on_ice"], info["on_snow"], info["splat"]),
                "charge %.3f s   wind %+.3f" % (info["charge"], info["wind"]),
                "time %s  jumps %d  falls %d" % (
                    self.fmt_time(info["ticks"]), info["jumps"], info["falls"]),
            ]
            if extra:
                lines += extra
            y = 6
            for ln in lines:
                txt = self.font.render(ln, True, (255, 255, 255))
                shadow = self.font.render(ln, True, (0, 0, 0))
                self.screen.blit(shadow, (9, y + 1))
                self.screen.blit(txt, (8, y))
                y += 16
            # charge bar
            if info["charging"]:
                w = int(120 * min(1.0, info["charge"] / C.JUMP_TIME))
                pg.draw.rect(self.screen, (40, 40, 40), (8, y + 4, 122, 10))
                pg.draw.rect(self.screen, (255, 200, 40), (9, y + 5, w, 8))
        if info["goal"]:
            msg = self.big.render("You reached the babe!", True, (255, 220, 120))
            r = msg.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 3))
            self.screen.blit(msg, r)
        pg.display.flip()

    @staticmethod
    def fmt_time(ticks):
        secs = ticks * C.TARGET_ELAPSED_SECONDS
        return "%d:%02d:%05.2f" % (secs // 3600, (secs % 3600) // 60, secs % 60)


def play(level="builtin", scale=2, hz=C.FPS, record_dir="replays"):
    from . import tas

    game = Game(level)
    v = Viewer(game, scale, hz)
    pg = v.pg
    saved = None
    recording = None
    running = True
    while running:
        for e in pg.event.get():
            if e.type == pg.QUIT:
                running = False
            elif e.type == pg.KEYDOWN:
                if e.key == pg.K_ESCAPE:
                    running = False
                elif e.key == pg.K_r:
                    game.reset()
                    v.trail.clear()
                elif e.key == pg.K_F1:
                    v.debug = not v.debug
                elif e.key == pg.K_F5:
                    saved = game.snapshot()
                elif e.key == pg.K_F9 and saved is not None:
                    game.restore(saved)
                elif e.key in (pg.K_PAGEUP, pg.K_PAGEDOWN):
                    target = game.screen + (1 if e.key == pg.K_PAGEUP else -1)
                    target = max(0, min(game.level.n_screens - 1, target))
                    try:
                        game.reset_to_screen(target)
                        game.settle()
                    except ValueError:
                        pass
                elif e.key == pg.K_F2:
                    if recording is None:
                        recording = {"state": game.snapshot(), "inputs": []}
                    else:
                        os.makedirs(record_dir, exist_ok=True)
                        path = os.path.join(record_dir, time.strftime("run-%Y%m%d-%H%M%S.tas"))
                        tas.save(path, recording["inputs"], header="recorded with the Jump King clone",
                                 state=recording["state"])
                        print("saved", path)
                        recording = None
        keys = pg.key.get_pressed()
        d = int(keys[pg.K_RIGHT] or keys[pg.K_d]) - int(keys[pg.K_LEFT] or keys[pg.K_a])
        j = bool(keys[pg.K_SPACE] or keys[pg.K_UP] or keys[pg.K_w])
        game.step(d, j)
        if recording is not None:
            recording["inputs"].append((d, int(j)))
        v.trail.append((game.x, game.y))
        extra = ["REC %d ticks" % len(recording["inputs"])] if recording is not None else None
        v.draw(extra)
        v.clock.tick(hz)
    pg.quit()


def watch_inputs(game, inputs, start=None, hz=C.FPS, speed=1.0, title="replay"):
    """Play back a list of per-tick inputs in a window."""
    v = Viewer(game, title=title)
    pg = v.pg
    if isinstance(start, np.ndarray):
        game.restore(start)
    elif start is not None:
        game.reset(start[0], start[1])
    i = 0
    paused = False
    while True:
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                pg.quit()
                return game
            if e.type == pg.KEYDOWN and e.key == pg.K_SPACE:
                paused = not paused
            if e.type == pg.KEYDOWN and e.key == pg.K_F1:
                v.debug = not v.debug
        if not paused:
            steps = max(1, int(round(speed)))
            for _ in range(steps):
                if i < len(inputs):
                    d, j = inputs[i]
                    game.step(d, j)
                    v.trail.append((game.x, game.y))
                    i += 1
        v.draw(["replay %d / %d  (space: pause)" % (i, len(inputs))])
        v.clock.tick(hz * (speed if speed < 1 else 1))
        if i >= len(inputs) and not paused:
            paused = True


def watch_policy(env, policy, episodes=1, hz=C.FPS):
    """Render an agent acting in a JumpKingEnv. ``policy(obs) -> action``."""
    v = None
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action = policy(obs)
            # expand macro actions tick by tick for display
            if env.action_mode == "macro":
                from .tas import macro_to_inputs

                kind, d, val = env.action_meaning(action)
                inputs = macro_to_inputs(env.game, [(kind, d, val)])
                if v is None:
                    v = Viewer(env.game, title="agent")
                for (dd, jj) in inputs:
                    env.game.step(dd, jj)
                    v.trail.append((env.game.x, env.game.y))
                    for e in v.pg.event.get():
                        if e.type == v.pg.QUIT:
                            return
                    v.draw()
                    v.clock.tick(hz)
                # keep env bookkeeping consistent
                prev = env._info
                env._info = env.game.info()
                env._steps += 1
                obs = env._obs()
                done = env._info["goal"] or env._steps >= env.max_episode_steps
                del prev
            else:
                obs, r, term, trunc, info = env.step(action)
                if v is None:
                    v = Viewer(env.game, title="agent")
                v.draw()
                v.clock.tick(hz)
                done = term or trunc
    if v is not None:
        v.pg.quit()


def snapshot_frames(game, inputs, every=1):
    """Render frames (numpy) of an input sequence without a window."""
    r = Renderer(game.level)
    frames = []
    for i, (d, j) in enumerate(inputs):
        game.step(d, j)
        if i % every == 0:
            frames.append(r.frame(game.state))
    return np.stack(frames) if frames else np.zeros((0, C.SCREEN_H, C.SCREEN_W, 3), np.uint8)
