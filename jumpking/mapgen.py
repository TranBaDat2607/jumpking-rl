"""Generator for the built-in 43-screen map.

The map is an original design that follows the base game's structure (10
areas, 43 screens, wind in Windswept Bluff / Stormwall Pass, ...). It is built
*by the physics engine itself*: a route is grown jump by jump, every landing
platform is placed under a simulated trajectory, and the jump is re-run with
the platform in place to confirm the landing. Scenery is added afterwards
without touching any cell swept by the route. The finished map is therefore
solvable by construction, and the route doubles as an expert demonstration.

Run ``python scripts/build_map.py`` to regenerate ``jumpking/data/level.png``.
"""
import json
import math

import numpy as np

from . import constants as C
from . import physics as P
from .game import Game
from .level import Level

W = C.SCREEN_TILES_W
H = C.SCREEN_TILES_H
NS = C.BASE_SCREENS


class Area:
    def __init__(self, name, screens, wall=(2, 3), width=(5, 10), gain=(24, 90),
                 charges=(8, 36), material="solid", wind=False, decor=0.5,
                 slopes=0.0, style="rock", thickness=(1, 2), bounce_ok=False):
        self.name = name
        self.screens = screens
        self.wall = wall
        self.width = width
        self.gain = gain
        self.charges = charges
        self.material = material
        self.wind = wind
        self.decor = decor
        self.slopes = slopes
        self.style = style
        self.thickness = thickness
        self.bounce_ok = bounce_ok


AREAS = [
    Area("Redcrown Woods", range(0, 5), wall=(2, 3), width=(6, 11), gain=(16, 80),
         charges=(6, 30), decor=0.6, slopes=0.1, style="woods", thickness=(1, 2)),
    Area("Colossal Drain", range(5, 10), wall=(7, 12), width=(4, 8), gain=(32, 100),
         charges=(10, 34), decor=0.5, slopes=0.35, style="drain", thickness=(1, 3)),
    Area("False King's Keep", range(10, 14), wall=(3, 6), width=(4, 7), gain=(40, 112),
         charges=(12, 36), decor=0.7, slopes=0.2, style="keep", thickness=(2, 3)),
    Area("Bargainburg", range(14, 19), wall=(2, 4), width=(4, 8), gain=(32, 112),
         charges=(10, 36), decor=0.8, slopes=0.5, style="town", thickness=(1, 2)),
    Area("Great Frontier", range(19, 25), wall=(1, 2), width=(3, 5), gain=(24, 128),
         charges=(14, 36), decor=0.25, slopes=0.2, style="frontier", thickness=(1, 2)),
    Area("Windswept Bluff", range(25, 26), wall=(2, 3), width=(5, 8), gain=(32, 96),
         charges=(10, 32), wind=True, decor=0.3, slopes=0.1, style="bluff"),
    Area("Stormwall Pass", range(26, 32), wall=(2, 4), width=(4, 7), gain=(32, 112),
         charges=(10, 36), wind=True, material="snow", decor=0.4, slopes=0.2,
         style="storm"),
    Area("Chapel Perilous", range(32, 36), wall=(3, 5), width=(3, 6), gain=(40, 120),
         charges=(12, 36), decor=0.6, slopes=0.6, style="chapel", thickness=(1, 2)),
    Area("Blue Ruin", range(36, 39), wall=(2, 4), width=(6, 10), gain=(32, 112),
         charges=(10, 36), material="ice", decor=0.5, slopes=0.3, style="ruin"),
    Area("The Tower", range(39, 43), wall=(4, 7), width=(3, 4), gain=(48, 128),
         charges=(14, 36), decor=0.5, slopes=0.25, style="tower", thickness=(1, 2)),
]


def area_for(screen):
    for a in AREAS:
        if screen in a.screens:
            return a
    raise KeyError(screen)


MATERIAL_TILE = {"solid": P.T_SOLID, "ice": P.T_ICE, "snow": P.T_SNOW}


class MapBuilder:
    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)
        screens = [np.zeros((H, W), np.int8) for _ in range(NS)]
        wind = [1 if area_for(i).wind else 0 for i in range(NS)]
        self.level = Level(screens, wind, [-1] * NS, [-1] * NS,
                           goal=(-1e6, -1e6, -1e6 + 1, -1e6 + 1))
        self.grid = self.level.tiles             # mutated in place
        self.reserved = np.zeros(self.grid.shape, bool)
        self.route_mask = np.zeros(self.grid.shape, bool)
        self.game = Game(self.level)
        self.route = []                            # macro actions
        self.platforms = []                        # (row, c0, c1, material)
        self.offy = self.level.offset_y

    # ------------------------------------------------------------- helpers
    def row_of_world_y(self, wy):
        return (int(math.floor(wy)) + self.offy) // C.TILE

    def screen_of_row(self, r):
        return NS - 1 - r // H

    def wall_columns(self, screen):
        return self._walls[screen]

    def build_walls(self):
        self._walls = []
        for s in range(NS):
            a = area_for(s)
            lw = int(self.rng.integers(a.wall[0], a.wall[1] + 1))
            rw = int(self.rng.integers(a.wall[0], a.wall[1] + 1))
            self._walls.append((lw, rw))
            r0 = (NS - 1 - s) * H
            self.grid[r0:r0 + H, :lw] = P.T_SOLID
            self.grid[r0:r0 + H, W - rw:] = P.T_SOLID
        # smooth wall transitions between screens with small ledges
        bottom = (NS - 1) * H
        self.grid[bottom + 41:, :] = P.T_SOLID     # floor of screen 1

    def reserve_trajectory(self, positions, margin=0):
        for (x, y) in positions:
            hx, hy = int(x), int(y) + self.offy
            c0 = max(0, (hx - margin) // 8)
            c1 = min(W - 1, (hx + C.PLAYER_W - 1 + margin) // 8)
            r0 = max(0, (hy - margin) // 8)
            r1 = min(self.grid.shape[0] - 1, (hy + C.PLAYER_H - 1 + margin) // 8)
            self.reserved[r0:r1 + 1, c0:c1 + 1] = True

    def probe(self, d, c, walk=None):
        """Run a (walk +) jump with full physics tick by tick and return the
        per-tick hitbox positions after release plus collision events."""
        g = self.game
        s = g.state
        if walk is not None:
            g.walk(*walk)
        # charge
        start = g.snapshot()
        ev, n = 0, 0
        target = (c - 1) * P.DT - 1e-6
        while n < 60:
            e = g.step(d, True)
            ev |= e
            n += 1
            if e & P.EV_JUMP:
                break
            if s[P.S_CHARGING] > 0 and s[P.S_CHARGE] >= target:
                ev |= g.step(d, False)
                break
        if not ev & P.EV_JUMP:
            return start, None, ev
        traj = []
        for _ in range(400):
            e = g.step(0, False)
            ev |= e
            traj.append((g.x, g.y, g.vy, e))
            if g.on_ground or e & (P.EV_BUMP_WALL | P.EV_BUMP_CEIL | P.EV_SLOPE):
                break
        return start, traj, ev

    def platform_ok(self, row, c0, c1, thick):
        if c0 < 1 or c1 > W - 2 or row - 5 < 0:
            return False
        if self.reserved[row:row + thick, c0:c1 + 1].any():
            return False
        if (self.grid[row:row + thick, c0:c1 + 1] != 0).any():
            return False
        # head room: 4 free rows above (player is 26 px = 3.25 tiles)
        if (self.grid[row - 4:row, c0:c1 + 1] != 0).any():
            return False
        return True

    def place_platform(self, row, c0, c1, thick, tile):
        self.grid[row:row + thick, c0:c1 + 1] = P.T_SOLID
        self.grid[row, c0:c1 + 1] = tile

    def remove_platform(self, row, c0, c1, thick):
        self.grid[row:row + thick, c0:c1 + 1] = 0

    # --------------------------------------------------------------- route
    def try_step(self, area, screen):
        g = self.game
        rng = self.rng
        base = g.snapshot()
        lw, rw = self._walls[screen]
        x = g.x
        # optional walk to a new take-off point (not on ice / snow)
        walk = None
        if not g.state[P.S_ON_ICE] and not g.state[P.S_ON_SNOW] and rng.random() < 0.5:
            plat = self.current_platform
            if plat is not None:
                _, pc0, pc1, _ = plat
                lo, hi = pc0 * 8 - 6, (pc1 + 1) * 8 - C.PLAYER_W + 6
                if hi > lo:
                    tx = rng.uniform(lo, hi)
                    ticks = int(abs(tx - x) / C.WALK_SPEED)
                    if ticks > 0:
                        walk = (1 if tx > x else -1, ticks)
        # direction: away from the nearer wall, sometimes towards it
        centre = (lw * 8 + (W - rw) * 8) / 2.0
        if rng.random() < 0.75:
            d = 1 if x < centre else -1
        else:
            d = -1 if x < centre else 1
        if rng.random() < 0.08:
            d = 0
        c = int(rng.integers(area.charges[0], area.charges[1] + 1))
        start, traj, ev = self.probe(d, c, walk)
        if traj is None:
            g.restore(base)
            return False
        start_feet = int(start[P.S_Y]) + C.PLAYER_H
        if walk is not None:
            # take-off feet height after walking (same platform)
            pass
        # candidate landing ticks: descending, inside walls, height gain ok
        cands = []
        for i, (tx, ty, tvy, e) in enumerate(traj):
            if e & (P.EV_BUMP_WALL | P.EV_BUMP_CEIL | P.EV_SLOPE):
                break
            if tvy <= 0.5 or i < 4:
                continue
            feet = int(ty) + C.PLAYER_H
            gain = start_feet - feet
            if not area.gain[0] <= gain <= area.gain[1]:
                continue
            cands.append(i)
        if not cands:
            g.restore(base)
            return False
        i = cands[int(rng.integers(len(cands)))]
        tx, ty, _, _ = traj[i]
        # the feet cross the platform top during tick i: choose the first
        # 8-px aligned row top between the previous and current feet
        prev_feet = int(traj[i - 1][1]) + C.PLAYER_H
        feet = int(ty) + C.PLAYER_H
        top_world = int(math.ceil(prev_feet / 8.0)) * 8
        if not top_world < feet:
            g.restore(base)
            return False
        row = (top_world + self.offy) // 8
        if row < 6:
            g.restore(base)
            return False
        target_screen = self.screen_of_row(row)
        if target_screen not in (screen, screen + 1) or target_screen >= NS:
            g.restore(base)
            return False
        tarea = area_for(target_screen)
        tlw, trw = self._walls[target_screen]
        width = int(rng.integers(tarea.width[0], tarea.width[1] + 1))
        hx = int(tx)
        pc_lo = hx // 8
        pc_hi = (hx + C.PLAYER_W - 1) // 8
        anchor = int(rng.integers(pc_lo, pc_hi + 1))
        c0 = anchor - int(rng.integers(0, width))
        c0 = max(tlw, min(c0, W - trw - width))
        c1 = c0 + width - 1
        if c1 < pc_lo or c0 > pc_hi:
            g.restore(base)
            return False
        thick = int(rng.integers(tarea.thickness[0], tarea.thickness[1] + 1))
        if c0 <= tlw + 1:
            c0 = tlw
        if c1 >= W - trw - 2:
            c1 = W - trw - 1
        if not self.platform_ok(row, max(c0, 1), min(c1, W - 2), thick):
            g.restore(base)
            return False
        # the rising part of this very jump must not pass through it
        for (px, py, _, _) in traj[:i]:
            hx0, hy0 = int(px), int(py) + self.offy
            if (hx0 < (c1 + 1) * 8 and hx0 + C.PLAYER_W > c0 * 8
                    and hy0 < (row + thick) * 8 and hy0 + C.PLAYER_H > row * 8):
                g.restore(base)
                return False
        mat = tarea.material
        if mat != "solid" and rng.random() < 0.4:
            mat = "solid"
        tile = MATERIAL_TILE[mat]
        self.place_platform(row, c0, c1, thick, tile)
        # verify with the macros (exactly what the route will replay)
        g.restore(base)
        ev_total = 0
        acts = []
        if walk is not None:
            e, _ = g.walk(*walk)
            ev_total |= e
            acts.append(("walk", walk[0], walk[1]))
        before = g.snapshot()
        e, n = g.jump(d, c)
        ev_total |= e
        acts.append(("jump", d, c))
        s = g.state
        ok = (P.is_settled(s) and s[P.S_SPLAT] == 0
              and int(s[P.S_Y]) + C.PLAYER_H == top_world
              and s[P.S_ON_GROUND] > 0
              and not (ev_total & (P.EV_BUMP_WALL | P.EV_BUMP_CEIL | P.EV_SLOPE
                                   | P.EV_SCREEN_DOWN)))
        if ok:
            fx = int(s[P.S_X])
            if not (fx + C.PLAYER_W > c0 * 8 and fx < (c1 + 1) * 8):
                ok = False
        if not ok:
            self.remove_platform(row, c0, c1, thick)
            g.restore(base)
            return False
        after = g.snapshot()
        # reserve everything the replay sweeps (with a margin)
        positions = [(base[P.S_X], base[P.S_Y]), (before[P.S_X], before[P.S_Y])]
        g.restore(before)
        _, traj2, _ = self.probe(d, c)
        if traj2:
            positions += [(p[0], p[1]) for p in traj2]
        g.restore(after)
        positions.append((g.x, g.y))
        self.reserve_trajectory(positions, margin=6)
        if walk is not None:
            plat = self.current_platform
            r = plat[0]
            lo = int(min(base[P.S_X], before[P.S_X])) // 8
            hi = (int(max(base[P.S_X], before[P.S_X])) + C.PLAYER_W) // 8
            self.reserved[max(0, r - 5):r, max(0, lo):min(W, hi + 1)] = True
        # keep the platform's head room clear of scenery
        self.reserved[max(0, row - 5):row, c0:c1 + 1] = True
        self.route_mask[row:row + thick, c0:c1 + 1] = True
        self.route.extend(acts)
        self.platforms.append((row, c0, c1, mat))
        self.current_platform = (row, c0, c1, mat)
        return True

    def build_route(self, verbose=False, max_attempts=600):
        g = self.game
        g.reset(C.START_X, C.START_Y)
        g.settle()
        bottom = (NS - 1) * H
        self.current_platform = (bottom + 41, 0, W - 1, "solid")
        self.reserve_trajectory([(g.x, g.y)], margin=8)
        goal_row_limit = 22  # final platform must reach the upper half of screen 43
        history = []
        failures = 0
        while True:
            s = g.state
            screen = int(s[P.S_SCREEN])
            row_feet = self.row_of_world_y(int(s[P.S_Y]) + C.PLAYER_H)
            local = row_feet - (NS - 1 - screen) * H
            if screen == NS - 1 and local <= goal_row_limit:
                break
            area = area_for(screen)
            snap = (g.snapshot(), self.reserved.copy(), self.route_mask.copy(),
                    len(self.route), len(self.platforms), self.current_platform,
                    self.grid.copy())
            ok = False
            for attempt in range(max_attempts):
                if self.try_step(area, screen):
                    ok = True
                    break
            if ok:
                history.append(snap)
                failures = 0
            else:
                # backtrack: undo the last few platforms and try again
                failures += 1
                if not history or failures > 60:
                    raise RuntimeError("route generation stuck on screen %d" % (screen + 1))
                for _ in range(min(len(history), 1 + failures // 4)):
                    snap = history.pop()
                state, res, rmask, nroute, nplat, cur, grid = snap
                g.restore(state)
                self.reserved[:] = res
                self.route_mask[:] = rmask
                del self.route[nroute:]
                del self.platforms[nplat:]
                self.current_platform = cur
                self.grid[:] = grid
                if verbose:
                    print("backtrack to screen %d (failures %d)" % (int(state[P.S_SCREEN]) + 1, failures))
                continue
            if verbose:
                print("screen %2d  platforms %3d  route %3d" % (
                    int(g.state[P.S_SCREEN]) + 1, len(self.platforms), len(self.route)))
        # babe goal: widen the final platform and put the babe at its far end
        row, c0, c1, mat = self.current_platform
        lw, rw = self._walls[NS - 1]
        top_world = row * 8 - self.offy
        px = g.x
        go_right = (W - rw - 1 - c1) >= (c0 - lw)
        for _ in range(10):
            if go_right and c1 + 1 < W - rw and (self.grid[row - 5:row + 1, c1 + 1] == 0).all():
                c1 += 1
            elif not go_right and c0 - 1 >= lw and (self.grid[row - 5:row + 1, c0 - 1] == 0).all():
                c0 -= 1
            else:
                break
        self.grid[row, c0:c1 + 1] = P.T_SOLID
        bw = 20
        if go_right:
            gx0 = (c1 + 1) * 8 - bw - 1
        else:
            gx0 = c0 * 8 + 1
        self.goal = (float(gx0), float(top_world - 30), float(gx0 + bw), float(top_world))
        self.level.goal = self.goal
        self.level.params()  # (goal is read through Game params below)
        self.game.params[P.P_GOAL_X0:P.P_GOAL_Y1 + 1] = self.goal
        self.reserved[max(0, row - 8):row, c0:c1 + 1] = True
        self.route_mask[row, c0:c1 + 1] = True
        # final walk to the babe
        d = 1 if gx0 > px else -1
        for _ in range(20):
            if g.done:
                break
            e, n = g.walk(d, 6)
            self.route.append(("walk", d, 6))
        if not g.done:
            raise RuntimeError("could not reach the babe on the final platform")

    # ------------------------------------------------------------- scenery
    def free_rect(self, r0, c0, r1, c1, margin=1):
        R0, C0 = max(0, r0 - margin), max(0, c0 - margin)
        R1, C1 = min(self.grid.shape[0] - 1, r1 + margin), min(W - 1, c1 + margin)
        if self.reserved[R0:R1 + 1, C0:C1 + 1].any():
            return False
        return True

    SUPPORT = {  # style: (probability, kind)
        "woods": (0.7, "trunk"), "drain": (0.5, "block"), "keep": (0.8, "block"),
        "town": (0.5, "block"), "frontier": (0.35, "pillar"), "bluff": (0.5, "block"),
        "storm": (0.6, "block"), "chapel": (0.5, "pillar"), "ruin": (0.6, "block"),
        "tower": (0.7, "pillar"),
    }

    def add_supports(self):
        """Grow the floating route platforms down into pillars / trunks / blocks."""
        rng = self.rng
        rows = self.grid.shape[0]
        for (row, c0, c1, mat) in self.platforms:
            a = area_for(self.screen_of_row(row))
            prob, kind = self.SUPPORT[a.style]
            if rng.random() > prob:
                continue
            if kind == "trunk" or (kind == "pillar" and c1 - c0 >= 2):
                mid = (c0 + c1) // 2 + int(rng.integers(-1, 2))
                cols = range(max(c0, mid), min(c1, mid + 1) + 1)
            elif kind == "pillar":
                cols = range(c0, c1 + 1)
            else:
                cols = range(c0, c1 + 1)
            length = int(rng.integers(3, 26))
            r = row + 1
            while r < rows and r <= row + length:
                if self.reserved[r, cols.start:cols.stop].any():
                    break
                if (self.grid[r, cols.start:cols.stop] != 0).any():
                    # merge with what is below
                    self.grid[r, cols.start:cols.stop][self.grid[r, cols.start:cols.stop] == 0] = P.T_SOLID
                    break
                self.grid[r, cols.start:cols.stop] = P.T_SOLID
                r += 1

    def add_scenery(self):
        rng = self.rng
        self.add_supports()
        for s in range(NS):
            a = area_for(s)
            lw, rw = self._walls[s]
            base = (NS - 1 - s) * H
            n_attempts = int(160 * a.decor)
            for _ in range(n_attempts):
                kind = rng.random()
                if a.style in ("woods",) and kind < 0.4:
                    # tree trunk: thin vertical column with a canopy blob
                    c = int(rng.integers(lw + 1, W - rw - 2))
                    r1 = base + int(rng.integers(10, H))
                    r0 = r1 - int(rng.integers(6, 18))
                    if self.free_rect(r0, c, r1, c + 1, margin=1):
                        self.grid[r0:r1 + 1, c:c + 2] = P.T_SOLID
                    continue
                if a.style in ("town",) and kind < 0.35:
                    # house with a pitched roof (slopes)
                    w = int(rng.integers(4, 9)) * 2
                    c0 = int(rng.integers(lw, W - rw - w))
                    r1 = base + int(rng.integers(12, H))
                    h = int(rng.integers(3, 8))
                    roof = w // 2
                    if self.free_rect(r1 - h - roof, c0, r1, c0 + w - 1):
                        self.grid[r1 - h:r1 + 1, c0:c0 + w] = P.T_SOLID
                        for k in range(roof):
                            rr = r1 - h - 1 - k
                            self.grid[rr, c0 + k + 1:c0 + w - k - 1] = P.T_SOLID
                            self.grid[rr, c0 + k] = -1
                            self.grid[rr, c0 + w - 1 - k] = -1
                    continue
                if a.style in ("drain",) and kind < 0.3:
                    # horizontal pipe
                    w = int(rng.integers(5, 16))
                    left = rng.random() < 0.5
                    c0 = lw if left else W - rw - w
                    r = base + int(rng.integers(2, H - 2))
                    if self.free_rect(r, c0, r + 1, c0 + w - 1):
                        self.grid[r:r + 2, c0:c0 + w] = P.T_SOLID
                    continue
                if a.style in ("tower", "keep", "chapel") and kind < 0.3:
                    # wall buttress with crenellation
                    w = int(rng.integers(2, 6))
                    left = rng.random() < 0.5
                    c0 = lw if left else W - rw - w
                    r0 = base + int(rng.integers(0, H - 6))
                    hgt = int(rng.integers(3, 10))
                    if self.free_rect(r0, c0, r0 + hgt, c0 + w - 1):
                        self.grid[r0:r0 + hgt + 1, c0:c0 + w] = P.T_SOLID
                    continue
                # generic rock mass, attached to a wall or floating
                w = int(rng.integers(2, 9))
                h = int(rng.integers(2, 7))
                if rng.random() < 0.6:
                    left = rng.random() < 0.5
                    c0 = lw if left else W - rw - w
                else:
                    c0 = int(rng.integers(lw, W - rw - w + 1))
                r0 = base + int(rng.integers(0, H - h))
                if self.free_rect(r0, c0, r0 + h - 1, c0 + w - 1):
                    self.grid[r0:r0 + h, c0:c0 + w] = P.T_SOLID
                    if a.material == "ice" and rng.random() < 0.4:
                        self.grid[r0, c0:c0 + w] = P.T_ICE
                    if a.material == "snow" and rng.random() < 0.5:
                        self.grid[r0, c0:c0 + w] = P.T_SNOW
            # route platform supports (pillars / hanging chains look)
        self.add_slopes()

    def add_slopes(self):
        """Bevel exposed corners of scenery into slopes (never route cells)."""
        rng = self.rng
        g = self.grid
        rows, cols = g.shape
        solid = (g == P.T_SOLID) | (g == -1)
        for r in range(1, rows - 1):
            s = self.screen_of_row(r)
            p = area_for(s).slopes
            if p <= 0:
                continue
            for c in range(1, cols - 1):
                if g[r, c] != P.T_SOLID or self.route_mask[r, c]:
                    continue
                if self.reserved[max(0, r - 1):r + 2, max(0, c - 1):c + 2].any():
                    continue
                up = g[r - 1, c] == 0
                down = g[r + 1, c] == 0
                left = g[r, c - 1] == 0
                right = g[r, c + 1] == 0
                corner = (up and left and solid[r, c + 1] and solid[r + 1, c]) or \
                         (up and right and solid[r, c - 1] and solid[r + 1, c]) or \
                         (down and left and solid[r, c + 1] and solid[r - 1, c]) or \
                         (down and right and solid[r, c - 1] and solid[r - 1, c])
                if corner and rng.random() < p:
                    g[r, c] = -1

    # -------------------------------------------------------------- finish
    def finalize(self):
        from .level import resolve_slopes

        screens = []
        for i in range(NS):
            r0 = (NS - 1 - i) * H
            screens.append(resolve_slopes(self.grid[r0:r0 + H].copy()))
        wind = [1 if area_for(i).wind else 0 for i in range(NS)]
        return Level(screens, wind, [-1] * NS, [-1] * NS, goal=self.goal,
                     start=(C.START_X, C.START_Y), name="builtin")


def verify_route(level, route):
    """Replay the construction route; returns the final Game."""
    g = Game(level)
    g.reset()
    g.settle()
    for kind, d, v in route:
        if kind == "walk":
            g.walk(d, v)
        else:
            g.jump(d, v)
    return g


def build(seed=7, verbose=False, scenery_retries=12):
    mb = MapBuilder(seed)
    mb.build_walls()
    mb.build_route(verbose=verbose)
    route = list(mb.route)
    grid_backup = mb.grid.copy()
    for attempt in range(scenery_retries):
        mb.grid[:] = grid_backup
        mb.add_scenery()
        level = mb.finalize()
        g = verify_route(level, route)
        if g.done:
            if verbose:
                print("route verified with scenery (attempt %d), %d ticks" % (attempt + 1, g.ticks))
            return level, route
        if verbose:
            print("scenery attempt %d broke the route at screen %d; retrying" % (
                attempt + 1, g.screen + 1))
    raise RuntimeError("could not place scenery without breaking the route")


def save(level, route, png_path, meta_path, seed):
    level.save_png(png_path)
    meta = {
        "seed": seed,
        "start": list(level.start),
        "goal": list(level.goal),
        "route": [list(a) for a in route],
        "note": "Original layout in the native Jump King hitbox atlas format. "
                "'route' is a verified macro-action solution (walk/jump, direction, ticks/charge).",
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
