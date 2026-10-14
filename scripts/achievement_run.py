"""Play the whole game from new games to all 48 achievements, inputs only.

Every run starts from the title menu's New Game / New Babe+ / Ghost of the
Babe and is played with button inputs alone: the community speed-runs where
they apply (base game, New Babe+), and routes found by ``jumpking.planner``
(BFS over jumps and walks on the exact physics) for everything else. The
king is never placed or teleported, and the profile carries over between
runs as in the game.

The playthrough is saved as a list of runs (mode + logged actions) that
``planner.replay`` reproduces exactly (``--replay``). Planning the whole
thing takes a while, so every finished stage is checkpointed and the script
resumes from the last one.

    python scripts/achievement_run.py --out runs/all48.json          # plan + play
    python scripts/achievement_run.py --replay jumpking/data/playthrough_all48.json

``--tas-dir`` points at local copies of the speed-run files (otherwise they
are downloaded); ``--stage NAME[:ARG...]`` runs one stage on top of the
checkpoint, for debugging.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jumpking import physics as P  # noqa: E402
from jumpking import routes  # noqa: E402
from jumpking.achievements import CODES  # noqa: E402
from jumpking.campaign import Campaign, Profile, WorldItem  # noqa: E402
from jumpking.planner import (Planner, near_point, on_screen, rect_touch, replay,  # noqa: E402
                              screens_between, tas_frames)

WIND_PERIOD = 768        # ticks of 17 ms: one period of the wind (13.056 s)
# New Babe+ without the crown sacrifice: screens 1-18, the teleport on screen 18
# into the Imp Room, then the New Babe+ world (0-based screen indices)
NBP_ROUTE = routes.ROUTES["new_babe_plus"]


class Playthrough:
    def __init__(self, out, tas_dir=None, verbose=True, plan_verbose=False):
        self.out = out
        self.plan_verbose = plan_verbose
        if tas_dir:
            os.environ["JUMPKING_TAS_DIR"] = tas_dir
        self.verbose = verbose
        self.ckpt = out + ".ckpt.json"
        self.runs = []
        self.done = []
        self.camp = Campaign()
        self.t0 = time.time()
        if os.path.isfile(self.ckpt):
            with open(self.ckpt, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            self.runs, self.done = d["runs"], d["done"]
            self.camp.profile = Profile.from_dict(d["profile"])
            self.log("resumed after stages %s" % self.done)

    # ------------------------------------------------------------ helpers
    def log(self, msg):
        if self.verbose:
            st = self.camp.profile
            print("[%6.0fs] %-60s achievements %2d/48  jumps %d falls %d" % (
                time.time() - self.t0, msg, len(st.achievements), st.jumps, st.falls), flush=True)

    def save(self, stage):
        self.done.append(stage)
        with open(self.ckpt, "w", encoding="utf-8") as fh:
            json.dump({"runs": self.runs, "done": self.done,
                       "profile": self.camp.profile.to_dict()}, fh)
        self.log("stage %s done" % stage)

    def new_run(self, name, mode, sacrifice_crown=True, before=()):
        """Pause-menu equipment changes (``before``), then a new game."""
        c = self.camp
        for item, on in before:
            c.equip(item, on)
        c.new_game(mode, sacrifice_crown=sacrifice_crown)
        pl = Planner(c, max_nodes=3000, verbose=self.plan_verbose)
        self.runs.append({"name": name, "mode": mode, "sacrifice_crown": sacrifice_crown,
                          "before": [list(b) for b in before], "actions": pl.log})
        self.log("run %d: %s (%s)" % (len(self.runs), name, mode))
        return pl

    def tas_inputs(self, key):
        return tas_frames(key)

    def climb(self, pl, screens, **kw):
        """Visit ``screens`` in order. After a fall back to an earlier screen
        of the list (the Snake Ring's slides), carry on from there."""
        screens = list(screens)
        start = self.camp.camera
        i = retries = 0
        while i < len(screens):
            k = screens[i]
            try:
                pl.follow([k], **kw)
            except RuntimeError:
                cur = self.camp.camera
                back = [j for j in range(i) if screens[j] == cur]
                if retries >= 8 or not (back or cur == start):
                    raise
                retries += 1
                i = back[-1] + 1 if back else 0
                self.log("  fell to screen %d, climbing again" % (cur + 1))
                continue
            self.log("  on screen %d" % (k + 1))
            i += 1

    def stand_with(self, pl, name):
        e = self.camp.entity(name)
        home = e.s.home_screen - 1
        touch = rect_touch(e.rect)
        pl.go(lambda s: int(s[P.S_SCREEN]) == home and touch(s))
        return e

    def talk(self, pl, name, times):
        e = self.stand_with(pl, name)
        codes = [c for c, n in getattr(e.s, "achievements", ()) if n <= times]
        for _ in range(200):
            # done when the talks are over, or (the NPC keeps talking while
            # the king stands there) when its achievements are in
            if e.times_talked >= times and (not e.busy or all(c in self.camp.unlocked for c in codes)):
                break
            pl.wait(600)
        else:
            raise RuntimeError("%s: only %d conversations" % (name, e.times_talked))
        self.log("  %s: %d conversations" % (name, e.times_talked))

    def buy(self, pl, name):
        e = self.stand_with(pl, name)
        for _ in range(20):
            if e.state.sold:
                break
            pl.wait(600)
        else:
            raise RuntimeError("%s did not sell (price %d)" % (name, e.price))
        self.log("  bought %s" % e.m.sale_item)

    def pick(self, pl, item):
        """Pick up the ``item`` lying in the world (if not picked up already)."""
        for e in self.camp.entities + self.camp.dynamic:
            if isinstance(e, WorldItem) and e.alive and e.state[0] == item and not e.display_only:
                pl.go(rect_touch(e.rect), touch=e.rect)
                return

    def goto(self, pl, screen, route=None):
        """Travel to ``screen``: along ``route`` (a list of screens in travel
        order, e.g. through teleports) if given, else straight up or down."""
        cur = self.camp.camera
        if cur == screen:
            return
        if route is None:
            step = 1 if screen > cur else -1
            path = list(range(cur + step, screen + step, step))
        else:
            i = route.index(cur)
            if screen in route[i + 1:]:
                j = route.index(screen, i + 1)
                path = route[i + 1:j + 1]
            else:
                j = max(k for k in range(i) if route[k] == screen)
                path = route[j:i][::-1]
        self.climb(pl, path)

    def chase(self, pl, bird_name, upto=None, route=None, only_screen=None):
        """Scare a bird off its perches (until perch ``upto``, or while it
        sits on ``only_screen``)."""
        try:
            b = self.camp.entity(bird_name)
        except KeyError:                    # gone for good
            return
        while b.alive and not b.state.is_done and (upto is None or b.state.home < upto):
            p = b.perch
            scr = p["screen"] - 1
            if only_screen is not None and scr != only_screen:
                return
            if self.camp.camera != scr:
                self.goto(pl, scr, route)
                continue                    # the bird may have fled meanwhile
            x, y = p["position"]
            home = b.state.home
            pl.go(near_point(x, y - scr * 360, p["trigger_distance"]))
            for _ in range(100):
                if b.state.home != home or b.state.is_done:
                    break
                pl.wait(30)
            else:
                hx, hy, hw, hh = self.camp.hitbox
                raise RuntimeError("%s stays on perch %d: king centre (%d, %d), bird (%.1f, %.1f), "
                                   "fleeing %s, trigger %s, camera %d" % (
                                       bird_name, home, hx + hw // 2, hy + hh // 2, b.x, b.y,
                                       b.fleeing, p["trigger_distance"], self.camp.camera))
            self.log("  %s fled perch %d (screen %d)" % (bird_name, home, scr + 1))

    # ------------------------------------------------------------- stages
    def stage(self, name, fn):
        if name in self.done:
            return
        fn()
        self.save(name)

    def speedrun(self, key, mode, name):
        pl = self.new_run(name, mode)
        if key == "base":
            pl.wait(WIND_PERIOD - 3)     # + 1 settle frame: the route's wind timing, one period later
        pl.play([("tas", key, 0, len(self.tas_inputs(key)))])
        assert self.camp.ended, "%s did not reach the babe" % name

    def talks(self):
        pl = self.new_run("talks", "normal")
        pl.wait(WIND_PERIOD - 3)
        frames = self.tas_inputs("base")
        c = self.camp
        k = 0
        while c.camera < 32:                     # the speed-run up to Chapel Perilous
            c.step(*frames[k])
            k += 1
        pl.log.append(["tas", "base", 0, k])
        pl.settle()
        self.goto(pl, 24)
        self.talk(pl, "hermit", 10)
        self.goto(pl, 14)
        self.talk(pl, "merchant", 4)
        self.goto(pl, 0)
        self.talk(pl, "old_man", 40)

    def raven_runs(self):
        pl = self.new_run("raven 1", "normal")
        rings = self.camp.profile.count("GoldRing")
        self.chase(pl, "raven")
        self.pick(pl, "GoldRing")             # dropped by the raven on its last perch
        assert self.camp.profile.count("GoldRing") == rings + 1
        actions = list(pl.log)
        for i in range(2, 11):
            pl = self.new_run("raven %d" % i, "normal")
            pl.play(actions)
        self.goto(pl, 14)
        self.buy(pl, "merchant")

    def gear_run(self):
        self.speedrun("base", "normal", "crown and shoes")

    def nbp_items(self):
        pl = self.new_run("silver and the Giant Boots", "new_babe_plus")
        route = NBP_ROUTE
        for e in sorted([e for e in self.camp.entities if isinstance(e, WorldItem)
                         and e.state[0] == "Silver"], key=lambda e: e.state[1]):
            self.goto(pl, e.state[1] - 1, route)
            pl.go(rect_touch(e.rect), touch=e.rect)
            self.log("  silver on screen %d" % e.state[1])
            if e.state[1] == 68:
                self.goto(pl, 70, route)
                r = [r for k, r in self.camp.content.achievement_hitboxes if k == "FIND_NOSE"][0][0]
                pl.go(rect_touch(r), touch=r)
        self.goto(pl, 70, route)
        self.buy(pl, "merchant_nbp")

    def white_raven_run(self):
        # the white raven starts on screen 1: keep the crown and climb from there
        pl = self.new_run("the ruby and the Snake Ring", "new_babe_plus", sacrifice_crown=False,
                          before=[("GiantBoots", False)])
        self.chase(pl, "white_raven", route=NBP_ROUTE)
        self.pick(pl, "Ruby")
        assert self.camp.profile.has("Ruby")
        self.goto(pl, 87, NBP_ROUTE)
        self.buy(pl, "snake")

    def reach(self, pl, k, no_splat=False):
        """Reach screen ``k`` (without a fall if possible when ``no_splat``)."""
        cur = self.camp.camera
        keep = None if abs(k - cur) > 1 else screens_between(min(cur, k) - 1, max(cur, k) + 1)
        goal = pl.stays(on_screen(k))
        if no_splat:
            try:
                pl.go(goal, keep, no_splat=True, budgets=(1, 4))
                return
            except RuntimeError:
                self.log("  screen %d needs a fall" % (k + 1))
        pl.go(goal, keep)

    @staticmethod
    def ghost_screens():
        """Screens in travel order: through Philosopher's Forest, down the
        Hole and the Bog, back up to the Tower of Antumbra. The Hole's
        screens 163 and 162 have nowhere to stop: the king falls from 164
        straight to 161."""
        return [k for k in routes.ROUTES["ghost_of_the_babe"] if k not in (162, 161)]

    def ghost_route(self, pl, upto=153, no_splat=False, hook=None):
        """Walk the Ghost of the Babe route screen by screen (through the
        teleports, down the Hole and the Bog), calling ``hook(screen)`` on
        each screen reached."""
        screens = self.ghost_screens()
        pos = self.__dict__.setdefault("ghost_pos", 0)
        while pos + 1 < len(screens):
            # the Hole drops through screens without a floor: if the next
            # screen has no standing place, aim one or two screens further
            for ahead in (1, 2, 3):
                k = screens[min(pos + ahead, len(screens) - 1)]
                try:
                    self.reach(pl, k, no_splat)
                    break
                except RuntimeError:
                    if ahead == 3:
                        raise
                    self.log("  no stop on screen %d" % (k + 1))
            pos += ahead
            self.ghost_pos = pos
            self.log("  on screen %d" % (k + 1))
            if hook:
                hook(k)
            if k == upto:
                break

    def ghost_fast(self):
        pl = self.new_run("ghost: two falls, under 30 minutes", "ghost_of_the_babe",
                          before=[("SnakeRing", False)])
        c = self.camp
        start, n0 = c.snapshot(), len(pl.log)
        self.ghost_pos = 0
        self.ghost_route(pl, no_splat=True)
        falls = c.run_stats()["falls"]
        if falls < 2:
            # the route itself falls fewer than two times: replay it after
            # the missing falls from screen 2 to screen 1
            self.log("  the route has %d falls; adding %d first" % (falls, 2 - falls))
            c.restore(start)
            del pl.log[n0:]
            self.falls_loop(pl, 2 - falls)
            pl.go(on_screen(0))
            self.ghost_pos = 0
            self.ghost_route(pl, no_splat=True)
        assert c.run_stats()["falls"] == 2, c.run_stats()
        pl.go(lambda s: P.check_endings(s, c.game.params) == 3, no_splat=True)
        pl.wait(1)
        assert self.camp.ended and self.camp.ending == 3

    def ghost_items(self):
        pl = self.new_run("ghost: the forest, the manor, the cats, the fly",
                          "ghost_of_the_babe")
        c = self.camp
        self.ghost_pos = 0
        route = self.ghost_screens()

        def items_here(k, kinds):
            # items on this screen or the two around it; one that cannot be
            # reached from here is tried again from the next screens
            for e in [e for e in c.entities if isinstance(e, WorldItem) and e.alive
                      and not e.display_only and e.state[0] in kinds
                      and abs(-((e.rect[1] + e.rect[3] // 2) // 360) - k) <= 1]:
                scr = -((e.rect[1] + e.rect[3] // 2) // 360)
                lo, hi = min(scr, k) - 1, max(scr, k) + 1
                try:
                    pl.go(rect_touch(e.rect), touch=e.rect, keep=lambda s: lo <= s[P.S_SCREEN] <= hi,
                          budgets=(1, 4))
                except RuntimeError:
                    self.log("  %s on screen %d: not from screen %d" % (e.state[0], scr + 1, k + 1))
                    continue
                self.log("  picked %s on screen %d" % (e.state[0], scr + 1))

        def hook(k):
            items_here(k, ("Shroom", "GnomeHat", "GhostFragment"))
            self.chase(pl, "tsuchinoko", only_screen=k, route=route)
            if k == 159:
                self.buy(pl, "owl_chars/owl_shroom_guy")
            if k == 114:                         # the fourth fragment: back to the mould man
                self.goto(pl, 108)
                self.buy(pl, "owl_chars/owl_mold_man")
                self.goto(pl, 114)
            if k == 118:
                self.chase(pl, "fly")
                self.goto(pl, 118)
            if k == 124:
                self.talk(pl, "owl_chars/cat_outsider", 1)
            if k == 141:
                self.talk(pl, "owl_chars/owl_boss_gargoyle", 1)

        self.ghost_route(pl, hook=hook)
        pl.go(lambda s: P.check_endings(s, c.game.params) == 3)
        pl.wait(1)
        assert c.ended and c.ending == 3

    def outfits(self):
        pl = self.new_run("outfits", "normal")
        for item in ("GnomeHat", "Tunic", "YellowShoes"):
            pl.play([("equip", item, 1)])
        pl.wait(2)
        for item in ("Shoes", "SnakeRing", "CapeOwl"):
            pl.play([("equip", item, 1)])
        pl.wait(2)
        pl.play([("equip", "GiantBoots", 1)])
        pl.wait(2)

    def item_run(self, mode, ending):
        pl = self.new_run("Giant Boots and Snake Ring: %s" % mode, mode, sacrifice_crown=False,
                          before=[("CrownNBP", False), ("GiantBoots", True), ("SnakeRing", True)])
        pl.wait(2)
        if mode == "normal":
            self.climb(pl, range(1, 43))
        elif mode == "new_babe_plus":
            self.climb(pl, NBP_ROUTE[1:])
        else:
            self.ghost_pos = 0
            self.ghost_route(pl)
        pl.go(lambda s: P.check_endings(s, self.camp.game.params) == ending)
        pl.wait(1)
        assert self.camp.ended and self.camp.ending == ending

    def drop(self, pl):
        """Climb to screen 2 and come down with a splat on screen 1."""
        c = self.camp
        start = len(pl.log)
        splats = c.game.state[P.S_SPLATS]
        pl.go(on_screen(1))
        pl.go(lambda s: s[P.S_SCREEN] == 0 and s[P.S_SPLATS] > splats)
        pl.settle()
        return pl.log[start:]

    def falls_loop(self, pl, n):
        """``n`` splats: the drop is planned until it repeats exactly, then
        the cycle is replayed."""
        c = self.camp

        def key():
            s = c.game.state
            return (round(s[P.S_X], 3), round(s[P.S_Y], 3), round(s[P.S_VX], 3), int(s[P.S_SCREEN]))

        done = 0
        prev = None
        while done < n:
            k0 = key()
            cycle = self.drop(pl)
            done += 1
            if prev is not None and key() == k0 == prev[0]:
                if n - done > 0:
                    pl.play([("repeat", n - done, cycle)])
                return
            prev = (key(), cycle)

    def counters(self):
        pl = self.new_run("jumps and falls", "normal")
        hop = [("frames", [[0, 1]] + [[0, 0]] * 11)]
        need = 20000 - self.camp.profile.jumps
        if need > 0:
            pl.play([("repeat", need, hop)])
        if self.camp.profile.falls < 1000:
            self.falls_loop(pl, 1000 - self.camp.profile.falls)

    # --------------------------------------------------------------- run
    def run(self):
        self.stage("speedrun", lambda: self.speedrun("base", "normal", "speed-run"))
        self.stage("talks", self.talks)
        self.stage("raven", self.raven_runs)
        self.stage("gear", self.gear_run)
        self.stage("nbp", lambda: self.speedrun("nbp", "new_babe_plus", "New Babe+ speed-run"))
        self.stage("nbp_items", self.nbp_items)
        self.stage("white_raven", self.white_raven_run)
        self.stage("ghost_fast", self.ghost_fast)
        self.stage("ghost_items", self.ghost_items)
        self.stage("outfits", self.outfits)
        for mode, ending in (("normal", 1), ("new_babe_plus", 2), ("ghost_of_the_babe", 3)):
            self.stage("items_" + mode, lambda m=mode, e=ending: self.item_run(m, e))
        self.stage("counters", self.counters)
        missing = [c for c in CODES if c not in self.camp.profile.achievements]
        with open(self.out, "w", encoding="utf-8") as fh:
            json.dump({"runs": self.runs}, fh)
        self.log("saved %s; missing: %s" % (self.out, missing or "none"))
        return missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="all48.json")
    ap.add_argument("--replay")
    ap.add_argument("--tas-dir", help="local copies of the speed-run .tas files")
    ap.add_argument("--stage", help="run only this stage (after the checkpointed ones)")
    ap.add_argument("--plan-verbose", action="store_true", help="log every search")
    args = ap.parse_args()
    if args.tas_dir:
        os.environ["JUMPKING_TAS_DIR"] = args.tas_dir
    if args.replay:
        with open(args.replay, "r", encoding="utf-8") as fh:
            runs = json.load(fh)["runs"]
        c = Campaign()
        t0 = time.time()

        def progress(i, run, camp):
            print("[%5.0fs] run %2d %-50s %2d/48" % (time.time() - t0, i + 1, run["name"],
                                                     len(camp.unlocked)), flush=True)
        replay(c, runs, progress)
        missing = [code for code in CODES if code not in c.unlocked]
        print("missing:", missing or "none")
        sys.exit(1 if missing else 0)
    pt = Playthrough(args.out, args.tas_dir, plan_verbose=args.plan_verbose)
    if args.stage:                      # e.g. ghost_items, item_run:normal:1
        name, *rest = args.stage.split(":")
        try:
            getattr(pt, name)(*[int(r) if r.isdigit() else r for r in rest])
        finally:                        # the runs so far, for replaying up to a failure
            with open(args.out + ".stage.json", "w", encoding="utf-8") as fh:
                json.dump({"runs": pt.runs}, fh)
        return
    sys.exit(1 if pt.run() else 0)


if __name__ == "__main__":
    main()
