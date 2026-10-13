"""The whole game on top of the physics: runs, items, NPCs and achievements.

``Campaign`` plays Jump King the way the game does, frame by frame, with
everything that is needed to unlock all 48 Steam achievements:

* the shared 169-screen world with its three babes (normal ending, New
  Babe+, Ghost of the Babe), new games in each mode (including the New Babe+
  crown sacrifice that starts in the Imp Room);
* world items (gold rings, silver coins, mushrooms, ghost fragments, the gnome
  hat, ...), the inventory and the wearable skins with their layers, the Giant
  Boots and the Snake Ring with their physical effects;
* the NPCs and merchants (dialogue pools, conversation timing, prices fixed per
  run from your inventory, sales), the ravens/birds/the fly that flee along
  their paths, achievement hitboxes and illusory walls;
* run statistics and the permanent profile (inventory, equipment, story flags,
  lifetime jumps/falls, achievements), so that multi-run achievements (Big
  Spender, Giant Leaps, Double King, ...) work as in the game.

The per-frame order follows the game: GameLoop (achievement clock, win check),
the run manager, the player, then props, items, merchants, NPCs and ravens.
The NPC, merchant and raven logic is a port of their behaviour trees
(``bt.py``); dialogue lengths come from the game's own text. The content is
read from your local install (``content.GameContent``); nothing of it is
shipped with this package.

Example::

    from jumpking.campaign import Campaign
    c = Campaign()
    c.new_game("normal")
    for _ in range(600):
        c.step(0, False)          # stand next to the old man...
    print(c.unlocked)             # ...and listen
"""
import copy

import numpy as np

from . import bt
from . import constants as C
from . import physics as P
from .achievements import BY_CODE, CODES
from .game import Game
from .level import Level
from .routes import START_POSITIONS

MODES = ("normal", "new_babe_plus", "ghost_of_the_babe")
ENDING_OF_MODE = {"normal": 1, "new_babe_plus": 2, "ghost_of_the_babe": 3}
ITEMS = ("Crown", "Shoes", "GoldRing", "CrownNBP", "Ruby", "SnakeRing", "GiantBoots", "Silver",
         "Cap", "GnomeHat", "Tunic", "YellowShoes", "CrownOwl", "CapeOwl", "BabeGhostItem",
         "GhostFragment", "Shroom", "BugNote")
# StoryEventFlags
COMPLETED_NORMAL = "CompletedNormalGame"
COMPLETED_NBP = "CompletedNBP"
STARTED_NBP = "StartedNBP"
COMPLETED_GHOST = "CompletedGhost"
STARTED_GHOST = "StartedGhost"

TICK_SECONDS = C.TARGET_ELAPSED_SECONDS     # the in-game timer advances 17 ms per frame
WHITE_RAVEN_NBP_HOME = 17                  # NBPDecisionState.MoveWhiteBird


def _intersects(a, b):
    """MonoGame Rectangle.Intersects (x, y, w, h)."""
    return (b[0] < a[0] + a[2] and a[0] < b[0] + b[2]
            and b[1] < a[1] + a[3] and a[1] < b[1] + b[3])


def _f32(v):
    return np.float32(v)


# ---------------------------------------------------------------------------
# Permanent profile (SavesPerma) and per-run state (Saves/CombinedSave)
# ---------------------------------------------------------------------------
class Profile:
    """Everything that survives a new game: inventory, equipment, story
    flags, lifetime statistics and the unlocked achievements."""

    def __init__(self):
        self.inventory = {}          # item -> count
        self.item_options = {}       # item -> equipped (ItemEquipOptions; absent = True)
        self.applied_skins = []      # worn skins, in the order they were put on
        self.flags = set()           # StoryEventFlags
        self.jumps = 0               # lifetime statistics (PermanentPlayerStats)
        self.falls = 0
        self.ticks = 0
        self.attempts = 1
        self.times_won = 0
        self.snapshot = {"jumps": 0, "falls": 0, "ticks": 0}   # at the start of the attempt
        self.achievements = {}       # code -> (attempt, run tick) of the unlock

    def copy(self):
        return copy.deepcopy(self)

    # persistence (a JSON "SavesPerma")
    _FIELDS = ("inventory", "item_options", "applied_skins", "jumps", "falls", "ticks",
               "attempts", "times_won", "snapshot")

    def to_dict(self):
        d = {k: copy.deepcopy(getattr(self, k)) for k in self._FIELDS}
        d["flags"] = sorted(self.flags)
        d["achievements"] = {k: list(v) for k, v in self.achievements.items()}
        return d

    @classmethod
    def from_dict(cls, d):
        p = cls()
        for k in cls._FIELDS:
            if k in d:
                setattr(p, k, copy.deepcopy(d[k]))
        p.flags = set(d.get("flags", ()))
        p.achievements = {k: tuple(v) for k, v in d.get("achievements", {}).items()}
        return p

    def save(self, path):
        import json

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    @classmethod
    def load(cls, path):
        import json

        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # InventoryManager
    def count(self, item):
        return self.inventory.get(item, 0)

    def has(self, item):
        return self.inventory.get(item, 0) > 0

    def add(self, item, n=1):
        self.inventory[item] = max(0, self.inventory.get(item, 0) + n)

    def add_once(self, item):
        if not self.has(item):
            self.add(item)

    def item_enabled(self, item):
        return self.item_options.get(item, True)


class RunState:
    """CombinedSaveFile: what a new game resets."""

    def __init__(self, world_items):
        self.old_men = {}            # name -> OldManState
        self.merchants = {}          # name -> MerchantState
        self.ravens = {}             # name -> RavenState
        self.world_items = list(world_items)   # (item, screen, x, y) still in the world
        self.wear_giant_boots = False   # FullRunSave
        self.wear_snake_ring = False


class OldManState:
    def __init__(self, name):
        self.name = name
        self.best_screen = 0
        self.quote_pool = []
        self.say_intro = True
        self.intro_quote = None
        self.times_talked = 0
        self.flag_quotes = None      # [(flag, [quotes])]

    def is_empty(self):
        return len(self.quote_pool) == 0 and not self.say_intro

    def consume(self, flags):
        self.times_talked += 1
        if self.flag_quotes is not None:
            for flag, quotes in self.flag_quotes:
                if flag in flags and quotes:
                    self.say_intro = False
                    return quotes.pop(0)
        if self.say_intro:
            self.say_intro = False
            return self.intro_quote
        if not self.quote_pool:
            raise RuntimeError("not enough quotes")
        return self.quote_pool.pop()


class MerchantState:
    def __init__(self):
        self.has_introduced = False
        self.sold = False
        self.required_gold = -1


class RavenState:
    def __init__(self):
        self.home = 0
        self.is_done = False
        self.picked_up_gold = False

    def goto_next(self, n_positions):
        self.home += 1
        if self.home == n_positions:
            self.is_done = True
        self.home = min(self.home, n_positions - 1)


class QuoteIndexer:
    def __init__(self, quotes):
        self.quotes = quotes or []
        self.i = 0

    def next(self):
        q = self.quotes[self.i]
        self.i = (self.i + 1) % len(self.quotes)
        return q


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------
class Entity:
    alive = True

    def update(self):
        pass


class OldManComp:
    """OldManComp: the quote pool, fed by new screens, items and falls."""

    def __init__(self, camp, settings):
        self.camp = camp
        self.s = settings
        self.looping = QuoteIndexer(settings.looping_lines) if settings.looping_lines else None
        st = camp.run.old_men.get(settings.name)
        if st is None:
            st = OldManState(settings.name)
            camp.run.old_men[settings.name] = st
        if st.say_intro and settings.flag_quotes is not None:
            st.flag_quotes = [(f, list(q)) for f, q in settings.flag_quotes]
        st.intro_quote = settings.intro_quote
        if settings.intro_quote is None:
            st.say_intro = False
        self.state = st
        self.current_screen = 0
        self.screens = {s: q for s, q in settings.screens}

    def update(self):
        num = self.camp.camera + 1
        if self.current_screen == num:
            return
        self.current_screen = num
        st = self.state
        if num > st.best_screen and num in self.screens:
            st.best_screen = num
            st.quote_pool.extend(reversed(self.screens[num]))
        if self.s.items:
            for item in self.camp.enabled_items():
                for it, quotes in self.s.items:
                    if it == item:
                        st.quote_pool.extend(reversed(quotes))
                        break
        if self.s.falls:
            falls = self.camp.run_falls
            for start, end, quotes in self.s.falls:
                if start >= falls and falls < end:   # the game's condition
                    st.quote_pool.extend(reversed(quotes))
                    break

    def has_quotes(self):
        if self.state.is_empty():
            return self.looping is not None
        return True

    def said_first_line(self):
        return not self.state.say_intro

    def consume(self):
        if self.state.is_empty():
            return self.looping.next()
        return self.state.consume(self.camp.profile.flags)


class _Talk:
    """The dialogue state shared by an NPC's nodes: the blackboard of
    FetchQuote / TargetLine / SayLine and the talking speed."""

    def __init__(self, camp, comp, talking_speed):
        self.camp = camp
        self.comp = comp
        self.bb = {"index": 0, "quote": None, "line": ""}
        self.speed = talking_speed if talking_speed else 0.05

    def fetch_quote(self):
        """FetchQuote."""
        if not self.comp.has_quotes():
            return bt.FAILURE
        self.bb["quote"] = self.comp.consume()
        return bt.SUCCESS

    def set_index0(self):
        self.bb["index"] = 0


class TargetLine(bt.Node):
    """Puts the next line of the current quote on the blackboard."""

    __slots__ = ("talk", "index")

    def __init__(self, talk):
        super().__init__()
        self.talk = talk
        self.index = 0

    def on_new_run(self):
        bb = self.talk.bb
        self.index = bb["index"]
        bb["index"] = self.index + 1

    def my_run(self, tick):
        quote = self.talk.bb["quote"] or []
        if len(quote) <= self.index:
            return bt.FAILURE
        self.talk.bb["line"] = self.talk.camp.content.line_text(quote[self.index])
        return bt.SUCCESS


class SayLine(bt.Node):
    """Reveals the line one character per ``talking_speed`` seconds."""

    __slots__ = ("talk", "n", "length", "timer")

    def __init__(self, talk):
        super().__init__()
        self.talk = talk
        self.n = self.length = 0
        self.timer = bt.Timer(talk.speed)

    def _reset(self):
        self.length = len(self.talk.bb["line"])
        self.n = 0
        self.timer = bt.Timer(self.talk.speed)

    def on_new_run(self):
        self._reset()

    def resume_run(self):
        self._reset()

    def my_run(self, tick):
        if self.n == self.length:
            return bt.SUCCESS
        if self.timer.update():
            self.n += 1
        return bt.RUNNING


class TalkAchievementCheck(bt.Node):
    __slots__ = ("talk", "code", "times")

    def __init__(self, talk, code, times):
        super().__init__()
        self.talk = talk
        self.code = code
        self.times = times

    def my_run(self, tick):
        if self.talk.comp.state.times_talked == self.times:
            self.talk.camp.unlock(self.code)
        return bt.SUCCESS


class OldMan(Entity):
    """OldManEntity (the logic half of its behaviour tree)."""

    def __init__(self, camp, settings):
        self.camp = camp
        self.s = settings
        self.comp = OldManComp(camp, settings)
        self.rect = settings.trigger_rect()
        self.home = settings.home_screen - 1
        talk = _Talk(camp, self.comp, settings.talking_speed)
        self.talk = talk
        ok = bt.Constant(bt.SUCCESS)           # animation-state nodes
        hitbox = bt.Check(self.touching)
        seq = bt.Sequence(
            bt.Selector(bt.Sequence(bt.Check(self.comp.said_first_line), ok), ok),
            bt.Check(self.on_home_screen),
            hitbox)
        if settings.talk_delay > 0.5:
            seq.add(bt.Pause(0.5), hitbox, bt.Pause(settings.talk_delay - 0.5))
        else:
            seq.add(bt.Pause(0.5), hitbox)
        lines = bt.Sequence(ok, TargetLine(talk), SayLine(talk), ok, bt.Pause(2.2))
        seq2 = bt.Sequence(bt.Selector(bt.Call(talk.fetch_quote)),
                           bt.Call(talk.set_index0),
                           bt.Repeat(lines, bt.FAILURE), ok,
                           *[TalkAchievementCheck(talk, c, n) for c, n in settings.achievements])
        seq.add(seq2, ok, bt.Pause(30.0))
        seq4 = bt.Sequence(bt.Invert(bt.Check(self.comp.has_quotes)), ok)
        self.tree = bt.Manager(bt.SelectorRunning(seq4, seq))

    def touching(self):
        return _intersects(self.rect, self.camp.hitbox)

    def on_home_screen(self):
        return self.camp.camera == self.home

    def update(self):
        self.comp.update()
        tree = self.tree
        if tree.root.last_result != bt.RUNNING and self.camp.camera != self.home:
            # the tree would fail before touching any state: skip it
            tree.tick += 1
            return
        tree.run()

    @property
    def busy(self):
        return self.tree.root.last_result == bt.RUNNING

    @property
    def times_talked(self):
        return self.comp.state.times_talked


class TargetMerchLine(bt.Node):
    __slots__ = ("merchant", "kind")

    def __init__(self, merchant, kind):
        super().__init__()
        self.merchant = merchant
        self.kind = kind

    def my_run(self, tick):
        m = self.merchant
        if self.kind == "sale":
            q = m.sale.next()
        elif self.kind == "no_gold":
            q = m.no_gold.next()
        elif self.kind == "on_sale":
            q = m.m.sell_quote
        else:
            q = m.sold_lines.next()
        m.talk.bb["quote"] = q
        return bt.SUCCESS


class Merchant(Entity):
    """MerchantEntity (the logic half of its behaviour tree)."""

    def __init__(self, camp, settings):
        self.camp = camp
        self.m = settings
        s = settings.settings
        self.s = s
        self.comp = OldManComp(camp, s)
        st = camp.run.merchants.get(s.name)
        if st is None:
            st = MerchantState()
            camp.run.merchants[s.name] = st
        if st.required_gold < 0:
            st.required_gold = (camp.profile.count(settings.currency_type)
                                + settings.price_increase + settings.base_price)
        self.state = st
        self.sale = QuoteIndexer(settings.sale_lines)
        self.sold_lines = QuoteIndexer(settings.sold_lines)
        self.no_gold = QuoteIndexer(settings.no_gold_lines)
        self.rect = s.trigger_rect()
        self.price_text = str(st.required_gold)
        talk = _Talk(camp, self.comp, s.talking_speed)
        self.talk = talk
        ok = bt.Constant(bt.SUCCESS)
        hitbox = bt.Check(self.touching)
        ach = bt.Sequence(*[TalkAchievementCheck(talk, c, n) for c, n in s.achievements])
        seq3 = bt.Sequence(ok, TargetLine(talk), bt.Call(self.format_line), SayLine(talk), ok,
                           bt.Pause(2.2))
        seq2 = bt.Sequence(bt.Call(talk.set_index0), bt.Repeat(seq3, bt.FAILURE), ok)
        seq4 = bt.Sequence(bt.Call(talk.fetch_quote), seq2)
        sel2 = bt.Selector(seq4, bt.Static(seq4, bt.SUCCESS))
        seq5 = bt.Sequence(bt.Check(self.can_sell), TargetMerchLine(self, "on_sale"), seq2,
                           bt.Call(self.sell))
        seq6 = bt.Sequence(TargetMerchLine(self, "no_gold"), seq2, sel2)
        p_child2 = bt.SelectorRunning(bt.Invert(hitbox), seq5, seq6)
        seq7 = bt.Sequence(bt.Invert(bt.Check(self.introduced)), sel2, bt.Call(self.introduce),
                           p_child2, ach)
        seq8 = bt.Sequence(TargetMerchLine(self, "sale"), seq2, p_child2, ach)
        seq9 = bt.Sequence(bt.Check(self.has_sold), TargetMerchLine(self, "sold"), seq2, sel2, ach)
        self.tree = bt.Manager(bt.Sequence(ok, hitbox, bt.Selector(seq9, seq7, seq8), ok,
                                           bt.Pause(30.0)))

    # leaves
    def touching(self):
        return _intersects(self.rect, self.camp.hitbox)

    def format_line(self):
        """FormatLine("[gold_count]", price fixed when the merchant was made)."""
        bb = self.talk.bb
        bb["line"] = bb["line"].replace("[gold_count]", self.price_text)

    def can_sell(self):
        return self.camp.profile.count(self.m.currency_type) >= self.state.required_gold

    def introduced(self):
        return self.state.has_introduced

    def introduce(self):
        self.state.has_introduced = True

    def has_sold(self):
        return self.state.sold

    def sell(self):
        """MerchantComp.OnSell."""
        camp, m, st = self.camp, self.m, self.state
        st.sold = True
        camp.run.world_items = [w for w in camp.run.world_items if w[0] != m.sale_item]
        camp.remove_display_items(m.sale_item)
        camp.profile.add_once(m.sale_item)
        if m.sale_item in camp.content.skins:
            camp.enable_skin(m.sale_item)
        camp.profile.add(m.currency_type, -st.required_gold)
        for code, amount in m.sale_achievements:
            if st.required_gold >= amount:
                camp.unlock(code)
        camp._sync_items()

    def update(self):
        self.comp.update()
        tree = self.tree
        if tree.root.last_result != bt.RUNNING and not self.touching():
            tree.tick += 1
            return
        tree.run()

    @property
    def price(self):
        return self.state.required_gold

    @property
    def busy(self):
        return self.tree.root.last_result == bt.RUNNING

    @property
    def times_talked(self):
        return self.comp.state.times_talked


class Raven(Entity):
    """RavenEntity + RavenComp + RavenFlee: a bird that flees to its next
    perch when the king comes within its trigger distance."""

    def __init__(self, camp, settings):
        self.camp = camp
        self.s = settings
        st = camp.run.ravens.get(settings.name)
        if st is None:
            st = RavenState()
            camp.run.ravens[settings.name] = st
        self.state = st
        self.fleeing = False
        self.spawned_gold = False
        self.vx = self.vy = _f32(0.0)
        self.buffer_tick = 0
        self._update_transform()

    @property
    def n(self):
        return len(self.s.positions)

    def _real(self, i):
        p = self.s.positions[i]
        x, y = p["position"]
        return x, y - (p["screen"] - 1) * 360

    @property
    def perch(self):
        return self.s.positions[self.state.home]

    def _update_transform(self):
        x, y = self._real(self.state.home)
        self.x, self.y = _f32(x), _f32(y)
        if self.state.is_done:
            self.alive = False
        if self.state.home >= self.n - 1 and not self.state.picked_up_gold and not self.spawned_gold:
            self.spawned_gold = True
            if self.s.item and self.s.item != "NULL":
                lx, ly = self._real(self.n - 1)
                self.camp.spawn_world_item((self.s.item, 0, lx - 5, ly + 8))
            self.state.picked_up_gold = True

    def fly_direction(self):
        """RavenComp.GetFlyDirection: 1 = down to a lower screen, -1 = up."""
        if self.state.home >= self.n - 1:
            return -1
        if self.perch["screen"] > self.s.positions[self.state.home + 1]["screen"]:
            return 1
        return -1

    def _flee_start(self):
        self.buffer_tick = 0
        self.vx = _f32(0.0)
        self.vy = _f32(2 * self.fly_direction())
        if self.fly_direction() >= 0:
            d = 1 if self.perch["fly_direction"] == "Right" else -1
            self.vx = _f32(d * 3.5)
            self.vy = _f32(6.0)

    def _on_flee(self):
        st = self.state
        st.goto_next(self.n)
        if st.home == self.n - 1 and self.s.texture == "raven":
            self.camp.unlock("CHASE_BIRD_PAST_SNOWSTORM")
        if st.is_done and self.s.texture == "fly":
            self.camp.unlock("CHASE_AWAY_BUG")
        self._update_transform()

    def update(self):
        camp = self.camp
        if not self.fleeing:
            hx, hy, hw, hh = camp.hitbox
            cx, cy = _f32(hx + hw // 2), _f32(hy + hh // 2)
            dx = _f32(cx - self.x)
            dy = _f32(cy - self.y)
            d2 = _f32(_f32(dx * dx) + _f32(dy * dy))
            trig = self.perch["trigger_distance"]
            if not d2 < _f32(_f32(trig) * _f32(trig)):
                return
            self.fleeing = True
            self._flee_start()
        # RavenFlee.MyRun: done once the bird leaves the camera's view
        vy = _f32(self.y + camp.camera * 360)
        if not (0 <= self.x < 480 and 0 <= vy < 360):
            self.fleeing = False
            self._on_flee()
            return
        self.buffer_tick += 1
        if self.buffer_tick >= 2:
            self.buffer_tick = 0
            d = 1 if self.perch["fly_direction"] == "Right" else -1
            if self.fly_direction() < 0:
                self.vx = _f32(self.vx + _f32(d * 0.5))
                self.vy = _f32(self.vy * _f32(1.15))
            else:
                self.vx = _f32(self.vx + _f32(d * 0.3))
                self.vy = _f32(self.vy * _f32(1.1))
        self.x = _f32(self.x + _f32(self.vx / _f32(2.0)))
        self.y = _f32(self.y + _f32(self.vy / _f32(2.0)))


class WorldItem(Entity):
    """BasicWorldItemEntity + WorldItemComp (picked up on touch), or
    WorldItemEntityDisplay (shown, never picked up) for the shop items."""

    DISPLAY_ONLY = ("GiantBoots", "YellowShoes")

    def __init__(self, camp, state):
        self.camp = camp
        self.state = state
        item, screen, x, y = state
        if screen != 0:
            y = y - (screen - 1) * 360
        self.display_only = item in self.DISPLAY_ONLY
        w, h = camp.content.item_sizes.get(item, (0, 0))
        if item == "BabeGhostItem":
            # WorldItemFactory: the whole screen below the item's screen
            yy = int(y / 360) * 360 + 360
            self.rect = (0, yy, 480, 360)
        else:
            self.rect = (x - int(w * 0.5), y - int(h * 1.0), w, h)

    def update(self):
        camp = self.camp
        if self.display_only:
            if self.state not in camp.run.world_items:
                self.alive = False
            return
        if _intersects(camp.hitbox, self.rect):
            self.alive = False
            camp.pick_up(self.state)


class Trigger(Entity):
    """AchievementTrigger: unlocks its achievement when the king touches it."""

    def __init__(self, camp, code, rects):
        self.camp = camp
        self.code = code
        self.rects = rects

    def update(self):
        hb = self.camp.hitbox
        for r in self.rects:
            if _intersects(r, hb):
                self.camp.unlock(self.code)
                self.alive = False
                return


# ---------------------------------------------------------------------------
# The campaign
# ---------------------------------------------------------------------------
class Campaign:
    """The full game: physics + world entities + profile + achievements.

    ``level`` defaults to the original world; ``content`` to the content of
    your local install. ``profile`` (a ``Profile``) carries everything that
    persists between runs; pass one around to continue a save."""

    def __init__(self, level=None, content=None, profile=None, legacy_teleport=False):
        from .content import GameContent

        if level is None:
            level = Level.original()
        if content is None:
            content = GameContent.load()
        self.level = level
        self.content = content
        self.game = Game(level, legacy_teleport=legacy_teleport)
        self.profile = profile if profile is not None else Profile()
        self.mode = None
        self.run = None
        self.entities = []
        self.new_achievements = []
        self.ended = False
        self.ending = 0
        self.win_stats = None
        self._start_skins()

    # ------------------------------------------------------------ skins
    def _start_skins(self):
        """SkinManager.AddSkinSprite: owned and enabled items are worn."""
        if self.profile.applied_skins:
            return
        for item in self.content.skins:
            if self.profile.has(item) and self.profile.item_enabled(item):
                self.enable_skin(item)

    def wearing(self, item):
        return item in self.profile.applied_skins

    def enable_skin(self, item):
        """SkinManager.EnableSkin: put on ``item``, taking off whatever shares
        one of its layers."""
        layers = self.content.skins.get(item)
        if layers is None:
            return False
        for layer in layers:
            for worn in list(reversed(self.profile.applied_skins)):
                if layer in self.content.skins.get(worn, ()):
                    self._remove_skin(worn)
                    break
        self.profile.applied_skins.append(item)
        self.profile.item_options[item] = True
        return True

    def _remove_skin(self, item):
        if item in self.profile.applied_skins:
            self.profile.applied_skins.remove(item)
            self.profile.item_options[item] = False

    def equip(self, item, on=True):
        """Put on / take off an owned item (the pause-menu possessions)."""
        if item not in self.content.skins:
            raise ValueError("%s is not wearable" % item)
        if on:
            if not self.profile.has(item):
                raise ValueError("you do not own %s" % item)
            self.enable_skin(item)
        else:
            self._remove_skin(item)
        self._sync_items()

    def enabled_items(self):
        return [i for i, on in self.profile.item_options.items() if on]

    def _sync_items(self):
        s = self.game.state
        s[P.S_BOOTS] = 1.0 if self.wearing("GiantBoots") else 0.0
        ring = self.profile.has("SnakeRing") and self.profile.item_enabled("SnakeRing")
        s[P.S_RING] = 1.0 if ring else 0.0

    # --------------------------------------------------------- new game
    def new_game(self, mode="normal", sacrifice_crown=True):
        """Start a new run in ``mode`` (normal, new_babe_plus,
        ghost_of_the_babe). In New Babe+ with the crown in the inventory the
        game offers to sacrifice it (``sacrifice_crown=True``) and start in the
        Imp Room; with the New Babe+ crown worn the Imp Room start is free."""
        if mode not in MODES:
            raise ValueError("mode must be one of %s" % (MODES,))
        prof = self.profile
        # AchievementManager.OnNewGame -> OnGiveUp: every new game counts an attempt
        prof.attempts += 1
        self._take_snapshot()
        self.mode = mode
        prof.flags.discard(STARTED_NBP)
        prof.flags.discard(STARTED_GHOST)
        if mode == "new_babe_plus":
            prof.flags.add(STARTED_NBP)
        elif mode == "ghost_of_the_babe":
            prof.flags.add(STARTED_GHOST)
        self.run = RunState(self.content.world_items)
        start = START_POSITIONS["start"]
        white = None
        if mode == "new_babe_plus":
            # NBPDecisionState
            if self.wearing("CrownNBP"):
                start = START_POSITIONS["imp_room"]
                white = WHITE_RAVEN_NBP_HOME
            elif prof.has("Crown") and sacrifice_crown:
                start = START_POSITIONS["imp_room"]
                white = WHITE_RAVEN_NBP_HOME
                prof.add("Crown", -1)
                self._remove_skin("Crown")
        else:
            white = WHITE_RAVEN_NBP_HOME
        if white is not None:
            st = RavenState()
            st.home = white
            self.run.ravens["white_raven"] = st
        self.ended = False
        self.ending = 0
        self.win_stats = None
        self.game.reset(*start)
        self.game.state[P.S_FACING] = -1 if start == START_POSITIONS["imp_room"] else 1
        self._sync_items()
        self.frames = 0
        self.run_jumps = 0
        self.run_falls = 0
        self._spawn_entities()
        return self

    def _take_snapshot(self):
        p = self.profile
        p.snapshot = {"jumps": p.jumps, "falls": p.falls, "ticks": p.ticks}

    def _spawn_entities(self):
        """IntroState.InitializeEntities + PropManager.Load, in creation order."""
        c = self.content
        ents = []
        for r in c.ravens:
            ents.append(Raven(self, r))
        for o in c.old_men:
            ents.append(OldMan(self, o))
        for m in c.merchants:
            ents.append(Merchant(self, m))
        for w in self.run.world_items:
            ents.append(WorldItem(self, w))
        walls = sorted(c.illusory_walls, key=lambda cw: -cw[1][0][1])
        for code, rects in walls:
            ents.append(Trigger(self, code, rects))
        for code, rects in c.achievement_hitboxes:
            ents.append(Trigger(self, code, rects))
        self.entities = [e for e in ents if e.alive]
        self.dynamic = []            # entities created during the run update first

    def spawn_world_item(self, state):
        self.run.world_items.append(state)
        self.dynamic.append(WorldItem(self, state))

    def remove_display_items(self, item):
        for e in self.entities + self.dynamic:
            if isinstance(e, WorldItem) and e.state[0] == item:
                e.alive = False

    def pick_up(self, state):
        """WorldItemComp.OnPickup."""
        item = state[0]
        self.profile.add(item)
        if item in self.content.skins:
            self.enable_skin(item)
        if state in self.run.world_items:
            self.run.world_items.remove(state)
        self._sync_items()

    # ------------------------------------------------------ achievements
    def unlock(self, code):
        if code not in BY_CODE:
            raise KeyError(code)
        if code not in self.profile.achievements:
            self.profile.achievements[code] = (self.profile.attempts, self.frames)
            self.new_achievements.append(code)

    @property
    def unlocked(self):
        return set(self.profile.achievements)

    @property
    def locked(self):
        return [c for c in CODES if c not in self.profile.achievements]

    def _check_skin_combo(self):
        w = self.wearing
        if w("SnakeRing") and w("GiantBoots"):
            self.unlock("WEAR_GIANTBOOTS_AND_ICERING")
        if w("GnomeHat") and w("Tunic") and w("YellowShoes"):
            self.unlock("WEAR_TUNIC_YELLOWSHOES_AND_GNOMEHAT")
        if w("Shoes") and w("SnakeRing") and w("CapeOwl"):
            self.unlock("WEAR_SHOES_RING_AND_OWLCAPE")
        if w("GnomeHat"):
            self.unlock("GET_GNOME_HAT")

    def _on_jump(self):
        p = self.profile
        p.jumps += 1
        self.run_jumps += 1
        if p.jumps >= 1000:
            self.unlock("JUMP_1000")
        if p.jumps >= 10000:
            self.unlock("JUMP_10000")
        if p.jumps >= 20000:
            self.unlock("JUMP_100000")

    def _on_fall(self):
        p = self.profile
        p.falls += 1
        self.run_falls += 1
        if p.falls == 100:
            self.unlock("FALL_100")
        if p.falls == 1000:
            self.unlock("FALL_1000")

    def run_stats(self):
        p = self.profile
        ticks = p.ticks - p.snapshot["ticks"]
        return {"jumps": p.jumps - p.snapshot["jumps"], "falls": p.falls - p.snapshot["falls"],
                "ticks": ticks, "seconds": ticks * TICK_SECONDS}

    def _on_victory(self, ending):
        """AchievementManager.OnVictory, then the ending cutscene's rewards."""
        p = self.profile
        stats = self.run_stats()
        self.win_stats = stats
        p.attempts += 1
        p.times_won += 1
        minutes = stats["seconds"] / 60.0
        boots, ring = self.run.wear_giant_boots, self.run.wear_snake_ring
        if ending == 1:
            if minutes < 60:
                self.unlock("WIN_TIME_60")
            if minutes < 15:
                self.unlock("WIN_TIME_15")
            self.unlock("WIN")
            if stats["falls"] == 0:
                self.unlock("WIN_NO_FALLS")
            if boots:
                self.unlock("WIN_GIANTBOOTS_REGULAR")
            if ring:
                self.unlock("WIN_ICERING_REGULAR")
            raven = self.run.ravens.get("raven")
            if (raven.home if raven else 0) == 2:
                self.unlock("WIN_TOUCH_CROW_2")
        elif ending == 2:
            self.unlock("WIN_NBP")
            if stats["falls"] == 0:
                self.unlock("WIN_NBP_NO_FALLS")
            if boots:
                self.unlock("WIN_GIANTBOOTS_NBP")
            if ring:
                self.unlock("WIN_ICERING_NPB")
        elif ending == 3:
            if minutes < 60:
                self.unlock("WIN_TIME_60_OWL")
            if minutes < 30:
                self.unlock("WIN_TIME_30_OWL")
            bird = self.run.ravens.get("tsuchinoko")
            if bird is not None and bird.is_done:
                self.unlock("GET_OWLCAPE")
            self.unlock("WIN_OWL")
            if stats["falls"] == 2:
                self.unlock("WIN_OWL_2_FALL")
            if boots:
                self.unlock("WIN_GIANTBOOTS_OWL")
            if ring:
                self.unlock("WIN_ICERING_OWL")
        if self.wearing("Crown"):
            self.unlock("WIN_WITH_CROWN")
        if self.wearing("Shoes"):
            self.unlock("WIN_WITH_SHOES")
        self._take_snapshot()
        # the cutscenes: story flag and the reward
        if ending == 1:
            p.flags.add(COMPLETED_NORMAL)
            self._give("Crown")
        elif ending == 2:
            p.flags.add(COMPLETED_NBP)
            self._give("CrownNBP")
        elif ending == 3:
            p.flags.add(COMPLETED_GHOST)
            bird = self.run.ravens.get("tsuchinoko")
            if bird is not None and bird.is_done:
                self._give("CapeOwl")
            self._give("CrownOwl")

    def _give(self, item):
        """GiveWearableItemNode."""
        self.profile.add_once(item)
        self.enable_skin(item)

    # ------------------------------------------------------------- tick
    @property
    def camera(self):
        return int(self.game.state[P.S_SCREEN])

    @property
    def hitbox(self):
        s = self.game.state
        return (int(s[P.S_X]), int(s[P.S_Y]), C.PLAYER_W, C.PLAYER_H)

    def step(self, direction=0, jump=False):
        """One frame. Returns the kernel's event flags; achievements unlocked
        during the frame are in ``new_achievements``."""
        if self.run is None:
            raise RuntimeError("call new_game() first")
        self.new_achievements = []
        if self.ended:
            return 0
        prof = self.profile
        # GameLoop: the achievement clock, then (inside the kernel) the win check
        prof.ticks += 1
        self.frames += 1
        self._check_skin_combo()
        # dynamic entities (created this run) update first: they were added last
        for e in reversed(self.dynamic):
            if e.alive:
                e.update()
        # FullRunManager
        if self.camera == 0:
            self.run.wear_snake_ring = self.wearing("SnakeRing")
            self.run.wear_giant_boots = self.wearing("GiantBoots")
        else:
            if not self.wearing("SnakeRing"):
                self.run.wear_snake_ring = False
            if not self.wearing("GiantBoots"):
                self.run.wear_giant_boots = False
        # the player
        ev = self.game.step(direction, jump)
        if ev & P.EV_GOAL:
            self.ended = True
            self.ending = int(self.game.state[P.S_ENDING])
            self._on_victory(self.ending)
            return ev
        if ev & P.EV_JUMP:
            self._on_jump()
        if ev & P.EV_SPLAT:
            self._on_fall()
        # props, items, merchants, NPCs and ravens (reverse creation order)
        for e in reversed(self.entities):
            if e.alive:
                e.update()
        if self.frames % 600 == 0:
            self.entities = [e for e in self.entities if e.alive]
            self.dynamic = [e for e in self.dynamic if e.alive]
        return ev

    def run_inputs(self, inputs):
        ev = 0
        for d, j in inputs:
            ev |= self.step(d, j)
            if self.ended:
                break
        return ev

    # ---------------------------------------------------------- queries
    def entity(self, name):
        """The NPC, merchant or raven called ``name``."""
        for e in self.entities:
            s = getattr(e, "s", None)
            if isinstance(e, Merchant) and e.m.settings.name == name:
                return e
            if s is not None and getattr(s, "name", None) == name:
                return e
        raise KeyError(name)

    _SHARED = ("game", "content", "level")

    def _memo(self):
        """deepcopy memo that keeps the campaign, the physics and the
        (read-only) game content shared instead of copied."""
        memo = {id(self): self}
        for name in self._SHARED:
            obj = getattr(self, name)
            memo[id(obj)] = obj
        c = self.content
        for group in (c.old_men, c.ravens, c.merchants):
            for obj in group:
                memo[id(obj)] = obj
                inner = getattr(obj, "settings", None)
                if inner is not None:
                    memo[id(inner)] = inner
        return memo

    def snapshot(self):
        """A copy of the whole campaign (profile, run, entities, physics)."""
        data = {k: v for k, v in self.__dict__.items() if k not in self._SHARED}
        snap = copy.deepcopy(data, self._memo())
        snap["_state"] = self.game.snapshot()
        return snap

    def restore(self, snap):
        snap = dict(snap)
        state = snap.pop("_state")
        self.__dict__.update(copy.deepcopy(snap, self._memo()))
        self.game.restore(state)
        return self
