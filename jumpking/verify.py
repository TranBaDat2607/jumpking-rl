"""Tick-level parity checks against the real game's community TAS routes.

ShootMe/JumpKingTAS (``LevelFiles/*.tas``) holds frame-perfect speed-runs
recorded in the real game for each of the three endings:

* ``NewGame.tas``  - the base game, Redcrown Woods to the babe on screen 43;
* ``NewBabe.tas``  - New Babe+, from the Imp Room (the crown sacrifice) to the
  babe on screen 100;
* ``NewGhost.tas`` - Ghost of the Babe, through Philosopher's Forest, the Hole
  and the Bog to the babe on screen 154.

Each run ends with the in-game timer written as a comment (``#3:44.859``).
Replaying a route in the clone must visit every section's screen and win on the
very tick the timer shows, which only happens if every jump, bump, slope
slide, teleport, wind gust, water and sand step matches the game exactly.

The routes were recorded in 2019-2020. Two details of that era are
reproduced by the replay settings below: the TAS tool's ``X`` (statistics
reset) leaves the run clock at -2 ticks for ``NewGame.tas`` and at 0 for the
two DLC routes, and side teleports of that game build kept the fractional
y coordinate (``legacy_teleport``; today's build truncates it, see
``physics.P_LEGACY_TELEPORT``). No route crosses a teleport at a fractional y
except ``NewGhost.tas`` (screen 3 -> 156).
"""
import re
import urllib.request

from . import physics as P
from . import tas

TAS_BASE_URL = "https://raw.githubusercontent.com/ShootMe/JumpKingTAS/master/LevelFiles/"

# section name prefix -> 1-based screens climbed in order
_BASE = (("Redcrown Woods", range(1, 6)), ("Colossal Drain", range(6, 11)),
         ("False Kings Keep", range(11, 15)), ("Bargainburg", range(15, 20)),
         ("Great Frontier", range(20, 26)), ("Windswept Bluff", (26,)),
         ("Stormwall Pass", range(27, 33)), ("Chapel Perilous", range(33, 37)),
         ("Blue Ruin", range(37, 40)), ("The Tower", range(40, 44)))
# the route names the Imp Room and the passage above it "Bargainburg 6/7"
_NBP = (("Bargainburg", range(40, 47)), ("Brightcrown Woods", range(47, 53)),
        ("Colossal Dungeon", range(53, 60)), ("False Kings Castle", range(60, 64)),
        ("Underburg", range(64, 71)), ("Lost Frontier", range(71, 78)),
        ("Hidden Kingdom", range(78, 84)), ("Black Sanctum", range(84, 90)),
        ("Deep Ruin", range(90, 95)), ("The Dark Tower", range(95, 101)))
_GHOST = (("Redcrown Woods", range(1, 4)), ("Philosophers Forest", range(156, 161)),
          ("Hole", (164, 163, 162, 161)), ("Bog", range(102, 109)),
          ("Moulding Manor", range(109, 117)), ("Bugstalk", range(117, 124)),
          ("House Of Nine Lives", range(124, 131)), ("The Phantom Tower", range(131, 140)),
          ("Halted Ruin", range(140, 148)), ("Tower Of Antumbra", range(148, 155)))


class Route:
    def __init__(self, name, filename, sections, ending, start_tick, legacy_teleport):
        self.name = name
        self.filename = filename
        self.sections = sections
        self.ending = ending              # 1 normal, 2 New Babe+, 3 Ghost of the Babe
        self.start_tick = start_tick
        self.legacy_teleport = legacy_teleport

    @property
    def url(self):
        return TAS_BASE_URL + self.filename

    def expected_screen(self, section):
        name = re.sub(r"\(.*?\)", "", section).strip()
        m = re.match(r"(.+?)\s+(\d+)$", name)
        if not m:
            return None
        area = m.group(1).replace("King's", "Kings").replace("Philosopher's", "Philosophers")
        for a_name, screens in self.sections:
            if a_name.lower() == area.lower():
                screens = list(screens)
                k = int(m.group(2)) - 1
                return screens[k] if 0 <= k < len(screens) else None
        return None


ROUTES = {
    "base": Route("base game", "NewGame.tas", _BASE, 1, -2, False),
    "nbp": Route("New Babe+", "NewBabe.tas", _NBP, 2, 0, True),
    "ghost": Route("Ghost of the Babe", "NewGhost.tas", _GHOST, 3, 0, True),
}


def download(route, timeout=30):
    return urllib.request.urlopen(route.url, timeout=timeout).read().decode("utf-8", "replace")


def recorded_time(text):
    """The final in-game time written at the end of a route, in seconds."""
    found = re.findall(r"^#\s*(?:(\d+):)?(\d+\.\d+)\s*$", text, re.M)
    if not found:
        return None
    mm, ss = found[-1]
    return (int(mm) * 60 if mm else 0) + float(ss)


class Report:
    def __init__(self):
        self.sections = []        # (name, expected screen, screens seen, ok)
        self.win_tick = None      # run clock (ticks) on the frame the babe is reached
        self.ending = 0
        self.recorded = None

    @property
    def ok(self):
        return sum(1 for s in self.sections if s[3])

    @property
    def checked(self):
        return len(self.sections)

    @property
    def win_time(self):
        from . import constants as C

        return None if self.win_tick is None else self.win_tick * C.TARGET_ELAPSED_SECONDS

    @property
    def exact(self):
        """All sections visited and the win lands on the recorded timer."""
        return (self.ok == self.checked and self.win_time is not None
                and self.recorded is not None and abs(self.win_time - self.recorded) < 5e-4)


def replay(level, route, text):
    """Replay a TAS route on ``level`` (normally ``Level.original()``)."""
    from .game import Game

    sections = []
    current, buf = None, []
    for line in text.splitlines():
        if line.startswith("#") and not line.startswith("#!"):
            if current is not None:
                sections.append((current, "\n".join(buf)))
            current, buf = line[1:].strip(), []
        else:
            buf.append(line)
    if current is not None:
        sections.append((current, "\n".join(buf)))

    rep = Report()
    rep.recorded = recorded_time(text)
    g = Game(level, legacy_teleport=route.legacy_teleport)
    started = False
    for name, body in sections:
        exp = route.expected_screen(name)
        seen = {g.screen + 1}
        for seg in tas.parse(body):
            if seg.state is not None and not started:
                x, y, facing = seg.state
                g.reset(x, y)
                g.state[P.S_FACING] = facing
                g.state[P.S_TICK] = route.start_tick
                started = True
                # the "@" line itself lasts one frame
                g.step(0, False)
                continue
            if not started:
                continue
            for _ in range(seg.frames):
                if g.step(seg.direction, seg.jump) & P.EV_GOAL and rep.win_tick is None:
                    # the timer shows the clock of the frame whose GameLoop
                    # detected the win; the kernel advances it past that frame
                    rep.win_tick = int(g.state[P.S_TICK]) - 1
                    rep.ending = int(g.state[P.S_ENDING])
                seen.add(g.screen + 1)
        if exp is not None and started:
            rep.sections.append((name, exp, sorted(seen), exp in seen))
    return rep
