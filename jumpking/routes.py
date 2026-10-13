"""The three ways to finish Jump King in the shared 169-screen world.

* ``normal``            - base game: screens 1-43, the babe on screen 43.
* ``new_babe_plus``     - New Babe+: Bargainburg's side passage (screen 18) to
  the Imp Room (45), up through 46-100 to the babe on screen 100. In the game
  the New Babe+ mode starts on screen 1; owning the crown lets you sacrifice it
  and start in the Imp Room instead (``NBPDecisionState.TeleportToImpRoom``).
* ``ghost_of_the_babe`` - Ghost of the Babe: Redcrown Woods' side passage
  (screen 3) into Philosopher's Forest (156-160), its side exit to the top of
  the Hole (164), down the Hole to 161, the side passage to the Bog (108),
  down to the bottom of the Bog (102) and up 102-154 to the babe.

``ROUTES`` lists, per ending, the screens (0-based) in the order the community
TAS routes visit them; ``RouteProgress`` turns that into a progress counter
for rewards and curricula.
"""
import numpy as np

ENDING_NAMES = ("normal", "new_babe_plus", "ghost_of_the_babe")
ENDING_IDS = {name: i + 1 for i, name in enumerate(ENDING_NAMES)}

# player start positions (hitbox top-left, world coordinates)
START_POSITIONS = {
    "start": (231.0, 302.0),            # SaveState.GetDefault (screen 1)
    "imp_room": (171.0, -15570.0),      # NBPDecisionState.TeleportToImpRoom (screen 45)
}

_NORMAL = list(range(1, 44))
_NBP = list(range(1, 19)) + [45] + list(range(46, 101))
_GHOST = ([1, 2, 3] + list(range(156, 161)) + [164, 163, 162, 161]
          + list(range(108, 101, -1)) + list(range(103, 155)))

ROUTES = {
    "normal": [s - 1 for s in _NORMAL],
    "new_babe_plus": [s - 1 for s in _NBP],
    "ghost_of_the_babe": [s - 1 for s in _GHOST],
}


def ending_id(name):
    """1-based ending id for ``name`` (0 for ``"any"``)."""
    if name == "any":
        return 0
    try:
        return ENDING_IDS[name]
    except KeyError:
        raise ValueError("ending must be one of %s or 'any'" % (ENDING_NAMES,)) from None


def route_for(name):
    return ROUTES["normal" if name == "any" else name]


class RouteProgress:
    """Index of the furthest route screen reached in order (batched).

    The index only advances when the king enters the next screen of the
    route, so detours and falls never count as progress."""

    def __init__(self, route, n=1):
        self.route = np.asarray(route, np.int64)
        self.index = np.zeros(n, np.int64)

    def reset(self, screens, idx=None):
        screens = np.atleast_1d(np.asarray(screens, np.int64))
        idx = np.arange(len(self.index)) if idx is None else np.atleast_1d(idx)
        for j, i in enumerate(idx):
            # start at the first occurrence of the current screen
            hits = np.nonzero(self.route == screens[j])[0]
            self.index[i] = hits[0] if len(hits) else 0

    def update(self, screens):
        """Advance with the current screens; returns the number of new steps."""
        screens = np.atleast_1d(np.asarray(screens, np.int64))
        before = self.index.copy()
        n = len(self.route)
        # a macro action can cross several screens: look a few steps ahead
        for k in range(1, 6):
            j = self.index + k
            ok = (j < n) & (self.index == before)
            ok[ok] = self.route[j[ok]] == screens[ok]
            self.index[ok] = j[ok]
        return self.index - before

    @property
    def fraction(self):
        return self.index / max(1, len(self.route) - 1)
