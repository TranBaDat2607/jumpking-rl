"""The saved all-48 playthrough: new games played with button inputs only.

``jumpking/data/playthrough_all48.json`` was produced by
``scripts/achievement_run.py``: the community speed-runs (referenced, not
copied, see ``planner.tas_frames``) and routes found by the planner. Replaying
it on a fresh profile must unlock every one of the 48 achievements, with the
king never placed or teleported. Skipped unless Jump King is installed (the
maps and NPC scripts are read from the user's copy).
"""
import json
import os

import pytest

from jumpking.xnb import find_level_xnb

pytestmark = pytest.mark.skipif(find_level_xnb() is None, reason="Jump King is not installed")

PATH = os.path.join(os.path.dirname(__file__), "..", "jumpking", "data", "playthrough_all48.json")


def test_playthrough_unlocks_all_48():
    from jumpking import verify
    from jumpking.achievements import CODES
    from jumpking.campaign import Campaign
    from jumpking.planner import replay, tas_frames

    with open(PATH, "r", encoding="utf-8") as fh:
        runs = json.load(fh)["runs"]
    try:
        for key in verify.ROUTES:
            tas_frames(key)
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip("cannot load the speed-runs: %s" % exc)
    c = Campaign()
    replay(c, runs)
    assert set(CODES) - c.unlocked == set()
    assert c.profile.jumps >= 20000 and c.profile.falls >= 1000
