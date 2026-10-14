"""Every one of the 48 achievements, through the campaign's mechanics.

The king is placed next to the thing that matters (an NPC, a bird's perch,
an item, a babe) and the game is then played frame by frame, so each test
exercises the real trigger: dialogue pools and timing, merchant prices, raven
paths, pickups, skins, run statistics and endings. Skipped unless Jump King is
installed (the content is read from the install)."""
import pytest

from jumpking import physics as P
from jumpking.achievements import CODES
from jumpking.xnb import find_level_xnb

pytestmark = pytest.mark.skipif(find_level_xnb() is None, reason="Jump King is not installed")


@pytest.fixture(scope="module")
def shared():
    from jumpking.content import GameContent
    from jumpking.level import Level

    return Level.original(), GameContent.load()


@pytest.fixture
def camp(shared):
    from jumpking.campaign import Campaign

    level, content = shared
    c = Campaign(level=level, content=content)
    c.new_game("normal")
    return c


def rect_overlap(a, b):
    return b[0] < a[0] + a[2] and a[0] < b[0] + b[2] and b[1] < a[1] + a[3] and a[1] < b[1] + b[3]


def stand_in(c, screen0, rect):
    """Stand still on a floor of ``screen0`` with the hitbox touching ``rect``."""
    for x, y in c.game.spawn_points(screen0, stride=2):
        if rect_overlap((int(x), int(y), 18, 26), rect):
            c.game.reset(x, y, screen=screen0)
            c._sync_items()
            return x, y
    raise AssertionError("no standing spot touching %r on screen %d" % (rect, screen0 + 1))


def stand_on(c, screen0, k=0):
    x, y = c.game.spawn_points(screen0, stride=8)[k]
    c.game.reset(x, y, screen=screen0)
    c._sync_items()
    return x, y


def wait(c, frames):
    for _ in range(frames):
        c.step(0, False)


def visit(c, screens):
    """Stand on each screen for a few frames (the NPC quote pools grow when
    the camera reaches a new screen)."""
    for s in screens:
        stand_on(c, s)
        wait(c, 3)


def talk_until(c, name, times, limit_s=3600):
    npc = c.entity(name)
    s = npc.s
    stand_in(c, s.home_screen - 1, npc.rect)
    for _ in range(int(limit_s * 60)):
        c.step(0, False)
        if npc.times_talked >= times and not npc.busy:
            break
    return npc


def win(c, ending):
    """Stand where babe ``ending`` (1-3) is reached and play one frame."""
    e = c.level.endings[ending - 1]
    spots = [p for p in c.game.spawn_points(e.screen, stride=1)
             if (int(p[0]) + 9 >= e.x if e.side > 0 else int(p[0]) + 9 <= e.x)
             and int(p[1]) + 13 <= e.max_y]
    x, y = spots[0]
    c.game.reset(x, y, screen=e.screen)
    c._sync_items()
    c.step(0, False)
    assert c.ended and c.ending == ending
    return c


def test_there_are_48(camp):
    assert len(CODES) == 48
    assert not camp.unlocked


# ---------------------------------------------------------------- statistics
def test_jump_counts(camp):
    camp.profile.jumps = 998
    stand_on(camp, 0)
    for _ in range(2):
        camp.step(0, True)
        wait(camp, 30)
    assert "JUMP_1000" in camp.unlocked
    camp.profile.jumps = 9999
    camp.step(0, True)
    wait(camp, 30)
    assert "JUMP_10000" in camp.unlocked
    camp.profile.jumps = 19999
    camp.step(0, True)
    wait(camp, 30)
    assert "JUMP_100000" in camp.unlocked


def _splat(c):
    c.game.reset(231, 88, screen=0)      # 214 px above the floor of screen 1
    for _ in range(200):
        if c.step(0, False) & P.EV_SPLAT:
            return
    raise AssertionError("no splat")


def test_fall_counts(camp):
    camp.profile.falls = 98
    _splat(camp)
    _splat(camp)
    assert "FALL_100" in camp.unlocked
    camp.profile.falls = 999
    _splat(camp)
    assert "FALL_1000" in camp.unlocked


# --------------------------------------------------------------- the NPCs
def test_old_man_10_and_40_conversations(camp):
    visit(camp, range(1, 24))
    npc = talk_until(camp, "old_man", 40)
    assert npc.times_talked == 40
    assert {"TALK_OLDMAN_10", "TALK_OLDMAN_ALL"} <= camp.unlocked


def test_old_man_needs_new_screens_to_talk(camp):
    npc = talk_until(camp, "old_man", 2, limit_s=200)
    assert npc.times_talked == 1           # only the intro without new screens
    assert "TALK_OLDMAN_10" not in camp.unlocked


def test_hermit(camp):
    visit(camp, range(24, 33))
    npc = talk_until(camp, "hermit", 10)
    assert npc.times_talked >= 10
    assert "TALK_HERMIT_10" in camp.unlocked


def test_merchant_conversations(camp):
    visit(camp, [1])
    talk_until(camp, "merchant", 4)
    assert "TALK_MERCHANT_10" in camp.unlocked


def test_ninth_cat_and_the_gargoyles(camp):
    talk_until(camp, "owl_chars/cat_outsider", 1)
    assert "FIND_9TH_CAT" in camp.unlocked
    talk_until(camp, "owl_chars/owl_boss_gargoyle", 1)
    assert "TIMESTOP_GARGOYLE_TALK" in camp.unlocked


# ------------------------------------------------------------ the merchants
def test_shoes_for_ten_gold_rings(shared):
    from jumpking.campaign import Campaign, Profile

    level, content = shared
    prof = Profile()
    prof.add("GoldRing", 9)                 # nine rings from earlier runs
    c = Campaign(level=level, content=content, profile=prof)
    c.new_game("normal")
    shop = c.entity("merchant")
    assert shop.price == 10                 # rings at the start of the run + 1
    prof.add("GoldRing", 1)                 # the tenth ring of this run
    talk_until(c, "merchant", 1)
    wait(c, 60 * 90)
    assert shop.state.sold and prof.has("Shoes") and prof.count("GoldRing") == 0
    assert {"PURCHASE_SHOES", "PURCHASE_FOR_10"} <= c.unlocked
    assert c.wearing("Shoes")


def test_the_dlc_merchants(shared):
    from jumpking.campaign import Campaign, Profile

    level, content = shared
    prof = Profile()
    c = Campaign(level=level, content=content, profile=prof)
    c.new_game("new_babe_plus")
    for item, n in (("Silver", 10), ("Ruby", 1), ("GhostFragment", 4), ("Shroom", 9)):
        prof.add(item, n)
    for name in ("merchant_nbp", "snake", "owl_chars/owl_mold_man", "owl_chars/owl_shroom_guy"):
        talk_until(c, name, 1)
        wait(c, 60 * 120)
        assert c.entity(name).state.sold, name
    for item in ("GiantBoots", "SnakeRing", "YellowShoes", "Tunic"):
        assert prof.has(item)
    assert {"BUY_GIANTBOOTS", "BUY_ICERING", "PURCHASE_YELLOW_SHOES",
            "PURCHASE_SCHOLAR_TUNIC"} <= c.unlocked


# ---------------------------------------------------------------- the birds
def chase(c, name, upto=None):
    """Scare bird ``name`` off its perches until it sits on perch ``upto``
    (0-based), or until it has left its last perch when ``upto`` is None."""
    bird = c.entity(name)
    while not bird.state.is_done and (upto is None or bird.state.home < upto):
        p = bird.perch
        x, y = p["position"]
        c.game.reset(x - 9, y - (p["screen"] - 1) * 360 - 13, screen=p["screen"] - 1)
        before = (bird.state.home, bird.state.is_done)
        for _ in range(600):
            c.step(0, False)
            if (bird.state.home, bird.state.is_done) != before:
                break
        else:
            raise AssertionError("bird %s did not flee from perch %d" % (name, bird.state.home))
    return bird


def test_raven_to_the_snowstorm_drops_a_gold_ring(camp):
    bird = chase(camp, "raven", upto=13)
    assert bird.state.home == 13
    assert "CHASE_BIRD_PAST_SNOWSTORM" in camp.unlocked
    ring = [w for w in camp.run.world_items if w[0] == "GoldRing"]
    assert ring
    item, _, x, y = ring[0]
    camp.game.reset(x - 9, y - 26, screen=31)
    wait(camp, 5)
    assert camp.profile.count("GoldRing") == 1


def test_the_fly_has_39_perches(camp):
    bird = chase(camp, "fly")
    assert bird.state.is_done
    assert "CHASE_AWAY_BUG" in camp.unlocked


def test_white_raven_drops_the_ruby(shared):
    from jumpking.campaign import Campaign

    level, content = shared
    c = Campaign(level=level, content=content)
    c.new_game("normal")                    # the white raven moves to New Babe+
    bird = chase(c, "white_raven")
    assert any(w[0] == "Ruby" for w in c.run.world_items) or c.profile.has("Ruby")


# ------------------------------------------------------- places and items
@pytest.mark.parametrize("code", ["REACH_CATHEDRAL", "FIND_IMP_ROOM", "FIND_NOSE", "REACH_SEA",
                                  "GO_THOUGH_ILLUSORY_WALL"])
def test_places(camp, code):
    rects = [r for k, r in camp.content.achievement_hitboxes + camp.content.illusory_walls
             if k == code][0]
    x, y, w, h = rects[0]
    screen = -((y + h // 2) // 360)
    camp.game.reset(x + w // 2 - 9, y + h // 2 - 13, screen=screen)
    camp.step(0, False)
    assert code in camp.unlocked


def test_gnome_hat_pickup_and_outfits(camp):
    hat = [w for w in camp.run.world_items if w[0] == "GnomeHat"][0]
    item, screen, x, y = hat
    camp.game.reset(x - 9, y - (screen - 1) * 360 - 26, screen=screen - 1)
    wait(camp, 3)
    assert camp.wearing("GnomeHat")
    wait(camp, 1)
    assert "GET_GNOME_HAT" in camp.unlocked
    for item in ("Tunic", "YellowShoes", "Shoes", "SnakeRing", "CapeOwl", "GiantBoots"):
        camp.profile.add(item)
    camp.equip("Tunic")
    camp.equip("YellowShoes")
    wait(camp, 1)
    assert "WEAR_TUNIC_YELLOWSHOES_AND_GNOMEHAT" in camp.unlocked
    camp.equip("Shoes")                      # replaces the yellow shoes (same layer)
    assert not camp.wearing("YellowShoes")
    camp.equip("SnakeRing")
    camp.equip("CapeOwl")
    wait(camp, 1)
    assert "WEAR_SHOES_RING_AND_OWLCAPE" in camp.unlocked
    camp.equip("GiantBoots")
    wait(camp, 1)
    assert "WEAR_GIANTBOOTS_AND_ICERING" in camp.unlocked
    assert camp.game.state[P.S_BOOTS] == 1 and camp.game.state[P.S_RING] == 1


# ---------------------------------------------------------------- endings
def test_normal_ending_rewards(camp):
    win(camp, 1)
    assert {"WIN", "WIN_TIME_60", "WIN_TIME_15", "WIN_NO_FALLS"} <= camp.unlocked
    assert camp.profile.has("Crown") and camp.wearing("Crown")
    # a second win wearing the crown (and the shoes)
    camp.profile.add("Shoes")
    camp.new_game("normal")
    camp.equip("Shoes")
    win(camp, 1)
    assert {"WIN_WITH_CROWN", "WIN_WITH_SHOES"} <= camp.unlocked


def test_normal_ending_after_two_raven_flights(camp):
    chase(camp, "raven", upto=2)
    assert camp.entity("raven").state.home == 2
    win(camp, 1)
    assert "WIN_TOUCH_CROW_2" in camp.unlocked


def test_new_babe_plus_from_the_imp_room(camp):
    win(camp, 1)                              # the crown
    camp.new_game("new_babe_plus")            # sacrificed: start in the Imp Room
    assert not camp.profile.has("Crown")
    assert camp.camera == 44
    wait(camp, 2)
    assert "FIND_IMP_ROOM" in camp.unlocked
    win(camp, 2)
    assert {"WIN_NBP", "WIN_NBP_NO_FALLS"} <= camp.unlocked
    assert camp.profile.has("CrownNBP")


def test_ghost_ending_with_the_red_bird(camp):
    camp.new_game("ghost_of_the_babe")
    chase(camp, "tsuchinoko")
    win(camp, 3)
    assert {"WIN_OWL", "WIN_TIME_60_OWL", "WIN_TIME_30_OWL", "GET_OWLCAPE"} <= camp.unlocked
    assert camp.profile.has("CapeOwl") and camp.profile.has("CrownOwl")


def test_ghost_ending_with_two_falls(camp):
    camp.new_game("ghost_of_the_babe")
    _splat(camp)
    wait(camp, 60)
    _splat(camp)
    win(camp, 3)
    assert "WIN_OWL_2_FALL" in camp.unlocked


@pytest.mark.parametrize("mode,ending,codes", [
    ("normal", 1, {"WIN_GIANTBOOTS_REGULAR", "WIN_ICERING_REGULAR"}),
    ("new_babe_plus", 2, {"WIN_GIANTBOOTS_NBP", "WIN_ICERING_NPB"}),
    ("ghost_of_the_babe", 3, {"WIN_GIANTBOOTS_OWL", "WIN_ICERING_OWL"}),
])
def test_item_runs_must_wear_the_items_from_screen_one(camp, mode, ending, codes):
    for item in ("GiantBoots", "SnakeRing"):
        camp.profile.add(item)
        camp.equip(item)
    camp.new_game(mode, sacrifice_crown=False)
    wait(camp, 2)                             # on screen 1 with both items
    win(camp, ending)
    assert codes <= camp.unlocked
    # taking an item off mid-run voids its achievement
    camp.new_game(mode, sacrifice_crown=False)
    wait(camp, 2)
    stand_on(camp, 5)
    camp.equip("SnakeRing", False)
    wait(camp, 2)
    camp.equip("SnakeRing", True)
    win(camp, ending)


def test_item_run_without_screen_one_does_not_count(shared):
    from jumpking.campaign import Campaign, Profile

    level, content = shared
    prof = Profile()
    prof.add("Crown")
    prof.add("GiantBoots")
    c = Campaign(level=level, content=content, profile=prof)
    c.equip("GiantBoots")
    c.new_game("new_babe_plus")               # crown sacrifice: starts in the Imp Room
    win(c, 2)
    assert "WIN_GIANTBOOTS_NBP" not in c.unlocked


def test_snapshot_restore_roundtrip(camp):
    visit(camp, [1, 2])
    stand_in(camp, 0, camp.entity("old_man").rect)
    wait(camp, 100)
    snap = camp.snapshot()
    wait(camp, 400)
    a = (camp.entity("old_man").talk.bb["line"], camp.frames, camp.game.x)
    camp.restore(snap)
    wait(camp, 400)
    assert (camp.entity("old_man").talk.bb["line"], camp.frames, camp.game.x) == a


def test_all_48_on_one_profile(shared):
    """One profile, several runs, the way a player collects them all."""
    from jumpking.campaign import Campaign, Profile

    level, content = shared
    prof = Profile()
    c = Campaign(level=level, content=content, profile=prof)

    # run 1: the old man, the merchant, the hermit, the places, the raven
    # (twice only), the normal babe within 15 minutes without a fall
    c.new_game("normal")
    visit(c, range(1, 33))
    talk_until(c, "old_man", 40)
    talk_until(c, "merchant", 4)
    talk_until(c, "hermit", 10)
    for code in ("REACH_CATHEDRAL", "GO_THOUGH_ILLUSORY_WALL"):
        r = [r for k, r in content.achievement_hitboxes + content.illusory_walls if k == code][0][0]
        c.game.reset(r[0] + r[2] // 2 - 9, r[1] + r[3] // 2 - 13, screen=-((r[1] + r[3] // 2) // 360))
        c.step(0, False)
    c.new_game("normal")                      # (talking took too long for 15 minutes)
    chase(c, "raven", upto=2)
    win(c, 1)                                 # crown, Jump King, Fool me Twice...
    # runs 2-10: chase the raven to the snowstorm for its gold ring
    for _ in range(9):
        c.new_game("normal")
        chase(c, "raven")
    assert prof.count("GoldRing") == 9
    # run 11: the tenth ring, then the shoes for ten rings; win with crown + shoes
    c.new_game("normal")
    assert c.entity("merchant").price == 10
    chase(c, "raven")
    talk_until(c, "merchant", 1)
    wait(c, 60 * 90)
    assert c.wearing("Shoes") and c.wearing("Crown")
    for _ in range(100 - prof.falls):
        _splat(c)
    prof.jumps = max(prof.jumps, 19990)
    stand_on(c, 0)
    for _ in range(20):
        c.step(0, True)
        wait(c, 30)
    win(c, 1)
    # run 12: New Babe+ from the Imp Room (crown sacrificed): ten silver coins
    # for the Giant Boots, the white raven's ruby for the Snake Ring, the
    # nose, the sea and the babe on screen 100 without a fall
    c.new_game("new_babe_plus")
    wait(c, 2)                                # in the Imp Room
    for item in [w for w in c.run.world_items if w[0] == "Silver"]:
        _, screen, x, y = item
        c.game.reset(x - 9, y - (screen - 1) * 360 - 26, screen=screen - 1)
        wait(c, 2)
    talk_until(c, "merchant_nbp", 1)
    wait(c, 60 * 120)
    chase(c, "white_raven")
    talk_until(c, "snake", 1)
    wait(c, 60 * 120)
    for code in ("FIND_NOSE", "REACH_SEA"):
        r = [r for k, r in content.achievement_hitboxes if k == code][0][0]
        c.game.reset(r[0] + r[2] // 2 - 9, r[1] + r[3] // 2 - 13, screen=-((r[1] + r[3] // 2) // 360))
        c.step(0, False)
    win(c, 2)
    # run 13: Ghost of the Babe: shrooms, fragments, the gnome hat, the cats,
    # the gargoyles, the fly, the red bird; two falls; the babe on screen 154
    c.new_game("ghost_of_the_babe")
    for kind in ("Shroom", "GhostFragment", "GnomeHat"):
        for item in [w for w in c.run.world_items if w[0] == kind]:
            _, screen, x, y = item
            c.game.reset(x - 9, y - (screen - 1) * 360 - 26, screen=screen - 1)
            wait(c, 2)
    talk_until(c, "owl_chars/owl_shroom_guy", 1)
    wait(c, 60 * 120)
    talk_until(c, "owl_chars/owl_mold_man", 1)
    wait(c, 60 * 120)
    talk_until(c, "owl_chars/cat_outsider", 1)
    talk_until(c, "owl_chars/owl_boss_gargoyle", 1)
    chase(c, "fly")
    chase(c, "tsuchinoko")
    _splat(c)
    wait(c, 60)
    _splat(c)
    win(c, 3)
    # dressing up: gnome hat + tunic + yellow shoes; shoes + ring + owl cape
    c.new_game("normal")
    c.equip("GnomeHat")
    c.equip("Tunic")
    c.equip("YellowShoes")
    c.step(0, False)
    c.equip("Shoes")
    c.equip("SnakeRing")
    c.equip("CapeOwl")
    c.step(0, False)
    # runs 14-16: each babe wearing the Giant Boots and the Snake Ring from screen 1
    c.equip("GiantBoots")
    for mode, ending in (("normal", 1), ("new_babe_plus", 2), ("ghost_of_the_babe", 3)):
        c.new_game(mode, sacrifice_crown=False)
        wait(c, 2)
        win(c, ending)
    # the last falls
    c.new_game("normal")
    while prof.falls < 1000:
        _splat(c)
        wait(c, 31)
    assert set(CODES) - c.unlocked == set()


def test_campaign_env(shared):
    import gymnasium as gym

    import jumpking.envs  # noqa: F401
    from jumpking.envs.campaign_env import CAMPAIGN_FEATURES

    level, content = shared
    env = gym.make("JumpKingCampaign-v0", level=level, content=content)
    obs, info = env.reset(seed=0)
    assert obs.shape == (len(CAMPAIGN_FEATURES),)
    total = 0.0
    for _ in range(60 * 30 // 4):           # stand by the old man for 30 s
        obs, r, term, trunc, info = env.step(0)
        total += r
    assert info["achievements"] == 0 and not term
    # the macro variant: unlocking a place pays the achievement bonus
    menv = gym.make("JumpKingCampaignMacro-v0", level=level, content=content)
    menv.reset(seed=0, options={"profile": "new"})
    camp = menv.unwrapped.camp
    x, y, w, h = [r for k, r in content.achievement_hitboxes if k == "REACH_CATHEDRAL"][0][0]
    camp.game.reset(200, -11520 + 302 - 26, screen=32)
    obs, r, term, trunc, info = menv.step(menv.unwrapped._macro.index(("walk", 1, 4)))
    assert "REACH_CATHEDRAL" in info["new_achievements"] and r >= 1.0


@pytest.mark.parametrize("key,mode,expected", [
    ("base", "normal", {"GO_THOUGH_ILLUSORY_WALL", "REACH_CATHEDRAL", "WIN", "WIN_TIME_60",
                        "WIN_TIME_15", "WIN_NO_FALLS", "WIN_TOUCH_CROW_2"}),
    ("nbp", "new_babe_plus", {"FIND_IMP_ROOM", "REACH_SEA", "WIN_NBP", "WIN_NBP_NO_FALLS"}),
    ("ghost", "ghost_of_the_babe", {"TIMESTOP_GARGOYLE_TALK", "WIN_OWL", "WIN_TIME_60_OWL",
                                    "WIN_TIME_30_OWL", "WIN_OWL_2_FALL"}),
])
def test_speedruns_played_in_the_campaign(shared, key, mode, expected):
    """The real speed-runs, played frame by frame in the full campaign (NPCs,
    birds, triggers all running), unlock what a real player would get."""
    from jumpking import tas, verify
    from jumpking.campaign import Campaign, Profile

    from test_original import _route_text

    level, content = shared
    route = verify.ROUTES[key]
    start, inputs = tas.to_inputs(tas.parse(_route_text(route)))
    prof = Profile()
    prof.add("Crown")                          # New Babe+: the crown is sacrificed
    c = Campaign(level=level, content=content, profile=prof,
                 legacy_teleport=route.legacy_teleport)
    c.new_game(mode)
    c.game.reset(start[0], start[1])
    c.game.state[P.S_FACING] = start[2]
    c.game.state[P.S_TICK] = route.start_tick
    c.step(0, False)
    for d, j in inputs:
        c.step(d, j)
        if c.ended:
            break
    assert c.ended and c.ending == route.ending
    assert expected <= c.unlocked
