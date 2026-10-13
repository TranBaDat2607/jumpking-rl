"""The 48 Steam achievements of Jump King (app 1061090).

Codes and their order are the game's ``AchievementCode`` enum (the Steam API
names); display names and descriptions are Steam's achievement schema for the
app (what SteamDB's stats page shows; ``steam_schema()`` reads it from a local
Steam client's cache, spelling included). ``how`` summarises the
exact unlock rule from the game's code (``AchievementManager`` and the
entities that call ``AchievementRegister.RegisterAchievement``).
"""
from collections import namedtuple

Achievement = namedtuple("Achievement", "code name description how")

ACHIEVEMENTS = (
    Achievement("FALL_100", "Just the Beginning", "If only you knew... (100 Falls)",
                "the 100th splat over all runs"),
    Achievement("FALL_1000", "Thick Skull", "Be happy you have a helmet. (1000 Falls)",
                "the 1000th splat over all runs"),
    Achievement("JUMP_1000", "Jump Squire", "You have much to learn... (1000 Jumps)",
                "1000 jumps over all runs"),
    Achievement("JUMP_10000", "Jump Knight", "You are a seasoned jumper. (10 000 Jumps)",
                "10 000 jumps over all runs"),
    Achievement("JUMP_100000", "Jump Lord",
                "My lord is a true veteran in the art of jumping! (20 000 Jumps)",
                "20 000 jumps over all runs"),
    Achievement("TALK_OLDMAN_10", "Talkative",
                "You have heard a taste of the old man's wisdom. (Listen to Old Man 10 times)",
                "finish the 10th conversation with the old man (screen 1) in one run"),
    Achievement("TALK_OLDMAN_ALL", "Apprentice of the Pond-Sage",
                "You have heard all that the old man has to teach. (Listen to Old Man 40 times)",
                "finish the 40th conversation with the old man in one run"),
    Achievement("TALK_HERMIT_10", "Meditations",
                "Hear the advice of the hermit on the mountain. (Listen to Hermit 10 times)",
                "finish the 10th conversation with the hermit (screen 25) in one run"),
    Achievement("TALK_MERCHANT_10", "Barter", "Each man has his price? (Talk to the merchant 4 times)",
                "finish the 4th conversation with the Bargainburg merchant (screen 15) in one run"),
    Achievement("WIN", "Jump King", "Reach the top.", "reach the babe on screen 43"),
    Achievement("WIN_TIME_60", "Swift Jumper", "Complete the game within 1 hour.",
                "normal ending in under 60 minutes of run time"),
    Achievement("WIN_TIME_15", "Lightning-Speed Jumper", "Complete the game within 15 minutes.",
                "normal ending in under 15 minutes of run time"),
    Achievement("WIN_WITH_CROWN", "Double King", "Finish the game wearing the crown.",
                "any ending while wearing the crown (from the normal ending)"),
    Achievement("WIN_WITH_SHOES", "Fashion King", "Finish game wearing the special boots.",
                "any ending while wearing the shoes bought from the merchant"),
    Achievement("WIN_TOUCH_CROW_2", "Fool me Twice...", "You seem to dislike birds?",
                "normal ending with the raven chased off exactly twice"),
    Achievement("WIN_NO_FALLS", "Godlike Jumper", "Complete the game without falling.",
                "normal ending without a splat"),
    Achievement("PURCHASE_SHOES", "Fashionable", "Own the special boots.",
                "buy the shoes from the Bargainburg merchant (gold rings)"),
    Achievement("PURCHASE_FOR_10", "Big Spender", "Only the finest for a King.",
                "buy the shoes when their price is 10 gold rings or more"),
    Achievement("CHASE_BIRD_PAST_SNOWSTORM", "Fowler", "That bird...",
                "chase the raven to its last perch (screen 32)"),
    Achievement("REACH_CATHEDRAL", "Fool's Paradise", "Reach Chapel Perilous.",
                "touch screen 33"),
    Achievement("GO_THOUGH_ILLUSORY_WALL", "Detective", "Enter through an illusory wall.",
                "touch an illusory wall on screens 6, 7, 8 or 22"),
    Achievement("FIND_IMP_ROOM", "Off the Beaten Path", "Find the room of The Imp.",
                "touch screen 45 (Bargainburg's side passage)"),
    Achievement("WIN_GIANTBOOTS_REGULAR", "Giant Leaps",
                "Complete game with giant boots.",
                "normal ending wearing the Giant Boots since screen 1"),
    Achievement("WIN_GIANTBOOTS_NBP", "Giant Leaps+",
                "Complete NB+ with giant boots.",
                "New Babe+ ending wearing the Giant Boots since screen 1"),
    Achievement("WIN_ICERING_REGULAR", "Absolute Will",
                "Complete the game with Snake Ring.",
                "normal ending wearing the Snake Ring since screen 1"),
    Achievement("WIN_ICERING_NPB", "Absolute Will+",
                "Complete NB+ with Snake Ring.",
                "New Babe+ ending wearing the Snake Ring since screen 1"),
    Achievement("WIN_NBP", "New King+", "Complete New Babe+.", "reach the babe on screen 100"),
    Achievement("WIN_NBP_NO_FALLS", "Mad King", "Complete New Babe+ without falling.",
                "New Babe+ ending without a splat"),
    Achievement("BUY_GIANTBOOTS", "Large Feet", "Purchase the Giant Boots.",
                "buy the Giant Boots from the merchant on screen 71 (silver coins)"),
    Achievement("BUY_ICERING", "Favour of a Higher Being", "Recieve the Snake Ring",
                "buy the Snake Ring from the snake on screen 88 (a ruby)"),
    Achievement("REACH_SEA", "Blue Ordeal", "Reach Deep Ruin.",
                "touch the top of screen 90"),
    Achievement("FIND_NOSE", "Archeologist", "Find the hidden nose.",
                "touch the nose on screen 71"),
    Achievement("WEAR_GIANTBOOTS_AND_ICERING", "Deadly Outfit",
                "Wear the Snake Ring and the giant boots at the same time.",
                "wear the Giant Boots and the Snake Ring"),
    Achievement("WIN_GIANTBOOTS_OWL", "Giant Leaps Ghost",
                "Complete Ghost of the Babe with giant boots.",
                "Ghost of the Babe ending wearing the Giant Boots since screen 1"),
    Achievement("WIN_ICERING_OWL", "Absolute Will Ghost",
                "Complete Ghost of the Babe with Snake Ring.",
                "Ghost of the Babe ending wearing the Snake Ring since screen 1"),
    Achievement("GET_OWLCAPE", "Regal Attire", "Recieve the King's Cape.",
                "Ghost of the Babe ending after chasing the red bird (tsuchinoko) to its end"),
    Achievement("WIN_OWL", "Exorcist King", "Complete Ghost of the Babe.",
                "reach the babe on screen 154"),
    Achievement("WEAR_TUNIC_YELLOWSHOES_AND_GNOMEHAT", "Gnomed Out",
                "Wear the tunic, yellow shoes and the green hat at the same time.",
                "wear the tunic, the yellow shoes and the gnome hat"),
    Achievement("WEAR_SHOES_RING_AND_OWLCAPE", "Bird Outfit",
                "Wear all garments of the most dedicated at once.",
                "wear the shoes, the Snake Ring and the owl cape"),
    Achievement("WIN_OWL_2_FALL", "Nobody's Perfect",
                "Complete Ghost of the Babe with only 2 falls.",
                "Ghost of the Babe ending with exactly 2 splats"),
    Achievement("GET_GNOME_HAT", "Disciple of the Forest Scholar",
                "How could anyone deny your intellect now?",
                "wear the gnome hat (picked up on screen 160)"),
    Achievement("TIMESTOP_GARGOYLE_TALK", "King & The Gang", "Hang with the gang.",
                "finish a conversation with the gargoyles on screen 142"),
    Achievement("PURCHASE_SCHOLAR_TUNIC", "True Scholar",
                "Become initiated amongst the forest scholars.",
                "buy the tunic from the mushroom scholar on screen 160 (9 mushrooms)"),
    Achievement("PURCHASE_YELLOW_SHOES", "Helping a Brother Out", "Many thanks, brother!",
                "buy the yellow shoes from the mould man on screen 109 (4 ghost fragments)"),
    Achievement("CHASE_AWAY_BUG", "King of the Flies", "You lose!",
                "chase the fly in Bugstalk off all 39 of its perches"),
    Achievement("FIND_9TH_CAT", "Cute Cat", "Find the 9th cat.",
                "finish a conversation with the outsider cat on screen 125"),
    Achievement("WIN_TIME_60_OWL", "Haunted Jumper", "Complete Ghost of the Babe within 1 hour.",
                "Ghost of the Babe ending in under 60 minutes of run time"),
    Achievement("WIN_TIME_30_OWL", "Possessed Jumper",
                "Complete Ghost of the Babe within 30 minutes.",
                "Ghost of the Babe ending in under 30 minutes of run time"),
)

CODES = tuple(a.code for a in ACHIEVEMENTS)


def steam_schema(path=None):
    """[(code, name, description)] from the Steam client's cached stats schema
    (``Steam/appcache/stats/UserGameStatsSchema_1061090.bin``), or None."""
    import os
    import struct

    if path is None:
        from .xnb import _steam_roots

        for root in _steam_roots():
            cand = os.path.join(root, "appcache", "stats", "UserGameStatsSchema_1061090.bin")
            if os.path.isfile(cand):
                path = cand
                break
    if path is None or not os.path.isfile(path):
        return None
    data = open(path, "rb").read()

    def cstr(p):
        e = data.index(b"\0", p)
        return data[p:e].decode("utf-8", "replace"), e + 1

    def parse(p):          # binary KeyValues
        out = {}
        while True:
            t = data[p]
            p += 1
            if t == 8:
                return out, p
            name, p = cstr(p)
            if t == 0:
                out[name], p = parse(p)
            elif t == 1:
                out[name], p = cstr(p)
            elif t in (2, 3):
                out[name] = struct.unpack_from("<i" if t == 2 else "<f", data, p)[0]
                p += 4
            elif t == 7:
                out[name] = struct.unpack_from("<Q", data, p)[0]
                p += 8
            else:
                raise ValueError("unknown KeyValues type %d" % t)

    found = []

    def walk(d):
        for v in d.values():
            if isinstance(v, dict):
                if "name" in v and "display" in v:
                    disp = v["display"]
                    found.append((v["name"], disp.get("name", {}).get("english"),
                                  disp.get("desc", {}).get("english")))
                else:
                    walk(v)

    walk(parse(0)[0])
    return found
BY_CODE = {a.code: a for a in ACHIEVEMENTS}
assert len(CODES) == 48 and len(set(CODES)) == 48
