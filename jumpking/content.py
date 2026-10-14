"""Game content read at runtime from a local Jump King install.

Everything the campaign needs besides the hitbox map lives in the game's own
content files: world items, NPC dialogue and trigger boxes, merchants and their
prices, raven paths, achievement hitboxes, illusory walls and skin layers
(``Content/props/**``, ``Content/king/skin_settings.xml``), and the English
dialogue text in ``LanguageJK.dll`` (whose length sets how long a conversation
lasts). None of it is copied into this repository; ``GameContent.load()``
parses it from your install, the way the game's XmlSerializer does.
"""
import glob
import os
import struct
import xml.etree.ElementTree as ET

from .xnb import find_game_dir


def _text(e, tag, default=None):
    x = e.find(tag)
    if x is None or x.text is None:
        return default
    return x.text.strip()


def _int(e, tag, default=0):
    v = _text(e, tag)
    return default if v in (None, "") else int(v)


def _float(e, tag, default=0.0):
    v = _text(e, tag)
    return default if v in (None, "") else float(v)


def _point(e, tag):
    x = e.find(tag)
    if x is None:
        return (0, 0)
    return (_int(x, "X"), _int(x, "Y"))


def _rect(e):
    return (_int(e, "X"), _int(e, "Y"), _int(e, "Width"), _int(e, "Height"))


def _quote(e):
    """An OldManQuote: its list of line keys (``None`` if absent)."""
    if e is None:
        return None
    lines = e.find("lines")
    if lines is None:
        return None
    return [(s.text or "") for s in lines.findall("string")]


def _quotes(e):
    if e is None:
        return []
    return [_quote(q) for q in e.findall("OldManQuote")]


def _files(folder, ext=".xml"):
    """UltraContent.LoadXmlFiles order (directory listing, i.e. by name)."""
    return sorted(glob.glob(os.path.join(folder, "*" + ext)), key=lambda p: os.path.basename(p).lower())


def texture_size(path):
    """(width, height) of an uncompressed XNB Texture2D, read from its header."""
    with open(path, "rb") as fh:
        data = fh.read(512)
    if data[:3] != b"XNB" or data[5] & 0xC0:
        raise ValueError("unsupported XNB: %s" % path)
    pos = 10

    def read7(pos):
        value = shift = 0
        while True:
            b = data[pos]
            pos += 1
            value |= (b & 0x7F) << shift
            if not b & 0x80:
                return value, pos
            shift += 7

    n, pos = read7(pos)
    for _ in range(n):
        m, pos = read7(pos)
        pos += m + 4
    _, pos = read7(pos)
    _, pos = read7(pos)
    _, width, height = struct.unpack_from("<iii", data, pos)
    return width, height


class OldManSettings:
    """Port of JumpKing.MiscEntities.OldMan.OldManSettings (logic fields)."""

    def __init__(self, e):
        self.name = _text(e, "name")
        self.position = _point(e, "position")
        self.home_screen = _int(e, "home_screen")
        tb = e.find("trigger_box")
        self.trigger_width = _int(tb, "width") if tb is not None else 0
        self.trigger_height = _int(tb, "height") if tb is not None else 0
        self.trigger_cx = _float(tb, "center_x") if tb is not None else 0.0
        self.trigger_cy = _float(tb, "center_y") if tb is not None else 0.0
        sr = tb.find("screen_rect") if tb is not None else None
        self.trigger_screen_rect = _rect(sr) if sr is not None else (0, 0, 0, 0)
        self.intro_quote = _quote(e.find("intro_quote"))
        self.talk_delay = _float(e, "talk_delay")
        self.talking_speed = _float(e, "talking_speed")
        self.achievements = [(_text(a, "code"), _int(a, "talk_times"))
                             for a in e.findall("achievements/OldmanTalkAchievement")]
        self.screens = [(_int(s, "screen"), _quotes(s.find("quotes")))
                        for s in e.findall("screens/OldManScreen")]
        self.looping_lines = _quotes(e.find("looping_lines")) or None
        self.flag_quotes = [(_text(f, "flag"), _quotes(f.find("quotes")))
                            for f in e.findall("flag_quotes/OldManFlagQuote")] or None
        self.falls = [(_int(f, "fallStart"), _int(f, "fallEnd"), _quotes(f.find("quotes")))
                      for f in e.findall("falls/OldManFall")] or None
        self.items = [(_text(i, "item"), _quotes(i.find("quotes")))
                      for i in e.findall("items/OldManItem")] or None

    def real_position(self):
        x, y = self.position
        return x, y - (self.home_screen - 1) * 360

    def trigger_rect(self):
        """OldManHitbox.GetRect(position) (C# integer arithmetic)."""
        px, py = self.real_position()
        if self.trigger_screen_rect[2] != 0:
            x, y, w, h = self.trigger_screen_rect
            q = py - 360
            q = int(q / 360)          # C# division truncates toward zero
            return (x, y + q * 360, w, h)
        x = int(px - self.trigger_width * self.trigger_cx)
        y = int(py - self.trigger_height * self.trigger_cy)
        return (x, y, self.trigger_width, self.trigger_height)


class MerchantSettings:
    def __init__(self, e):
        self.sale_item = _text(e, "sale_item")
        self.currency_type = _text(e, "currency_type")
        self.base_price = _int(e, "base_price")
        self.price_increase = _int(e, "price_increase")
        self.sale_achievements = [(_text(a, "code"), _int(a, "sell_amount"))
                                  for a in e.findall("sale_achievements/SaleAchievement")]
        self.sale_lines = _quotes(e.find("sale_lines"))
        self.sold_lines = _quotes(e.find("sold_lines"))
        self.no_gold_lines = _quotes(e.find("no_gold_lines"))
        self.sell_quote = _quote(e.find("sell_quote"))
        self.settings = OldManSettings(e.find("settings"))


class RavenSettings:
    def __init__(self, name, e):
        self.name = name
        self.texture = _text(e, "texture")
        self.item = _text(e, "item")
        self.positions = []
        for p in e.findall("positions/RavenPosition"):
            trig = _text(p, "trigger_distance")
            self.positions.append({
                "screen": _int(p, "screen"),
                "position": _point(p, "position"),
                "treasure": _text(p, "treasure") == "true",
                "look_direction": _text(p, "look_direction", "Left"),
                "fly_direction": _text(p, "fly_direction", "Left"),
                "trigger_distance": float(trig) if trig not in (None, "") else 40.0,
            })


class GameContent:
    """Everything the campaign reads from the install (see module docstring)."""

    def __init__(self):
        self.root = None
        self.skins = {}             # item -> list of layers (skin_settings.xml order)
        self.world_items = []       # (item, screen, x, y)
        self.item_sizes = {}        # item -> (w, h) of its world sprite
        self.ravens = []            # RavenSettings, by file name
        self.old_men = []           # OldManSettings, by file name
        self.merchants = []         # MerchantSettings, by file name
        self.achievement_hitboxes = []   # (code, [rects in world coordinates])
        self.illusory_walls = []    # (code, [rects in world coordinates])
        self.strings = {}           # dialogue key -> English text

    @classmethod
    def load(cls, game_dir=None):
        game_dir = game_dir or find_game_dir()
        if not game_dir or not os.path.isdir(os.path.join(game_dir, "Content")):
            raise FileNotFoundError(
                "Jump King's Content folder was not found. Install the game through Steam "
                "(or set JUMPKING_GAME_DIR).")
        c = cls()
        c.root = game_dir
        content = os.path.join(game_dir, "Content")
        props = os.path.join(content, "props")

        for sk in ET.parse(os.path.join(content, "king", "skin_settings.xml")).getroot().iter("Skin"):
            c.skins[_text(sk, "item")] = [l.text.strip() for l in sk.findall("layers/SkinLayer")]

        root = ET.parse(os.path.join(props, "worlditems", "worlditems.xml")).getroot()
        for w in root.iter("WorldItemState"):
            c.world_items.append((_text(w, "item"), _int(w, "screen"), _int(w, "_x"), _int(w, "_y")))
        sprites = {"GoldRing": "gold_ring", "Ruby": "ruby", "Silver": "silver_coin",
                   "GiantBoots": "shoes_iron", "GnomeHat": "gnome_hat", "Tunic": "tunic",
                   "YellowShoes": "yellow_shoes", "Cap": "cap", "Shroom": "shroom",
                   "GhostFragment": "ghost_fragment", "BugNote": "bug_note"}
        for item, name in sprites.items():
            path = os.path.join(props, "worlditems", name + ".xnb")
            if os.path.isfile(path):
                c.item_sizes[item] = texture_size(path)

        for path in _files(os.path.join(props, "textures", "raven"), ".ravset"):
            name = os.path.splitext(os.path.basename(path))[0]
            c.ravens.append(RavenSettings(name, ET.parse(path).getroot()))
        for path in _files(os.path.join(props, "textures", "old_man", "lines")):
            c.old_men.append(OldManSettings(ET.parse(path).getroot()))
        for path in _files(os.path.join(props, "textures", "old_man", "merchant")):
            c.merchants.append(MerchantSettings(ET.parse(path).getroot()))

        for path in _files(os.path.join(props, "achievements")):
            e = ET.parse(path).getroot()
            screen = _int(e, "screen")
            x, y, w, h = _rect(e.find("hitbox"))
            c.achievement_hitboxes.append((_text(e, "code"), [(x, y - (screen - 1) * 360, w, h)]))

        walls = os.path.join(props, "hidden_walls")
        for path in _files(walls):
            base = os.path.splitext(os.path.basename(path))[0]
            screen = int(base.replace("hidden_wall", ""))
            for d in ET.parse(path).getroot().iter("RaymanData"):
                codes = [a.text.strip() for a in d.findall("achievements/AchievementCode")]
                if not codes:
                    continue
                px, py = _point(d, "Position")
                dy = -(screen - 1) * 360
                screen_coords = _text(d, "use_screen_coords") == "true"
                boxes = [_rect(r) for r in d.findall("hitboxes/Rectangle")]
                if not boxes:
                    w, h = texture_size(os.path.join(walls, "textures", _text(d, "texture_name") + ".xnb"))
                    boxes = [(0, 0, w, h)]
                rects = []
                for (x, y, w, h) in boxes:
                    if screen_coords:
                        rects.append((x, y + dy, w, h))
                    else:
                        rects.append((x + px, y + py + dy, w, h))
                for code in codes:
                    c.illusory_walls.append((code, rects))

        c.strings = _load_strings(os.path.join(game_dir, "LanguageJK.dll"))
        return c

    def line_text(self, key):
        """The text TargetLine shows for a dialogue key (LanguageJK.Util)."""
        if key and "\n" in key:
            key = key.replace("\n", "")
        text = self.strings.get(key.replace("\r", "") if key else key)
        if text is None:
            text = key or ""
        else:
            text = text.replace("<newline>", "\n")
        for a, b in (("’", "'"), ("‘", "'"), (" .", "."), ("“", '"'), ("”", '"')):
            text = text.replace(a, b)
        return text


def _load_strings(dll_path):
    """The English resources of LanguageJK.dll (``language.resources``)."""
    if not os.path.isfile(dll_path):
        return {}
    try:
        import dnfile
    except ImportError:  # pragma: no cover - dnfile is a dependency
        return {}
    pe = dnfile.dnPE(dll_path)
    out = {}
    for res in pe.net.resources or []:
        if not str(res.name).endswith("language.resources"):
            continue
        for entry in res.data.entries:
            if isinstance(entry.data, (bytes, bytearray)):
                out[str(entry.name)] = bytes(entry.data).decode("utf-8", "replace")
            elif isinstance(entry.data, str):
                out[str(entry.name)] = entry.data
    return out
