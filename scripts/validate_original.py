"""Physics parity check against the real game's map and a real TAS route.

Requires Jump King to be installed (``Content/level.xnb`` is read from your
own Steam install) and the base-game TAS from ShootMe/JumpKingTAS
(``LevelFiles/NewGame.tas``), either as a path or fetched with --download.

The TAS is split into sections named after the screen being climbed
("#Redcrown Woods 3", ...). The script replays it in the clone on the original
map and reports, section by section, whether the king is on the expected
screen, i.e. how far the clone's physics reproduce the real game tick for
tick.

    python scripts/validate_original.py --download
    python scripts/validate_original.py --tas path/to/NewGame.tas
"""
import argparse
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jumpking import constants as C  # noqa: E402
from jumpking import physics as P  # noqa: E402
from jumpking import tas  # noqa: E402
from jumpking.game import Game  # noqa: E402
from jumpking.level import Level  # noqa: E402

TAS_URL = "https://raw.githubusercontent.com/ShootMe/JumpKingTAS/master/LevelFiles/NewGame.tas"


def expected_screen(section):
    name = re.sub(r"\(.*?\)", "", section).strip()
    m = re.match(r"(.+?)\s+(\d+)$", name)
    if not m:
        return None
    area, k = m.group(1).replace("Kings", "King's"), int(m.group(2))
    for a_name, first, last in C.AREAS:
        if a_name.lower() == area.lower():
            return first + k - 1
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tas")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--game-dir")
    ap.add_argument("--start-tick", type=int, default=-2,
                    help="wind/game clock value when the TAS places the king")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.tas:
        text = open(args.tas, "r", encoding="utf-8", errors="replace").read()
    elif args.download:
        text = urllib.request.urlopen(TAS_URL, timeout=30).read().decode("utf-8", "replace")
    else:
        ap.error("pass --tas PATH or --download")

    try:
        level = Level.original(args.game_dir)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    # split into sections and keep the input-tick index where each starts
    sections = []
    current = None
    buf = []
    for line in text.splitlines():
        if line.startswith("#") and not line.startswith("#!"):
            if current is not None:
                sections.append((current, "\n".join(buf)))
            current, buf = line[1:].strip(), []
        else:
            buf.append(line)
    if current is not None:
        sections.append((current, "\n".join(buf)))

    game = Game(level)
    started = False
    ok = 0
    checked = 0
    for name, body in sections:
        segs = tas.parse(body)
        exp = expected_screen(name)
        if exp is not None and started:
            checked += 1
            actual = game.screen + 1
            status = "ok" if actual == exp else "MISMATCH"
            if not args.quiet or actual != exp:
                print("%-32s expected screen %2d  clone screen %2d  %s" % (name, exp, actual, status))
            if actual == exp:
                ok += 1
            else:
                print("  first divergence; king at (%.2f, %.2f) after %d ticks" % (
                    game.x, game.y, game.ticks))
                break
        for seg in segs:
            if seg.state is not None and not started:
                x, y, facing = seg.state
                game.reset(x, y)
                game.state[P.S_TICK] = args.start_tick
                started = True
                # JumpKingTAS counts the "@" line as one frame: the state is
                # applied and the game then runs one tick without input
                game.step(0, False)
                continue
            if not started:
                continue
            for _ in range(seg.frames):
                game.step(seg.direction, seg.jump)
    print("\n%d / %d sections reproduced; goal reached: %s" % (ok, checked, game.done))


if __name__ == "__main__":
    main()
