"""Physics parity check against the real game's map and real TAS routes.

Requires Jump King to be installed (``Content/level.xnb`` is read from your
own Steam install). The community TAS routes from ShootMe/JumpKingTAS are
replayed tick for tick on the whole original world:

* ``NewGame.tas``  - base game, ends at the babe on screen 43 (3:44.859);
* ``NewBabe.tas``  - New Babe+, ends at the babe on screen 100 (5:20.637);
* ``NewGhost.tas`` - Ghost of the Babe, ends at the babe on screen 154 (5:35.138).

For every section of a route ("#Underburg 3", ...) the script checks that the
king visits the section's screen, and at the end that the right babe is
reached on the exact frame of the recorded in-game timer.

    python scripts/validate_original.py --download
    python scripts/validate_original.py --tas-dir path/to/LevelFiles
    python scripts/validate_original.py --download --route ghost
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jumpking import verify  # noqa: E402
from jumpking.level import Level  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tas-dir", help="folder with NewGame.tas, NewBabe.tas, NewGhost.tas")
    ap.add_argument("--tas", help="a single route file (with --route)")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--route", choices=sorted(verify.ROUTES), action="append",
                    help="routes to check (default: all)")
    ap.add_argument("--game-dir")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    keys = args.route or ["base", "nbp", "ghost"]
    if not (args.tas or args.tas_dir or args.download):
        ap.error("pass --download, --tas-dir DIR or --tas PATH")

    try:
        level = Level.original(args.game_dir)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    all_exact = True
    for key in keys:
        route = verify.ROUTES[key]
        if args.tas:
            text = open(args.tas, "r", encoding="utf-8", errors="replace").read()
        elif args.tas_dir:
            text = open(os.path.join(args.tas_dir, route.filename), "r", encoding="utf-8",
                        errors="replace").read()
        else:
            text = verify.download(route)
        rep = verify.replay(level, route, text)
        print("== %s (%s)" % (route.name, route.filename))
        for name, exp, seen, ok in rep.sections:
            if not args.quiet or not ok:
                print("  %-28s screen %3d  %s" % (name, exp, "ok" if ok else "MISSED (saw %s)" % seen))
        win = "%.3f s" % rep.win_time if rep.win_time is not None else "not reached"
        print("  %d / %d sections reproduced; ending %d reached at %s (recorded %.3f s): %s"
              % (rep.ok, rep.checked, rep.ending, win, rep.recorded or 0,
                 "EXACT" if rep.exact else "MISMATCH"))
        all_exact &= rep.exact
    sys.exit(0 if all_exact else 1)


if __name__ == "__main__":
    main()
