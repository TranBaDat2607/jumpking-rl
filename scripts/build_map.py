"""Regenerate the built-in map (jumpking/data/level.png + level_meta.json + docs/images/map.png)."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jumpking import mapgen  # noqa: E402
from jumpking.level import BUILTIN_META, BUILTIN_PNG  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-png", default=BUILTIN_PNG)
    ap.add_argument("--out-meta", default=BUILTIN_META)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()
    t = time.time()
    level, route = mapgen.build(seed=args.seed, verbose=not args.quiet)
    mapgen.save(level, route, args.out_png, args.out_meta, args.seed)
    print("built %r with %d route actions in %.1fs -> %s" % (
        level, len(route), time.time() - t, args.out_png))
    overview = mapgen.save_overview(level)
    if overview:
        print("overview ->", overview)


if __name__ == "__main__":
    main()
