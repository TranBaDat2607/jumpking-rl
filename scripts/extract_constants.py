"""Read Jump King's physics constants from your own JumpKing.exe.

The clone's constants come from public mod sources (docs/RESEARCH.md); one
(SPLAT_TIME) could not be found and is estimated. This tool reads the real
values from the installed game's .NET metadata / IL (``JumpKing.PlayerValues``)
without running it, prints them next to the clone's values and can write
``jumpking/data/constants_override.json``, which the engine loads at import.

    pip install dnfile
    python scripts/extract_constants.py                 # auto-detect install
    python scripts/extract_constants.py --write         # apply overrides
    python scripts/extract_constants.py --exe path/to/JumpKing.exe --type JumpKing.PlayerValues
"""
import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OVERRIDE = os.path.join(os.path.dirname(__file__), "..", "jumpking", "data", "constants_override.json")
# PlayerValues member -> clone constant name
MAPPING = {
    "GRAVITY": "GRAVITY", "JUMP": "JUMP", "SPEED": "SPEED", "MAX_FALL": "MAX_FALL",
    "JUMP_TIME": "JUMP_TIME", "WALK_SPEED": "WALK_SPEED", "BOUNCE": "BOUNCE",
    "ICE_FRICTION": "ICE_FRICTION", "SPLAT_TIME": "SPLAT_TIME",
}


def _row_index(ref):
    return getattr(ref, "row_index", None)


def read_method_il(pe, rva):
    if not rva:
        return b""
    hdr = pe.get_data(rva, 12)
    if hdr[0] & 0x3 == 0x2:          # tiny header
        size = hdr[0] >> 2
        return pe.get_data(rva + 1, size)
    flags_size = struct.unpack_from("<H", hdr, 0)[0]
    header_size = (flags_size >> 12) * 4
    code_size = struct.unpack_from("<I", hdr, 4)[0]
    return pe.get_data(rva + header_size, code_size)


def _operand_size(il, i):
    """Size of the instruction at ``i`` (opcode + operand), ECMA-335 III."""
    op = il[i]
    if op == 0xFE:
        op2 = il[i + 1]
        if op2 in (0x06, 0x07, 0x15, 0x16, 0x1C):
            return 6
        if op2 in (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E):
            return 4
        if op2 in (0x12, 0x19):
            return 3
        return 2
    if op in (0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x1F, 0xDE) or 0x2B <= op <= 0x37:
        return 2
    if op in (0x21, 0x23):
        return 9
    if op == 0x45:  # switch
        n = struct.unpack_from("<I", il, i + 1)[0]
        return 5 + 4 * n
    if op in (0x20, 0x22, 0x27, 0x28, 0x29, 0x6F, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75,
              0x79, 0x7B, 0x7C, 0x7D, 0x7E, 0x7F, 0x80, 0x81, 0x8C, 0x8D, 0x8F, 0xA3,
              0xA4, 0xA5, 0xC2, 0xC6, 0xD0, 0xDD) or 0x38 <= op <= 0x44:
        return 5
    return 1


def scan_il(il):
    """Yield (kind, value, token) for constant loads directly followed by
    ``stsfld`` (static initialisers) or ``ret`` (constant getters)."""
    i = 0
    last_const = None
    n = len(il)
    while i < n:
        op = il[i]
        size = _operand_size(il, i)
        if i + size > n:
            break
        if op == 0x22:
            last_const = struct.unpack_from("<f", il, i + 1)[0]
        elif op == 0x23:
            last_const = struct.unpack_from("<d", il, i + 1)[0]
        elif op == 0x20:
            last_const = struct.unpack_from("<i", il, i + 1)[0]
        elif op == 0x1F:
            last_const = struct.unpack_from("<b", il, i + 1)[0]
        elif 0x15 <= op <= 0x1E:
            last_const = op - 0x16
        elif op == 0x80:
            if last_const is not None:
                yield ("stsfld", last_const, struct.unpack_from("<I", il, i + 1)[0])
            last_const = None
        elif op == 0x2A:
            if last_const is not None:
                yield ("ret", last_const, None)
            last_const = None
        else:
            last_const = None
        i += size


def extract(exe, type_name="JumpKing.PlayerValues"):
    import dnfile

    pe = dnfile.dnPE(exe)
    md = pe.net.mdtables
    ns, _, name = type_name.rpartition(".")
    typedefs = md.TypeDef.rows
    target = None
    for idx, td in enumerate(typedefs):
        if str(td.TypeName) == name and str(td.TypeNamespace) == ns:
            target = idx
            break
    if target is None:
        raise SystemExit("type %s not found in %s" % (type_name, exe))
    td = typedefs[target]
    field_rows = [_row_index(f) for f in td.FieldList]
    method_rows = [_row_index(m) for m in td.MethodList]
    fields = {r: str(md.Field.rows[r - 1].Name) for r in field_rows if r}
    values = {}
    # 1) literal constants (C# const) live in the Constant table
    if getattr(md, "Constant", None) is not None:
        for row in md.Constant.rows:
            parent = row.Parent
            if getattr(parent, "table", None) is None:
                continue
            if parent.table.name != "Field" or parent.row_index not in fields:
                continue
            blob = bytes(row.Value.value if hasattr(row.Value, "value") else row.Value)
            t = row.Type
            if t == 0x0C:
                v = struct.unpack("<f", blob[:4])[0]
            elif t == 0x0D:
                v = struct.unpack("<d", blob[:8])[0]
            elif t == 0x08:
                v = struct.unpack("<i", blob[:4])[0]
            else:
                continue
            values[fields[parent.row_index]] = v
    # 2) static readonly fields assigned in .cctor, 3) trivial property getters
    for r in method_rows:
        if not r:
            continue
        m = md.MethodDef.rows[r - 1]
        mname = str(m.Name)
        il = read_method_il(pe, m.Rva)
        for kind, v, tok in scan_il(il):
            if kind == "stsfld" and mname == ".cctor":
                table, row = tok >> 24, tok & 0xFFFFFF
                if table == 0x04 and row in fields:
                    values.setdefault(fields[row], v)
            elif kind == "ret" and mname.startswith("get_") and len(il) <= 10:
                values.setdefault(mname[4:], v)
    return values


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe")
    ap.add_argument("--type", default="JumpKing.PlayerValues")
    ap.add_argument("--write", action="store_true", help="write constants_override.json")
    args = ap.parse_args()
    exe = args.exe
    if exe is None:
        from jumpking.xnb import find_game_dir

        d = find_game_dir()
        exe = os.path.join(d, "JumpKing.exe") if d else None
    if not exe or not os.path.isfile(exe):
        sys.exit("JumpKing.exe not found (install the game or pass --exe)")
    values = extract(exe, args.type)
    from jumpking import constants as C

    print("%-14s %-14s %s" % ("member", "game", "clone"))
    override = {}
    for k in sorted(values):
        clone = getattr(C, MAPPING.get(k, k), None)
        print("%-14s %-14r %r" % (k, values[k], clone))
        if k in MAPPING:
            override[MAPPING[k]] = float(values[k])
    if args.write:
        with open(OVERRIDE, "w", encoding="utf-8") as fh:
            json.dump(override, fh, indent=1)
        print("wrote", os.path.normpath(OVERRIDE))


if __name__ == "__main__":
    main()
