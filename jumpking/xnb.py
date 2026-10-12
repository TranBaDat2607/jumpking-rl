"""Read Jump King's hitbox atlas (``Content/level.xnb``) from a local install.

The game ships the atlas as an uncompressed XNB5 ``Texture2D`` in Color
(RGBA8) format. Nothing is copied into this repository: the file is read from
the user's own Steam installation at runtime.
"""
import os
import re
import struct

import numpy as np

STEAM_APP_DIR = os.path.join("steamapps", "common", "Jump King")


def _read_7bit(data, pos):
    value = 0
    shift = 0
    while shift < 35:
        b = data[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, pos
        shift += 7
    raise ValueError("invalid 7-bit encoded integer")


def read_texture(path):
    """Return the texture as an (H, W, 4) uint8 RGBA array."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:3] != b"XNB":
        raise ValueError("%s is not an XNB file" % path)
    version, flags = data[4], data[5]
    if version != 5:
        raise ValueError("unsupported XNB version %d" % version)
    if flags & 0xC0:
        raise ValueError("compressed XNB files are not supported (flags=0x%02x)" % flags)
    (length,) = struct.unpack_from("<I", data, 6)
    if length != len(data):
        raise ValueError("XNB length mismatch")
    pos = 10
    n_readers, pos = _read_7bit(data, pos)
    readers = []
    for _ in range(n_readers):
        n, pos = _read_7bit(data, pos)
        readers.append(data[pos:pos + n].decode("utf-8"))
        pos += n + 4  # name + version
    shared, pos = _read_7bit(data, pos)
    if shared:
        raise ValueError("shared resources are not supported")
    reader, pos = _read_7bit(data, pos)
    if not 1 <= reader <= len(readers) or "Texture2DReader" not in readers[reader - 1]:
        raise ValueError("expected a Texture2D")
    surface, width, height, mips = struct.unpack_from("<iiii", data, pos)
    pos += 16
    (size,) = struct.unpack_from("<I", data, pos)
    pos += 4
    if surface != 0 or size != width * height * 4:
        raise ValueError("expected an RGBA Color texture")
    pixels = np.frombuffer(data, dtype=np.uint8, count=size, offset=pos)
    return pixels.reshape(height, width, 4).copy()


def _steam_roots():
    roots = []
    if os.name == "nt":
        try:
            import winreg

            for hive, key, name in (
                (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam", "InstallPath"),
            ):
                try:
                    with winreg.OpenKey(hive, key) as k:
                        roots.append(winreg.QueryValueEx(k, name)[0])
                except OSError:
                    pass
        except ImportError:  # pragma: no cover
            pass
        roots += [r"C:\Program Files (x86)\Steam", r"C:\Program Files\Steam"]
    else:
        home = os.path.expanduser("~")
        roots += [os.path.join(home, ".steam", "steam"),
                  os.path.join(home, ".local", "share", "Steam"),
                  os.path.join(home, "Library", "Application Support", "Steam")]
    libraries = []
    for root in roots:
        root = os.path.normpath(root)
        if root not in libraries:
            libraries.append(root)
        vdf = os.path.join(root, "steamapps", "libraryfolders.vdf")
        if os.path.exists(vdf):
            try:
                with open(vdf, "r", encoding="utf-8", errors="ignore") as fh:
                    for m in re.finditer(r'"path"\s+"([^"]+)"', fh.read()):
                        p = os.path.normpath(m.group(1).replace("\\\\", "\\"))
                        if p not in libraries:
                            libraries.append(p)
            except OSError:
                pass
    return libraries


def find_game_dir():
    """Locate the Jump King install directory (``None`` if absent)."""
    env = os.environ.get("JUMPKING_GAME_DIR")
    if env and os.path.isdir(env):
        return env
    for lib in _steam_roots():
        d = os.path.join(lib, STEAM_APP_DIR)
        if os.path.isdir(d):
            return d
    return None


def find_level_xnb(game_dir=None):
    """Path of the original ``Content/level.xnb`` or ``None``."""
    candidates = []
    if game_dir:
        candidates.append(game_dir)
    else:
        found = find_game_dir()
        if found:
            candidates.append(found)
    for d in candidates:
        p = os.path.join(d, "Content", "level.xnb")
        if os.path.isfile(p):
            return p
    return None
