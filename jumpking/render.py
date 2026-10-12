"""Fast numpy software renderer (no window needed).

Used for RGB observations, videos and map previews. The pygame front end
blits these frames too, so headless and interactive views are identical.
"""
import numpy as np

from . import constants as C
from . import physics as P

# per-area palette: (sky top, sky bottom, solid, solid edge)
AREA_COLOURS = {
    "Redcrown Woods": ((38, 64, 52), (78, 110, 70), (58, 44, 30), (112, 160, 64)),
    "Colossal Drain": ((24, 30, 36), (44, 54, 60), (70, 82, 78), (110, 128, 110)),
    "False King's Keep": ((40, 36, 56), (70, 62, 86), (92, 88, 96), (150, 146, 150)),
    "Bargainburg": ((90, 120, 160), (170, 160, 140), (110, 74, 50), (180, 90, 60)),
    "Great Frontier": ((120, 170, 210), (190, 210, 220), (96, 86, 70), (140, 170, 90)),
    "Windswept Bluff": ((150, 170, 190), (200, 205, 210), (90, 90, 96), (180, 180, 190)),
    "Stormwall Pass": ((70, 80, 100), (130, 140, 160), (80, 84, 96), (210, 215, 225)),
    "Chapel Perilous": ((50, 30, 40), (100, 60, 70), (88, 70, 76), (170, 140, 120)),
    "Blue Ruin": ((20, 40, 80), (50, 80, 130), (60, 80, 120), (120, 160, 210)),
    "The Tower": ((20, 16, 40), (60, 40, 80), (70, 60, 70), (160, 130, 90)),
    "Unknown": ((30, 30, 30), (60, 60, 60), (100, 100, 100), (160, 160, 160)),
}
ICE = (150, 220, 255)
SNOW = (245, 245, 250)
NOWIND = None
PLAYER = (200, 40, 40)
PLAYER_CHARGE = (255, 200, 40)
PLAYER_SPLAT = (120, 20, 20)
CROWN = (255, 210, 60)
BABE = (255, 150, 200)


class Renderer:
    """Renders a ``Level`` + player state to (H, W, 3) uint8 arrays."""

    def __init__(self, level, scale=1.0):
        self.level = level
        self.scale = scale
        self._cache = {}

    # -------------------------------------------------------- backgrounds
    def screen_background(self, index):
        if index in self._cache:
            return self._cache[index]
        lvl = self.level
        area = lvl.area_names[index] if 0 <= index < lvl.n_screens else "Unknown"
        top, bottom, solid, edge = AREA_COLOURS.get(area, AREA_COLOURS["Unknown"])
        h, w = C.SCREEN_H, C.SCREEN_W
        t = np.linspace(0.0, 1.0, h)[:, None]
        sky = (np.array(top)[None, :] * (1 - t) + np.array(bottom)[None, :] * t)
        img = np.repeat(sky[:, None, :], w, axis=1).astype(np.uint8)
        grid = lvl.screen_grid(index)
        # tile layer at 8x8
        tile_rgb = np.zeros((C.SCREEN_TILES_H, C.SCREEN_TILES_W, 3), np.uint8)
        mask = np.zeros((C.SCREEN_TILES_H, C.SCREEN_TILES_W), bool)
        for code, col in ((P.T_SOLID, solid), (P.T_ICE, ICE), (P.T_SNOW, SNOW)):
            m = grid == code
            tile_rgb[m] = col
            mask |= m
        big = np.repeat(np.repeat(tile_rgb, 8, 0), 8, 1)
        bigm = np.repeat(np.repeat(mask, 8, 0), 8, 1)
        img[bigm] = big[bigm]
        # grass / edge highlight on exposed tops
        exposed = mask.copy()
        exposed[1:] &= ~mask[:-1]
        for r, c in zip(*np.nonzero(exposed & (grid == P.T_SOLID))):
            img[r * 8:r * 8 + 2, c * 8:c * 8 + 8] = edge
        # slopes: draw the solid triangle
        yy, xx = np.mgrid[0:8, 0:8]
        tri = {
            P.T_SLOPE_TL: (xx + yy) >= 7,
            P.T_SLOPE_TR: (yy - xx) >= 0,
            P.T_SLOPE_BL: (xx - yy) >= 0,
            P.T_SLOPE_BR: (xx + yy) <= 7,
        }
        for code, m in tri.items():
            for r, c in zip(*np.nonzero(grid == code)):
                cell = img[r * 8:r * 8 + 8, c * 8:c * 8 + 8]
                cell[m] = solid
        # no-wind zones: faint overlay
        for r, c in zip(*np.nonzero(grid == P.T_NOWIND)):
            cell = img[r * 8:r * 8 + 8, c * 8:c * 8 + 8]
            cell[:] = (cell.astype(np.int32) * 3 // 4 + 64).astype(np.uint8)
        # goal (babe)
        gx0, gy0, gx1, gy1 = lvl.goal
        top_y = -C.SCREEN_H * index
        if top_y <= gy0 < top_y + C.SCREEN_H:
            x0, x1 = int(gx0), int(gx1)
            y0, y1 = int(gy0 - top_y), int(gy1 - top_y)
            img[max(0, y0):y1, max(0, x0):x1] = BABE
            img[max(0, y0):y0 + 4, max(0, x0):x1] = CROWN
        self._cache[index] = img
        return img

    # ------------------------------------------------------------ frames
    def frame(self, state=None, screen=None, trail=None):
        """Full-resolution 480x360 RGB frame of the camera screen."""
        if screen is None:
            screen = int(state[P.S_SCREEN]) if state is not None else 0
        img = self.screen_background(screen).copy()
        top_y = -C.SCREEN_H * screen
        if trail:
            for (tx, ty) in trail:
                px, py = int(tx) + C.PLAYER_W // 2, int(ty) - top_y + C.PLAYER_H // 2
                if 0 <= px < C.SCREEN_W and 0 <= py < C.SCREEN_H:
                    img[max(0, py - 1):py + 1, max(0, px - 1):px + 1] = (255, 255, 255)
        if state is not None:
            self.draw_player(img, state, top_y)
        if self.scale != 1.0:
            img = self.resize(img, self.scale)
        return img

    def draw_player(self, img, s, top_y):
        x0 = int(s[P.S_X])
        y0 = int(s[P.S_Y]) - top_y
        w, h = C.PLAYER_W, C.PLAYER_H
        col = PLAYER
        if s[P.S_SPLAT] > 0:
            col = PLAYER_SPLAT
            y0 += h // 2
            h = h - h // 2
        elif s[P.S_CHARGING] > 0:
            col = PLAYER_CHARGE
            squash = int(6 * min(1.0, s[P.S_CHARGE] / C.JUMP_TIME))
            y0 += squash
            h -= squash
        ya, yb = max(0, y0), min(C.SCREEN_H, y0 + h)
        xa, xb = max(0, x0), min(C.SCREEN_W, x0 + w)
        if ya < yb and xa < xb:
            img[ya:yb, xa:xb] = col
            # crown
            cy = max(0, y0 - 3)
            img[cy:max(cy, y0), xa:xb] = CROWN
            # facing eye
            ex = x0 + (12 if s[P.S_FACING] >= 0 else 3)
            ey = y0 + 6
            if 0 <= ey < C.SCREEN_H - 2 and 0 <= ex < C.SCREEN_W - 3:
                img[ey:ey + 3, ex:ex + 3] = (255, 255, 255)

    @staticmethod
    def resize(img, scale):
        if scale == 1.0:
            return img
        if scale < 1.0:
            k = int(round(1.0 / scale))
            h, w = img.shape[0] // k * k, img.shape[1] // k * k
            v = img[:h, :w].reshape(h // k, k, w // k, k, img.shape[2]).mean(axis=(1, 3))
            return v.astype(np.uint8)
        k = int(round(scale))
        return np.repeat(np.repeat(img, k, 0), k, 1)

    def map_preview(self, screens=None, columns=11, scale=0.5, label=True):
        """Grid image of many screens (bottom-left = screen 1)."""
        lvl = self.level
        if screens is None:
            screens = list(range(lvl.n_screens))
        rows = (len(screens) + columns - 1) // columns
        tiles = [self.resize(self.screen_background(i), scale) for i in screens]
        th, tw = tiles[0].shape[:2]
        pad = 4
        out = np.full((rows * (th + pad) + pad, columns * (tw + pad) + pad, 3), 16, np.uint8)
        for k, (i, t) in enumerate(zip(screens, tiles)):
            col = k % columns
            row = rows - 1 - k // columns
            y = pad + row * (th + pad)
            x = pad + col * (tw + pad)
            out[y:y + th, x:x + tw] = t
        return out

    def tall_preview(self, first, last, scale=0.5, state_trail=None):
        """Vertical strip of screens first..last (top = last), like the game."""
        strips = []
        for i in range(last, first - 1, -1):
            strips.append(self.resize(self.screen_background(i), scale))
            strips.append(np.full((2, strips[-1].shape[1], 3), 255, np.uint8))
        return np.concatenate(strips[:-1], axis=0)


def save_image(img, path):
    from PIL import Image

    Image.fromarray(img).save(path)
