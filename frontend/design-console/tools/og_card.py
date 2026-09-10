#!/usr/bin/env python3
"""Draw the link-preview card to dist/og.png.

Every unfurler (LinkedIn, Slack, X, iMessage) reads Open Graph, and none of
them accept a data: URI for og:image — it has to be a file at an https URL.
Committed rather than generated at deploy time, the same way favicon-32.png is:
Render's build image runs `python3 build.py` with no Pillow and no fontTools.
build.py copies this to dist/og.png, which render.yaml publishes at
<console>/og.png, and points og:image there.

The mark is `assets/icon/logo.svg`'s geometry re-drawn in the same 120-unit
space, for the same reason tools/icon.py exists: generated rather than traced,
so the card and the SVG cannot drift. The wordmark uses the console's own
Anurati, decompressed from the shipped .woff2 in memory rather than checked in
a second time.

Drawn at 2x and downsampled: Pillow's shape primitives do not antialias, and
the reticle's 1.4-unit dashes do not survive to 1200 px without it.

Reproduce with  python3 tools/og_card.py
"""

from __future__ import annotations

import io
import math
import pathlib

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
FONTS = HERE.parent / "assets/fonts"
OUT = HERE.parent / "assets/icon/og-card.png"

#: 1.91:1, which is what every unfurler crops to.
W, H = 1200, 630
SS = 2

GROUND = (5, 6, 8, 255)
TEAL = (77, 255, 210, 255)
BONE = (240, 238, 232, 255)
SLATE = (138, 143, 168, 255)
GRID = (77, 255, 210, 26)

#: The mark's box on the card, in final pixels.
MARK_BOX = 300
MARK_CX, MARK_CY = 240, 315

#: Two lines, as on the identity sheet: Anurati is a display face and one long
#: line of it at a legible size does not fit beside the mark.
TITLE = ["EXOPLANET", "HUNTER"]
SUB = [
    "Calibrated deep-learning vetting of transit",
    "candidates in NASA TESS, Kepler and K2 photometry.",
]
STRAP = "5-FOLD DUAL-VIEW CNN  ·  PLATT-CALIBRATED  ·  LIVE"
EYEBROW = "NASA TESS  ·  KEPLER  ·  K2"
URL = "exoplanet-hunter-console.onrender.com"

#: Text starts here and must end before this. A card that silently clips its
#: own wordmark is the failure this pair of constants exists to catch.
TEXT_X = 470
TEXT_RIGHT = W - 56


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    """Pillow needs a TTF; the console ships woff2. fontTools does the rest."""
    f = TTFont(FONTS / name)
    f.flavor = None
    buf = io.BytesIO()
    f.save(buf)
    buf.seek(0)
    return ImageFont.truetype(buf, size)


def _tracked(d: ImageDraw.ImageDraw, xy, text, font, fill, tracking: int) -> int:
    """Draw with explicit letter spacing and return the width used. Anurati is
    a display face and its own spacing is far too tight at this size."""
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) + tracking
    return int(x - xy[0] - tracking)


def _mark(size: int) -> Image.Image:
    """logo.svg, in its own 120-unit space, at `size` pixels square."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size / 120

    def box(cx, cy, rx, ry):
        return [(cx - rx) * s, (cy - ry) * s, (cx + rx) * s, (cy + ry) * s]

    # Reticle: stroke-dasharray "3 7" on r=48. Pillow has no dash support, so
    # the pattern is walked as arcs — 3 units of 301.6 is 3.58 degrees.
    period = 360 * 10 / (2 * math.pi * 48)
    for i in range(round(360 / period)):
        a = i * period
        d.arc(
            box(60, 60, 48, 48),
            a,
            a + period * 0.3,
            fill=(77, 255, 210, 71),
            width=max(1, round(1.4 * s)),
        )

    d.ellipse(
        box(60, 60, 19, 19), fill=(77, 255, 210, 41), outline=TEAL, width=max(1, round(1.8 * s))
    )

    # Its own layer, because Pillow has no rotated-ellipse primitive and the
    # ring carries the SVG's rotate(-20). Same trick as tools/icon.py.
    ring = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(ring).ellipse(box(60, 60, 36, 10), outline=TEAL, width=max(1, round(1.8 * s)))
    img.alpha_composite(ring.rotate(20, resample=Image.BICUBIC, center=(60 * s, 60 * s)))

    tick = max(1, round(2.4 * s))
    for x0, y0, x1, y1 in (
        (60, 4, 60, 18),
        (60, 102, 60, 116),
        (4, 60, 18, 60),
        (102, 60, 116, 60),
    ):
        d.line([x0 * s, y0 * s, x1 * s, y1 * s], fill=TEAL, width=tick)
    return img


def main() -> None:
    img = Image.new("RGBA", (W * SS, H * SS), GROUND)
    d = ImageDraw.Draw(img)

    # On its own layer: ImageDraw replaces pixels rather than blending, so a
    # low-alpha fill drawn straight onto the ground comes out at full strength.
    grid = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(grid)
    for x in range(0, W * SS, 60 * SS):
        gd.line([x, 0, x, H * SS], fill=GRID, width=SS)
    for y in range(0, H * SS, 60 * SS):
        gd.line([0, y, W * SS, y], fill=GRID, width=SS)
    img.alpha_composite(grid)

    img.alpha_composite(
        _mark(MARK_BOX * SS),
        ((MARK_CX - MARK_BOX // 2) * SS, (MARK_CY - MARK_BOX // 2) * SS),
    )

    anurati = _font("anurati-regular.woff2", 66 * SS)
    inter = _font("inter-latin.woff2", 26 * SS)
    mono = _font("jetbrains-mono-latin.woff2", 16 * SS)

    x = TEXT_X * SS
    _tracked(d, (56 * SS, 44 * SS), EYEBROW, mono, SLATE, 2 * SS)

    for i, line in enumerate(TITLE):
        used = _tracked(d, (x, (172 + i * 84) * SS), line, anurati, BONE, 8 * SS)
        if TEXT_X + used / SS > TEXT_RIGHT:
            raise SystemExit(f"error: '{line}' is {used / SS:.0f}px and runs past {TEXT_RIGHT}px")

    d.line([x, 372 * SS, (x + 110 * SS), 372 * SS], fill=TEAL, width=2 * SS)
    for i, line in enumerate(SUB):
        d.text((x, (400 + i * 36) * SS), line, font=inter, fill=SLATE)
    _tracked(d, (x, 496 * SS), STRAP, mono, TEAL, 2 * SS)

    url_w = d.textlength(URL, font=mono) + 2 * SS * (len(URL) - 1)
    _tracked(d, (TEXT_RIGHT * SS - url_w, 552 * SS), URL, mono, SLATE, 2 * SS)

    img.convert("RGB").resize((W, H), Image.LANCZOS).save(OUT, optimize=True)
    print(f"wrote {OUT.relative_to(HERE.parent)} — {W}x{H}, {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
