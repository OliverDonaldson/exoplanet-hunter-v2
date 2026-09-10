#!/usr/bin/env python3
"""Rasterise the small-grade mark to assets/icon/favicon-32.png.

`build.py` inlines `favicon.svg` for browsers that take an SVG icon and this
PNG for the ones that do not. The PNG is generated rather than drawn so the two
cannot diverge: the geometry below is `favicon.svg`'s, in the same 120-unit
space, and changing one without the other is the failure this file exists to
prevent.

Drawn at 8x and downsampled, because Pillow's shape primitives do not
antialias. Supersampling is what makes the ring's 4.5-unit stroke survive the
trip to 32 px as a curve instead of a staircase.

Reproduce with  python3 tools/icon.py
"""

from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "assets/icon/favicon-32.png"

SIZE = 32
SS = 8  # supersample factor
UNITS = 120  # the SVG's viewBox
TEAL = (77, 255, 210, 255)
GROUND = (5, 6, 8, 255)


def main() -> None:
    px = SIZE * SS
    s = px / UNITS  # viewBox units -> supersampled pixels

    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, px - 1, px - 1], radius=int(26 * s), fill=GROUND)
    d.ellipse([(60 - 20) * s, (60 - 20) * s, (60 + 20) * s, (60 + 20) * s], fill=TEAL)

    # The ring is drawn on its own layer and rotated, because Pillow has no
    # rotated-ellipse primitive and an arc cannot carry the -20 degrees.
    ring = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ring)
    rd.ellipse(
        [(60 - 38) * s, (60 - 11) * s, (60 + 38) * s, (60 + 11) * s],
        outline=TEAL,
        width=max(1, round(4.5 * s)),
    )
    img.alpha_composite(ring.rotate(20, resample=Image.BICUBIC, center=(60 * s, 60 * s)))

    tick = max(1, round(6 * s))
    for x0, y0, x1, y1 in ((60, 8, 60, 22), (60, 98, 60, 112), (8, 60, 22, 60), (98, 60, 112, 60)):
        d.line([x0 * s, y0 * s, x1 * s, y1 * s], fill=TEAL, width=tick)

    img.resize((SIZE, SIZE), Image.LANCZOS).save(OUT, optimize=True)
    print(f"wrote {OUT.relative_to(HERE.parent)} — {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
