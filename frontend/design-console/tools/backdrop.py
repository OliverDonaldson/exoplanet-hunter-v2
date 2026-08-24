#!/usr/bin/env python3
"""Derive the Mission page backdrop from its source frame.

`build.py` base64-inlines `assets/backdrop/orion-hero.webp` into the single-file
console. This script is what produces that file, kept in the repo so the
transform is reproducible rather than a one-off somebody ran once and forgot.

Three decisions are baked in here rather than in CSS:

  Crop to 16:9. The hero is a wide band and `background-size: cover` discards
  whatever does not fit at render time. Cropping first is the same visible
  result with fewer pixels to encode. Either axis can be the one in excess, so
  the crop works out which and trims that one about its centre.

  Lift the midtones, then pull the black back down. The source is a night sky:
  mostly near-black, with the nebula and dust living in the lower midtones.
  Scaling brightness linearly raises the empty sky as fast as the nebula and the
  void stops reading as void, so a gamma curve does the lifting and a black
  point pushes the sky back down afterwards. Measured over the frame, this takes
  the nebula's bright half up about 30% while the empty sky moves barely at all.

  The contrast the headline needs is bought by the gradient in the stylesheet,
  over the left third where the type actually sits, which leaves the rest of the
  frame free. Saturation comes down slightly so the nebula does not fight the
  teal the UI is built on.

  Encode at high quality. A night sky is mostly low-amplitude gradient, which is
  exactly where WebP bands, and banding is the one artefact that would read as a
  rendering fault rather than as compression.

Provenance: assets/backdrop/CREDIT.txt. The frame is generated artwork, not an
observation, which is why nothing on the page captions it as one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy
from PIL import Image, ImageEnhance

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets" / "backdrop"

SOURCE = ASSETS / "orion-source.jpeg"
OUTPUT = ASSETS / "orion-hero.webp"

ASPECT = 16 / 9
GAMMA = 0.62  # < 1 lifts midtones, leaves 0 at 0
BLACK_POINT = 0.10  # re-seats the sky after the lift, in post-gamma units
SATURATION = 0.90
QUALITY = 94


def main() -> int:
    if not SOURCE.exists():
        print(f"error: {SOURCE} not found — see CREDIT.txt for its origin", file=sys.stderr)
        return 1

    src = Image.open(SOURCE).convert("RGB")
    width, height = src.size

    if width / height >= ASPECT:
        target_width = round(height * ASPECT)
        left = (width - target_width) // 2
        box = (left, 0, left + target_width, height)
    else:
        target_height = round(width / ASPECT)
        top = (height - target_height) // 2
        box = (0, top, width, top + target_height)

    frame = src.crop(box)

    levels = numpy.asarray(frame, dtype=numpy.float32) / 255.0
    levels = numpy.power(levels, GAMMA)
    levels = numpy.clip((levels - BLACK_POINT) / (1.0 - BLACK_POINT), 0.0, 1.0)
    frame = Image.fromarray((levels * 255.0).astype("uint8"))
    frame = ImageEnhance.Color(frame).enhance(SATURATION)
    frame.save(OUTPUT, "WEBP", quality=QUALITY, method=6)

    kb = OUTPUT.stat().st_size / 1024
    print(f"{OUTPUT.name} — {frame.size[0]}x{frame.size[1]}, {kb:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
