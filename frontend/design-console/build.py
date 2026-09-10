#!/usr/bin/env python3
"""Assemble the Exoplanet Hunter design console into one self-contained HTML file.

    python3 build.py            -> dist/exoplanet-hunter.html + dist/preview.html

Inputs
    src/shell.html          styles + static markup, with __FONT_*__ placeholders
    src/app.*.js            application source, concatenated in SRC_ORDER
    assets/fonts/*.woff2    latin subsets, base64-inlined (the artifact CSP
                            blocks external font hosts, so a <link> would
                            silently fall back to system fonts)
    ../node_modules/animejs anime.js v4 ESM bundle, inlined

The anime.js bundle ends in `export { a as animate, ... }`. Top-level exports
are legal in a module script but the names are not otherwise reachable, so the
export list is rewritten into `const ANIME = { animate: a, ... }` and the app
destructures from that.

Output is a single file with no network dependencies: it opens from disk, and
it is what gets published as the Claude artifact.
"""

import base64
import os
import pathlib
import re
import shutil
import sys
from urllib.parse import quote

HERE = pathlib.Path(__file__).resolve().parent
FRONTEND = HERE.parent
ANIME_BUNDLE = FRONTEND / "node_modules/animejs/dist/bundles/anime.esm.min.js"

SRC_ORDER = [
    "app.api.js",  # API base resolution, endpoint calls, hydrate()
    "app.data.js",  # data model, shared components, charts, router
    "app.health.js",  # /healthz state machine
    "app.home.js",  # Home + Catalogue
    "app.pages.js",  # Vetting + Model Performance + Upload
    "app.boot.js",  # boot preloader, then route()
]

FONTS = {
    "__FONT_INTER__": "inter-latin.woff2",
    "__FONT_JBM__": "jetbrains-mono-latin.woff2",
    "__FONT_ANURATI__": "anurati-regular.woff2",
    "__FONT_AILERONS__": "ailerons-trial.woff2",
}

# Same mechanism as the fonts, for the same reason: the artifact host's CSP
# blocks every external request, so the Mission backdrop has to travel inside
# the file. It is cropped, darkened and re-encoded by tools/backdrop.py rather
# than shipped at source quality -- see assets/backdrop/CREDIT.txt for the
# licence this image is used under, which requires the credit stay visible.
IMAGES = {
    "__BG_ORION__": "orion-hero.webp",
}

# The mark, inlined rather than linked: the console is one file and an <img>
# would be a second request the artifact host's CSP blocks anyway. Two grades of
# the same drawing -- see the comments in each file -- and the use site sizes
# them from CSS, so neither carries a width of its own.
ICONS = {
    "__ICON_MARK__": "favicon.svg",
    "__ICON_MARK_FULL__": "logo.svg",
}

# What an unfurler reads. Every one of them (LinkedIn, Slack, X, iMessage)
# wants Open Graph, and none accept a data: URI for og:image, so the card is a
# real file at a real origin: tools/og_card.py draws it, it is committed under
# assets/icon/, and it is copied into dist/ below because Render's build image
# has neither Pillow nor fontTools. preview.html shares these bytes with
# index.html by design, so a local preview carries the deployed URLs; harmless,
# and cheaper than maintaining two heads.
OG_DESCRIPTION = (
    "A calibrated deep-learning pipeline for vetting transit candidates in NASA "
    "TESS, Kepler and K2 photometry. Live scoring, with the evidence behind every number."
)
OG_IMAGE_ALT = (
    "The Exoplanet Hunter mark, a ringed planet inside an acquisition reticle, "
    "beside the wordmark on a dark grid."
)

# The favicon goes into the standalone wrapper's <head>, not into shell.html:
# that file is the artifact body, and the artifact host sets the tab icon
# itself. dist/index.html is the copy that gets deployed and needs its own.
# The SVG is percent-encoded rather than base64'd, so it stays readable in the
# built file and comes out smaller than its own base64 would be.


def main() -> int:
    if not ANIME_BUNDLE.exists():
        print(f"error: {ANIME_BUNDLE} not found — run `npm install` in {FRONTEND}", file=sys.stderr)
        return 1

    # The deployed console is a static file on one origin and the API is on
    # another, so "/api" cannot be the default there. EH_API_BASE is baked in
    # as a meta tag at build time; app.api.js reads it and a ?api= query
    # parameter still overrides it for one-off testing.
    api_base = os.environ.get("EH_API_BASE", "").strip()
    meta = f'<meta name="eh-api-base" content="{api_base}">\n' if api_base else ""

    # og:url and og:image have to be absolute, so the deployed origin is a
    # build-time input like EH_API_BASE. render.yaml supplies it.
    console_url = os.environ.get(
        "EH_CONSOLE_URL", "https://exoplanet-hunter-console.onrender.com"
    ).rstrip("/")

    shell = meta + (HERE / "src/shell.html").read_text()
    for token, filename in FONTS.items():
        blob = (HERE / "assets/fonts" / filename).read_bytes()
        shell = shell.replace(token, base64.b64encode(blob).decode())
    assert "__FONT_" not in shell, "unreplaced font placeholder"

    for token, filename in IMAGES.items():
        blob = (HERE / "assets/backdrop" / filename).read_bytes()
        shell = shell.replace(token, base64.b64encode(blob).decode())
    assert "__BG_" not in shell, "unreplaced image placeholder"

    svg = (HERE / "assets/icon/favicon.svg").read_text()
    png = (HERE / "assets/icon/favicon-32.png").read_bytes()
    social = (
        f'<meta name="description" content="{OG_DESCRIPTION}">'
        '<meta property="og:type" content="website">'
        '<meta property="og:site_name" content="Exoplanet Hunter">'
        '<meta property="og:title" content="Exoplanet Hunter">'
        f'<meta property="og:description" content="{OG_DESCRIPTION}">'
        f'<meta property="og:url" content="{console_url}/">'
        f'<meta property="og:image" content="{console_url}/og.png">'
        '<meta property="og:image:type" content="image/png">'
        '<meta property="og:image:width" content="1200">'
        '<meta property="og:image:height" content="630">'
        f'<meta property="og:image:alt" content="{OG_IMAGE_ALT}">'
        '<meta name="twitter:card" content="summary_large_image">'
    )

    icon_links = (
        '<link rel="icon" type="image/svg+xml" '
        f'href="data:image/svg+xml,{quote(svg, safe=chr(47) + chr(58))}">'
        '<link rel="alternate icon" type="image/png" sizes="32x32" '
        f'href="data:image/png;base64,{base64.b64encode(png).decode()}">'
    )

    app = "\n".join((HERE / "src" / name).read_text() for name in SRC_ORDER)

    anime = ANIME_BUNDLE.read_text()
    match = re.search(r"export\s*\{([^}]*)\}\s*;?\s*$", anime)
    if not match:
        print("error: no export statement at the end of the anime.js bundle", file=sys.stderr)
        return 1
    pairs = []
    for part in match.group(1).split(","):
        part = part.strip()
        if not part:
            continue
        local, exported = (
            [s.strip() for s in part.split(" as ")] if " as " in part else (part, part)
        )
        pairs.append(f"{exported}:{local}")
    anime = anime[: match.start()] + "const ANIME={" + ",".join(pairs) + "};\n"

    out = shell + '\n<script type="module">\n' + anime + "\n" + app + "\n</script>\n"

    # After concatenation, because the shell carries the nav mark and the About
    # page carries the full one, and one substitution over the whole file is
    # less to keep in step than two over its halves.
    for token, filename in ICONS.items():
        markup = (HERE / "assets/icon" / filename).read_text().strip()
        # A token can land inside a JS template literal -- the About page's
        # does -- where a backtick or ${ ends the string and the whole module
        # fails to parse. A comment in logo.svg did exactly that once, and the
        # only symptom was the console never leaving its boot overlay.
        bad = [c for c in ("`", "${") if c in markup]
        if bad:
            print(f"error: {filename} contains {bad} — see build.py", file=sys.stderr)
            return 1
        out = out.replace(token, markup)
    assert "__ICON_" not in out, "unreplaced icon placeholder"

    dist = HERE / "dist"
    dist.mkdir(exist_ok=True)
    (dist / "exoplanet-hunter.html").write_text(out)

    # standalone wrapper: the artifact host supplies <html>/<head>/<body>,
    # so the published file omits them and this adds them back for local viewing
    standalone = (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        # matches --void, so mobile browser chrome stops drawing its own grey
        # band above a near-black page
        '<meta name="theme-color" content="#050608">'
        "<title>Exoplanet Hunter</title>"
        + social
        + icon_links
        + "<style>body{margin:0;padding:0}</style></head><body>"
        + out
        + "</body></html>"
    )
    shutil.copyfile(HERE / "assets/icon/og-card.png", dist / "og.png")
    (dist / "preview.html").write_text(standalone)
    # index.html is what a static host serves at /. Same bytes as preview.html
    # — one file, two names, so opening from disk and deploying cannot diverge.
    (dist / "index.html").write_text(standalone)

    print(f"anime.js inlined — {len(pairs)} exports")
    print(f"dist/exoplanet-hunter.html — {len(out) / 1024:.0f} KB")
    print(f"dist/index.html + dist/preview.html — API base {api_base or '/api (default)'}")
    print(f"dist/og.png — social card, og:url {console_url}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
