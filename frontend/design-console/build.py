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
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
FRONTEND = HERE.parent
ANIME_BUNDLE = FRONTEND / "node_modules/animejs/dist/bundles/anime.esm.min.js"

SRC_ORDER = [
    "app.data.js",  # data model, shared components, charts, router
    "app.health.js",  # /healthz state machine
    "app.home.js",  # Home + Catalogue
    "app.pages.js",  # Vetting + Model Performance + Upload
    "app.boot.js",  # boot preloader, then route()
]

FONTS = {
    "__FONT_INTER__": "inter-latin.woff2",
    "__FONT_JBM__": "jetbrains-mono-latin.woff2",
    "__FONT_SG__": "space-grotesk-latin.woff2",
}


def main() -> int:
    if not ANIME_BUNDLE.exists():
        print(f"error: {ANIME_BUNDLE} not found — run `npm install` in {FRONTEND}", file=sys.stderr)
        return 1

    shell = (HERE / "src/shell.html").read_text()
    for token, filename in FONTS.items():
        blob = (HERE / "assets/fonts" / filename).read_bytes()
        shell = shell.replace(token, base64.b64encode(blob).decode())
    assert "__FONT_" not in shell, "unreplaced font placeholder"

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

    dist = HERE / "dist"
    dist.mkdir(exist_ok=True)
    (dist / "exoplanet-hunter.html").write_text(out)

    # standalone wrapper: the artifact host supplies <html>/<head>/<body>,
    # so the published file omits them and this adds them back for local viewing
    (dist / "preview.html").write_text(
        "<!doctype html><html><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        "<style>body{margin:0;padding:0}</style></head><body>" + out + "</body></html>"
    )

    print(f"anime.js inlined — {len(pairs)} exports")
    print(f"dist/exoplanet-hunter.html — {len(out) / 1024:.0f} KB")
    print("dist/preview.html — open this one directly in a browser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
