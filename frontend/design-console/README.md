# Design console

The Exoplanet Hunter vetting console: Mission, Catalogue, Vetting, Model
Performance, Discovery, Upload and About, built as one HTML file with fonts and
anime.js inlined.

This is the shipping console: Render builds it with `build.py` and serves
`dist/index.html` (see `render.yaml`). It talks to the API through
`src/app.api.js`, live or mock decided once per load by probing `/healthz`.
With no service reachable it falls back to mock data shaped like the real
contracts, and any figure the API does not serve renders as "not measured"
rather than as a number.

## Build

```bash
cd frontend && npm install          # provides animejs@4
cd design-console && python3 build.py
open dist/preview.html
```

`build.py` inlines the anime.js ESM bundle from `../node_modules/animejs` and
base64-inlines the three woff2 subsets in `assets/fonts/`, then concatenates
`src/app.*.js` in a fixed order into a single `<script type="module">`.

Three outputs:

| file | purpose |
| --- | --- |
| `dist/index.html` | the deployed page; what Render serves at `/` |
| `dist/preview.html` | the same bytes, for opening from disk |
| `dist/exoplanet-hunter.html` | body-only, for a host that supplies its own `<html>`/`<body>` |

The fonts are vendored rather than linked because the artifact host applies a
strict CSP that blocks external font hosts — a `<link>` to Google Fonts would
silently fall back to system fonts and the design would look wrong.

## Source layout

| file | contents |
| --- | --- |
| `src/shell.html` | design tokens, all CSS, static markup (nav, ticker, boot overlay) |
| `src/app.data.js` | candidate data, branch/fold/diagnostic/follow-up models, chart renderer, router |
| `src/app.health.js` | the `/healthz` state machine behind the service-status panel |
| `src/app.home.js` | Mission (home) and Catalogue |
| `src/app.pages.js` | Vetting, Model Performance, Upload |
| `src/app.boot.js` | boot preloader, then the initial `route()` |

## What the design asserts

Points worth carrying into the real console, because each one is a claim about
how the product should behave:

- **Branch evidence is a first-class view.** The eleven input views each get a
  signed contribution and a plain-English reading. Contributions are
  attributions on the probability scale, summing from the catalogue mean to the
  calibrated score, so the arithmetic on screen is checkable.
- **Diagnostics have three states, not two.** pass / fail / **not measured**.
  About 7% of TESS rows carry no Data Validation report; rendering an unmeasured
  flag as a green tick would reintroduce the exact defect the presence-mask
  machinery exists to prevent.
- **No pooled headline metric.** Per-mission cards driven by the served model's
  own evaluation list, with out-of-fold and zero-shot labelled differently and
  never compared. The card count comes from the data, not from a fixed layout.
- **Every headline number carries its interval.** The project measured its noise
  floor (AUC ±0.0070, shortlist recall ±0.0337); a bare figure invites reading
  noise as improvement.
- **Recall @ 1% FPR is a headline**, not a footnote — it is the promotion
  criterion and the thing "would this reach the shortlist" actually means.
- **Follow-up priority is surfaced.** TSM/ESM against Kempton (2018) thresholds,
  because ranking candidates for telescope time is the product's stated purpose.
- **Observation baseline is visible** wherever a score is, since long baselines
  inflate detectability and scores are not comparable across very different ones.
- **The console is never gated on a warm model.** See below.

## Service status panel

`src/app.health.js`. Polls `/healthz` and renders four states. The copy matters
because three of the four are fast:

| state | condition | UI |
| --- | --- | --- |
| Connecting | request in flight, < 2 s | small spinner — most visits never leave this |
| Waking | request in flight, > 2 s | "Waking the observatory…" — the suspend-resume path |
| Warming model | 200 with `ensemble_ready: false` | determinate bar against `uptime_s / 90` |
| Ready | `ensemble_ready: true` | teal dot, the served `model_version` |

A fifth state, Degraded, covers `status: "degraded"` (no promoted run in the
registry): scoring unavailable, catalogue still browsable.

Three rules the panel follows:

1. **Browsing is never gated.** `/candidates` and `/reliability` are parquet
   reads that answer in ~0.8 s whatever the ensemble is doing. Only Score waits.
2. **Name the served model once known**, so the console cannot quietly drift
   from what is actually deployed.
3. **Determinate bar only for the ~90 s ensemble load.** A bar that completes in
   2 s reads as broken; a spinner that runs for 90 s reads as hung.

### Measured timings this is designed around

| state | TTFB | what happened |
| --- | --- | --- |
| resume from suspend | 2.0 s | RAM snapshot restored, TF ensemble already in memory |
| warm | 0.14 s | steady state |
| `/candidates` | 0.78 s | parquet read |
| `/reliability` | 0.87 s | parquet read |

`fly.toml` uses `auto_stop_machines = "suspend"`, which snapshots RAM rather
than killing the process, so the common idle path is a ~2 s resume with the
model still loaded — not a cold start. A true cold start (stopped machine, or
post-deploy) runs `dvc pull` from R2, then FastAPI starts, then a background
thread imports TensorFlow and five folds: 60–180 s, and the Fly health check
allows a 180 s grace period.

### ⚠ Contract gap

The deployed `GET /healthz` currently returns only:

```json
{ "status": "ok", "model_loaded": true, "model_version": "cnn_dualview-cv-ca906040" }
```

The **Warming** state needs two fields that do not exist yet:

- `ensemble_ready: bool` — `model_loaded` is not a substitute; it only means a
  promoted run exists in `registry.json`, and the TF ensemble is loaded lazily
  on the first `/score` request.
- `uptime_s: number` — to drive the determinate bar against the ~90 s load.

Until those land, the client degrades to connecting → waking → ready, which is
correct for the suspend-resume path but skips cold-start progress entirely.
`phaseFor()` handles the absence explicitly; no API change is assumed.

### Wiring it to the real endpoint

The prototype runs a scripted clock so the states can be seen. To point it at
the live service, set `HEALTH.live = true` in `src/app.health.js` — `fetchHealth()`
is already written and sits next to the simulation. The prototype plays one
honest cold start per session (`sessionStorage`), then behaves like a warm
service, which is what the real thing does; the ↻ button replays the sequence.

## Service status

The panel on the Mission page is a **live reader**, not a one-shot. It polls
`/healthz` every second while connecting, waking or warming, and every 15 s once
ready — slow enough not to hold the Fly machine awake for a page left open,
often enough to notice. It stops while the tab is hidden and says so rather than
going on asserting READY off an ageing reading. Every figure on it comes from
the last ping: the model version the service reported, the round trip measured
here, and the age counting up to the next one. A ping that does not come back
renders as NO ANSWER with the error, not as CONNECTING.

It used to stop polling the moment it first reached ready, and printed a
hardcoded `0.14 s steady state` underneath — a single reading, presented as a
live one, that would still have said READY with the service long gone. The
prototype mode keeps that documented figure and labels it as the prototype's.

## Boot preloader

`src/app.boot.js`. An instrument dial in the shape of the anime.js hero
animation — glowing segmented outer ring, dense rotating tick ring, glass disc
with a specular highlight, thin concentric arcs drawn on with a stagger —
rendered in our palette, with our motif at the centre: a planet transiting its
star, tracing the light-curve dip.

The light curve is drawn on the star's own line (`y = 180`, the centre of the
360×360 viewBox) with the dip below it, so the planet riding the path crosses
the star it dims. It used to sit 66 units lower, which put the whole transit
under the star and made the two read as unrelated objects.

There is no particle field. It was a 7×7 lattice of circles drifting ±70 units
with nothing clipping them, and at boot resolution it read as noise over the
instrument rather than as candidates being scored.

anime.js features used: `svg.createDrawable` with `draw` for the ring segments,
arcs and the transit trace; `svg.createMotionPath` for the transiting planet;
`stagger` for the segment and tick reveals; `text.scrambleText` for the status
readout.

Skipped entirely under `prefers-reduced-motion`. A 9 s hard timeout guarantees
the console is never left behind the overlay if the animation stalls.

**Cleanup matters here.** `root.remove()` detaches the nodes but anime.js keeps
ticking anything still animating them, so a `loop: true` animation on a removed
element runs for the rest of the session. `finish()` cancels the handles it
holds *and* calls `utils.remove()` over the overlay's subtree, because the
scramble readout is re-animated on every stage change and the handles alone do
not cover it.

**Background tabs.** `engine.pauseOnDocumentHidden` is left at its default
(`true`). Setting it to `false` does not keep motion running in a hidden tab —
`requestAnimationFrame` does not fire there either way — it only skips the
`resetTime()` that `engine.resume()` performs, so the first frame back applies
the entire hidden interval at once and every animation snaps to its end state.
The 9 s failsafe is what keeps the overlay from parking in front of the console.

**Nothing that animates may carry a layout transform.** anime.js owns the whole
`transform` property on any element it touches, so the `translateX(-50%)` that
was centring `.boot-meter` was replaced by the timeline's `translateY` on the
first frame, and the meter spent every boot half its own width to the right. It
centres with `margin: 0 auto` now. The same rule is why the dial's own groups
carry no transform attribute in the markup.

**Short viewports.** The meter is pinned to the bottom while the dial and
readout are centred above it, so the two collided below about 660 px of height —
46 px of overlap at 1280x620. `#boot` reserves the meter's strip as padding and
the dial takes a `66vh` cap, so it gives way before the text does.

**Reviewing it:** `dist/preview.html?boot=hold` freezes the dial fully assembled,
just before it hands over, so the composition can be looked at without catching
a 4-second animation.

## Mission backdrop

`assets/backdrop/orion-hero.webp`, base64-inlined by `build.py` through the same
placeholder mechanism as the fonts, because the artifact host's CSP blocks every
external request. 258 KB, taking the single file from 442 KB to 829 KB.

It replaces a canvas that drew 900 procedural stars on every resize, and whose
own comment called it a stand-in for a hero photograph that never arrived.

`tools/backdrop.py` derives it: crop to 16:9 about the centre, gamma 0.62 with a
0.10 black point, saturation 0.90, WebP at quality 94. The lift is a curve
rather than a brightness multiplier because the source is a night sky. Scaling
it linearly raises the empty sky as fast as the nebula and the void stops
reading as void; a gamma curve lifts the midtones where the dust lives and the
black point puts the sky back down afterwards. Over the frame that is roughly
+30% on the nebula's bright half against almost no movement in the empty sky.

**Contrast is measured, not judged.** The check maps the real text bounding
boxes from the live page into image pixels, applies the sky opacity and the
gradient alpha at each position, and reads the 99.5th percentile luminance under
each block -- a single star in a glyph gap is not what legibility turns on.
Desktop lands at 5.05:1 worst case, against 4.5:1 for AA body text.

**Two gradients, because the copy moves.** On a wide screen the type sits in the
left third, so a 100deg ramp darkens that side and leaves the rest of the frame
alone. Below 820 px the copy spans the full width and that same ramp left the
paragraph at **2.02:1** -- a real failure, found by measuring rather than by
looking. Narrow screens get a flat vertical scrim instead, which reads 5.20:1.

**The frame is generated artwork, not an observation**, and the page says so.
On a console whose whole argument is that every number on it was measured, a
synthetic sky captioned like a real one would be the one fabricated thing on
screen. `assets/backdrop/CREDIT.txt` records the provenance, and notes what
would have to change to put a real, attributed frame back.

## Vetting: the pipeline timeline

The Vetting page opens on **Pipeline** — nine stages between "a TIC went in" and
"this number came out", each with what it actually produced, read back from that
target's own `/score` response. Ephemeris and its provenance (`catalogue`,
`user` or `bls`), bin coverage for each view, the five fold scores and their
spread, the MC-dropout passes and sigma, the Platt shift from raw mean to
calibrated score, the threshold and verdict, and which diagnostic suites
returned and which flagged.

**Stage 2 is deliberately empty.** `ScoreResponse` carries the binned views and
not the cadences they were built from, so the unprocessed light curve cannot be
drawn and the stage says so. Showing it would need the raw series added to the
score contract, which is an API change.

Coverage counts come from `viewCoverage()` in `app.api.js`: `flux` is null in a
phase bin no cadence landed in, so `filled / total` is a real measure and array
length is not.

A target scored from **Upload** that has no catalogue row still gets this page.
`rememberScoredTarget()` keeps the score against the target and returns the id
to route to; an unknown TIC becomes an `adHocCandidate` carrying the ephemeris
and the score and nothing else. Depth, T-mag, SNR, sectors and disposition are
ExoFOP columns the score does not return, so they render as not measured and
the follow-up panel — TSM and ESM are functions of depth and T-mag — is dropped
rather than computed from zeros.

**Upload and Vetting are the same scoring call.** Upload takes an identifier and
shows the answer; Vetting takes a target already on the page and shows the
working. The result panel hands off to `#/vetting/<id>` rather than dropping the
user on the catalogue.

## View state in the URL

The hash is the route. State that belongs to a view rather than to the route
rides in a query string on that hash, so the address bar is the whole of what
the screen is showing and can be pasted to someone else:

```
#/catalogue?q=TOI-43&disp=PC&src=TESS&sort=period&dir=asc&min=20
#/vetting/TOI-4328.01?tab=diagnostics
```

`?api=` and `?boot=` are unaffected — they live in `location.search`, and a
relative `replaceState` keeps them.

**Written with `replaceState`, never by assigning `location.hash`.** Assigning
the hash raises `hashchange`, which is `route()`, which would rebuild the page
underneath the control that was just used: the search box would lose focus and
its caret on every keystroke. Replacing also keeps filter changes out of the
history stack, so Back leaves the catalogue rather than stepping back through
fourteen keystrokes — and the single entry it does keep carries the current
filters, so Back out of a candidate returns to the catalogue you were reading
rather than to a reset one. `history.state` is passed through, so the parked
scroll offset survives the rewrite.

**Read on demand, not handed down through `page()`,** because a view can
re-enter itself. Vetting does, when a slow `/score` finally lands. That re-entry
used to throw you back to the default tab; it now reads the URL again and keeps
the tab you switched to during the wait — which a 20-60 s cold score invites.
The same change moved that repaint's guard off the whole hash and onto the route,
because the tab now rides on the hash and comparing the whole thing would have
made a score arrive to a hash that no longer matched and never paint at all.

**Only what differs from the default is written,** so an untouched catalogue
stays `#/catalogue` rather than carrying six parameters that say nothing.

**Every value is checked against the set that produced it,** and the URL is
normalised to what is actually on screen: `?min=999` becomes `min=90`, `?min=37`
snaps to `min=35` (the slider's step), and `?disp=BOGUS`, `?dir=sideways` or
`?sort=nonsense` are dropped. The sort key is the one that fails quietly rather
than loudly if it is not checked — the comparator would read `undefined` on both
rows, call every pair a tie, and render the catalogue in load order underneath a
header with nothing marked, which is an arbitrary order presented as a sorted
one. `score_std` is a spread rather than an order the table offers, and one
predicate now decides both which headers are clickable and which `?sort=` values
are accepted, so a URL cannot ask for a sort the header row has no way to show
or undo.

**Still open:** the pages with no view state of their own leave a stray query
alone rather than stripping it. Nothing reads it, and stripping it would fight
whatever wants to put state there next.

## Web Interface Guidelines

`AGENTS.md` at the repo root is Vercel's Web Interface Guidelines. Nine gaps
against it were closed:

- **Deep links work again.** Every load used to be rewritten to `#/`. Only a bare
  URL is now, and an unrecognised path falls through to Mission.
- **The URL reflects view state.** Catalogue filters, search, sort and the
  vetting tab are all in it, and all restore from it. See *View state in the URL*
  above for the scheme and why it is written with `replaceState`.
- **Scroll position survives Back.** `history.scrollRestoration` is `manual` and
  the offset is parked in the history entry itself, so it travels with the entry.
  Parked on a trailing timeout rather than `requestAnimationFrame`, because rAF
  does not fire in a hidden tab and "scrolled, switched tabs, came back, hit
  Back" is exactly when losing the offset stings.
- **Inputs are 16 px on touch.** Below that iOS Safari zooms the viewport in on
  focus and does not zoom out. The catalogue search was 12 px and the TIC field
  14.4 px. Scoped to `(hover: none) and (pointer: coarse)` so the desktop type
  scale is untouched: the bug is the zoom, not the size.
- **No `transition: all`.** Sixteen of them, now naming their properties.
- **Skip-to-content link**, ahead of seven nav links on every page.
- **`<meta name="theme-color">`** so mobile chrome stops drawing grey above a
  near-black page.
- **Touch targets at 44 px.** Measured at 375 px: filter chips 41x29, menu button
  42x32, candidate shortcuts 95x28, health replay worst at 18x17. Grown with a
  centred overlay rather than padding, so the hit area changes and nothing moves.
  A range input cannot use a pseudo-element for this, so it gets padding instead.

**One is deliberately not met.** The guidelines want `<title>` to track the
current view; it reads "Exoplanet Hunter" everywhere by choice. The console is
one product and the tab is its name, not a readout of which page is open.

**One cannot be met without giving up something asked for.** Sort headers in the
catalogue are 35x14 before the fit scale, so about 5 px of tap target once the
table is scaled to a phone. Growing them would overlap adjacent columns, and the
table is scaled precisely so it keeps its desktop layout. Sorting on a phone is
the cost of that.

## Narrow screens

Below **900 px** the seven nav links move into a panel behind a menu button;
they need about 810 px of bar in Ailerons and the wordmark is already hidden by
1150. Below **820 px** the bottom ticker goes — it is a horizontal marquee that
shows two cells at a time on a phone and costs a fixed strip of screen for them
— and the space every page reserves for it is given back.

**Tables are scaled, not reflowed.** A fifteen-column comparison turned into one
column per row is fifteen unrelated lines, which is not what a catalogue is for.
`fitTables()` measures each `[data-fit-table]` wrapper and sets `zoom` on the
table so the whole thing fits the viewport at desktop layout. `zoom` rather than
a transform, because it scales the layout box too, so the wrapper still sizes to
what is drawn and the hit targets stay under the text.

Two things make the scale affordable. Cell gutters drop from 1 rem a side to
0.4 rem below 820 px — 270 px of the catalogue's 1224 was gutter — and the
catalogue's `min-width` drops with them, from 1180 to 940, or the fit would be
paying for 240 px of table that is no longer there. Together the full catalogue
lands at about 0.36 and fits a 390 px phone exactly. `FIT_FLOOR` is a backstop
below which the type stops resolving at all; past it the wrapper scrolls.

The fit runs on `route()` and on every catalogue repaint — sorting and filtering
change the widest cell in several columns — and is coalesced onto a frame only
for resize.
