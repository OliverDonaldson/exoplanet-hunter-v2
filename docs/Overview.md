## Exoplanet Hunter V2

# What it is
An automated, cloud-hosted machine-learning system that decides which transit signals in NASA's TESS, Kepler and K2 data are likely to be real planets — and, crucially, tells you how much to trust each decision.

The framing from your own architecture doc still holds: V1 was about where things live (moving storage and orchestration off the laptop). V2 is about making the whole thing run itself — self-refreshing, self-validating, self-serving, with a live inference layer anyone can hit in a browser. The governing principle, borrowed from your DATA305 sequence-modelling lecture, is beat the baseline before you cheer: every component has to beat the simplest thing that already works, or it doesn't ship.

# The problem it addresses
When a space telescope watches a star and the brightness dips periodically, that's a transit — possibly a planet crossing in front. Most of the time it isn't. The dip could be an eclipsing binary star, a background star's eclipse bleeding into the aperture, instrumental systematics, or stellar variability.

Separating these is called vetting, and it's the bottleneck. TESS produces far more candidates than humans can examine. Each one traditionally needs an expert to look at phase-folded light curves, centroid motion, odd-versus-even transit depths, and secondary eclipses. There are currently ~7,000 unvetted TESS candidates in the catalogue this system tracks.

The purpose is to triage that queue: rank candidates so human attention goes where it pays, and attach a calibrated probability rather than a bare yes/no.

# How it works, end to end
1. Catalogue refresh. Pulls labelled targets from the NASA Exoplanet Archive via TAP — confirmed planets and false positives from TESS (TOI), Kepler (KOI) and K2 (k2pandc) — plus the unvetted candidate list from ExoFOP. Kepler negatives are restricted to DR25-certified false positives, so the "not a planet" class is genuinely certain rather than merely unconfirmed. Current data of record: 5,686 labelled targets (2,656 TESS / 2,500 Kepler / 530 K2) yielding 5,380 training examples.

2. Validation gates. Five Pandera-based gates run before anything trains: schema checks on the label and candidate catalogues, structural checks on the processed views, a leakage guard that quarantines label flips into a prospective holdout, and a shrink guard that fails the run if the catalogue loses more than 10% of its rows or drops a mission entirely. That last one exists because a bug once silently rewrote the catalogue from 5,686 rows to 1,000.

3. Preprocessing. Downloads the light curve, cleans it, then flattens with a Savitzky-Golay filter with the transit masked out so the detrending can't eat the signal it's meant to preserve. Phase-folds at the known ephemeris into two views: a global view (2,001 bins, full orbit) and a local view (201 bins, ±3 transit durations). Adds a 13-dimensional auxiliary vector — stellar temperature, radius, surface gravity, magnitude, transit depth/duration/period, plus vetting diagnostics.

4. Training. A dual-view 1D CNN — two convolutional towers, one per view, fused with the auxiliary features. Trained 5-fold cross-validated with MC-Dropout for uncertainty, then Platt-scaled for calibration (not temperature scaling — temperature has no bias term and can't correct a distribution shift, which cost an 0.136 ECE regression before it was fixed).

5. The promotion gate. A new run only becomes the served model if it beats the incumbent's cross-validated ROC-AUC without degrading Brier score or calibration error. It has rejected two of your own retrains — that's the discipline working, not failing.

6. Serving. FastAPI on Fly.io (exoplanet-hunter-api.fly.dev), scale-to-zero. GET /score/{tic_id} fetches the light curve, runs the full preprocessing and ensemble, and returns a calibrated probability plus the diagnostic panels. A React console on Render presents it for human vetting.

7. Automation. A weekly Saturday job refreshes the catalogue, runs the gates, retrains only if the data changed materially, runs the promotion gate, and versions everything to Cloudflare R2 via DVC.

8. Statistical validation. For the top-ranked candidates, a TRICERATOPS layer computes false-positive probability (FPP) and nearby-FPP (NFPP) from the actual pixel data and surrounding stars — the background-eclipsing-binary discrimination a light-curve-only CNN is structurally blind to. That's the run in your terminal right now.

# Where it stands
Model ca906040, live: 5-fold CV ROC-AUC 0.9581 ± 0.0057, Brier 0.0791, ECE 0.0276. Pooled out-of-fold ECE 0.0129 — well-calibrated, meaning a stated 0.9 really does mean roughly 90%.

Sensitivity, measured not assumed: injection–recovery gives 50% completeness at S/N ≈ 15 and 90% at S/N ≈ 44. That's a defensible sensitivity statement of the kind cross-validated AUC cannot give you.

Candidate shortlist: 3,919 TESS candidates scored, 50 above probability 0.9, currently undergoing FPP/NFPP validation.

# What makes it distinctive
Honestly, not the CNN — dual-view CNNs for transit classification are well-trodden. Three other things are:

It runs itself. Refresh, validate, retrain-if-warranted, gate, publish, serve. Most student ML projects are a notebook and a number.

It's calibrated and interactive. Not a leaderboard score but a live service returning probabilities you can act on, with the diagnostic evidence attached.

It is adversarial about its own results. This is the part I'd put first. Three examples:

The injection–recovery run includes a zero-depth control arm, which revealed that 26.4% of hosts pass threshold with no injected signal at all (46.7% for planet hosts, 12.3% for false-positive hosts). The model is partly scoring the star, not the transit.
Scoring the real candidates confirmed it independently: probability correlates with observation baseline at +0.21 but with the number of transits actually observed at −0.003. It rewards how long a target was watched, not how much transit evidence was collected.
A 13-dimensional vetting-feature retrain came back a dead-flat null (ΔAUC −1.3×10⁻⁵) and was rejected rather than rationalised.
Negative results like these are what separate a system you can trust from one that merely reports a good number.

# What it aims to achieve
Near term — a trustworthy shortlist. Rank the ~7,000 unvetted TESS candidates, attach FPP/NFPP to the top of that list, and produce a defensible set worth human follow-up. Not "discover a planet" — narrow the search space with stated confidence.

The current push — close the gap the measurements exposed. The system knows two concrete things about its own weakness: it reads the host rather than the transit, and TESS AUC (0.906) trails Kepler (0.989) by 8 points on the mission that actually matters. The rebuild, inspired directly by NASA's ExoMiner++, addresses both by restructuring inputs and architecture: per-diagnostic convolutional branches instead of one fused aux vector, a branch that sees each transit separately rather than only the stacked fold, and difference-image data that locates the signal at pixel level. The success criterion is explicit and falsifiable — the transit-count correlation must move off zero, and the 26.4% control rate must fall.

Longer term — a genuine vetting instrument. Per-branch explanations in the console so a human sees why the model said what it said; a prospective holdout that scores candidates before their dispositions are published, which is the only honest test of a vetting system; and the Mission Control interface redesign, deliberately held until last so it's built around real evidence rather than rebuilt twice.

# Honest limitations
It vets known ephemerides — it does not search raw light curves for undiscovered signals. Its labels inherit the archive's biases. It's blind to anything requiring pixel-level information, which is exactly why the TRICERATOPS layer exists and why difference images are the priority upgrade. And it has never been evaluated prospectively on candidates whose true disposition was unknown at scoring time; that clock started 25 July when the weekly refresh first published successfully.
