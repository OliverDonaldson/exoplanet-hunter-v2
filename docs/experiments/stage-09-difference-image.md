> Moved verbatim from `docs/roadmap.md` §4.2 on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

### 4.2 Stage 9 — difference-image branch · 6–9 h build · 3–4 h compute

**Stage 9 *(old 2(d))* — difference-image branch.** The only genuine *build*
left in the model, with quality attention. Blocked on a known problem: the stamps
are **11–17 px, not the fixed 33×33** the design assumed — that is Kepler's size
— so they must be re-gridded to a fixed size first.

**What.** Re-grid the 11–17 px stamps to a fixed size, then the branch with
quality attention. View-set rebuild (~95 min) plus 2 CV runs.

**Why here.** The last genuine build in the model, and the direct instrument
against **W2**: a centroid shift under the transit is how a background eclipsing
binary is caught — the host-scoring pathology at its source rather than at its
symptom. After stage 8, or the distribution moves under it and it is measured
twice.

**Deliverable.** A branch model carrying the difference-image branch, measured
against a post-stage-8 control.

**Stops if.** Re-gridding costs more than ~3 h or is lossy enough to need a
design decision. That is a stop-and-ask, not a judgement call to make alone.

#### The blocker dissolved — the stamps were never sparse (2026-08-17)

**The stop-condition did not fire, and the reason is a measurement rather than a
judgement.** Every claim below is over the whole archive — all 6,484 DV reports,
33,540 difference images on the selected TCE — not a sample.

| what the note assumed | what the archive says |
|---|---|
| sparse pixels on a variable bounding box | **dense**: `n_pixels / box area` is **1.000 at the minimum**, zero repeated coordinates |
| re-gridding is a resampling, so lossy | scattering the list back into a rectangle is **exact** |
| 11–17 px | **11–25 px**; 95.8% are exactly 11x11, 99.88% are within 11–17 |
| — | 0 parse failures, 0 reports with no difference image |

So the "re-grid" is a placement, not an interpolation, and it cost well under
the 3 h the stop-condition names.

**The grid is 17x17, and the criterion is the peak pixel.** A centroid shift is
read from *where the difference is brightest*, so a crop that keeps most of the
flux but moves the peak out of frame has destroyed the measurement while looking
almost lossless. Peak pixels lost by a centred crop:

| grid | stamps cropped | **peak pixel lost** | mean flux lost, of those cropped | padding on a typical 11x11 |
|---|---:|---:|---:|---:|
| 11x11 | 4.24% | 366 (1.09%) | 5.06% | 0% |
| 13x13 | 1.06% | 109 (0.33%) | 6.64% | 28.4% |
| 15x15 | 0.28% | 24 (0.07%) | 1.49% | 46.2% |
| **17x17** | **0.12%** (39) | **0** | 2.28% | 58.1% |
| 25x25 | 0 | 0 | 0 | 80.6% |

**17x17 is the smallest grid that loses no peak pixel.** 25x25 is the smallest
fully lossless one and is rejected: it buys 39 stamps' edge rows at the price of
leaving a typical stamp in 19% of its own view. **What that throws away, stated
plainly:** 39 stamps of 33,540 lose edge rows, mean 2.28% of their absolute flux
and at worst 18.2%. Nothing is interpolated — one grid cell is one CCD pixel on
every target, because the branch's subject is *where* flux moved, and a
per-target pixel scale is the defect `_periodogram_view` and `_centroid_view`
already refuse.

#### A third state nobody had looked for — DV declines 26.6% of its own images

**DV writes a sector it did not measure as every pixel `value="0.0"` with
`uncertainty="-1.0"`** — its documented "attempted, undefined" sentinel, applied
per pixel. **14,154 of 53,118 difference images (26.6%) are in that state.** It
is all-or-nothing: an image carries the sentinel on every pixel or on none, with
**nothing in between**, and every declined image also reports `quality_metric`
exactly 0.0 with `quality_valid` false.

This is the third case the presence convention exists for, and it was invisible.
It survived only by accident: `_f(...) or np.nan` in the pixel loop maps a
measured 0.0 to NaN, so a declined image happened to read as unreadable. The
same expression would map a *genuine* zero-flux pixel to NaN — latent today,
since no non-declined image contains one, and now removed. `DVDifferenceImage`
carries the uncertainty and names the state, so the distinction is a fact about
the data rather than a side effect.

| state | encoding | rows |
|---|---|---:|
| no DV report at all | stamp absent, `present` 0 | **58.9%** — all Kepler, all K2, 6.8% of TESS |
| DV declined this sector | that slot absent, `present` 0 | 26.6% of images; costs only 3 targets their last sector |
| measured, and flat | stamp present, `present` 1, values 0 | the only one that is evidence |

**Presence is 41.1% of rows (2,232 of 5,426), not the 81.5% the "18.5% have no
difference images" note implies.** That 18.5% (19.96% on the manifest) is the
share of *TESS targets queried* with no DV product. On the view set the branch is
absent for every Kepler and K2 row by construction, and present on **93.0% of
TESS**.

**The presence flag adds no leakage, checked rather than assumed.** It is
label-correlated (TESS: 57.6% positive when present against 23.2% absent), which
would be a real hazard on a stage aimed at W2. But it is a **strict subset** of
`dv_usable` — `dv_usable = 1` implies a stamp exists, with 142 rows the other way
— and `dv_usable` is already a mask column riding into fusion. Alone it is the
*weaker* discriminator: AUC 0.5443 against `dv_usable`'s 0.5649 on TESS. The
branch therefore hands the model no separation it did not already have.

#### The control, fixed before the run

**§4.2 asked for "a post-stage-8 control" and named none.** Five runs on disk
could answer to it. Decided from the record, before any stage-9 number exists:

| candidate | verdict |
|---|---|
| `branches-20260808-rebaseline` | **no.** Stage 6 named it "the control for every stage after it", but that was written before stage 8 and 10.5 existed. It predates the label work and builds its own `StratifiedGroupKFold` |
| `stage8-propensity`, `stage105-propensity` | **no.** Propensity weighting was never adopted — 4.8 still carries "stage 8's qualified second win" as Ollie's open decision, and 3.11e weakened it. Stage 9 runs unweighted, so its control is unweighted |
| `stage8-control` | **no.** Right architecture and `baseline_intervention: None`, but no `fold_assignment`: it builds its own partition at n=5,426, so part of any margin is only which rows fell where |
| **`stage105-control`** | **yes** — the only post-stage-8 run of the unchanged branch architecture carrying the pinned `models/fold_assignments/stage10_5.json` |

**But it is the anchor, not the comparison.** Stage 9 rebuilds the shard set —
new views, and from a labels table that has moved since. Measuring a rebuilt
shard set against `stage105-control`'s old one would confound the branch with the
rebuild, which is the error this section exists to avoid. So:

1. **Arm D** — the rebuilt shard set **with** the difference branch.
2. **Arm C, the paired control** — the *same* shard set with
   `drop_branches: ["difference"]`. The declared-ablation mechanism keeps every
   `Input` in the signature, so the two differ in the branch and in nothing else.
   **The branch's effect is Arm D minus Arm C.**
3. **`stage105-control` is the anchor**: Arm C against it says whether the
   rebuild moved the baseline. A large move there is a finding about the data,
   reported separately, and does not touch the D−C contrast.

**Both arms run on `models/fold_assignments/stage10_5.json` at
`n_models_per_fold: 3`.** The trainer *drops* rows the artefact does not cover,
so this restricts the rebuild to the control's own **5,375** groups on the
identical partition — no extension, no new targets, comparability for free from
the artefact 4.8 already identified as the shared prerequisite.

#### Pre-registered — recorded 2026-08-17, before either arm is launched

**The floor, and which one.** Per **3.11d**, the pairing between two runs' members
is arbitrary and the floor **marginalises over all 3! = 6 pairings**, each by
stage 6's `2 x sd(draws) / sqrt(3)`. The **maximum**-pairing floor is the bar; the
mean is the headline; the minimum is reported and is explicitly not the headline.
Each arm measures its own — a fixed floor is not available, since the recall
floors on disk span **22.1x** (0.00286 to 0.06318 in `pooled_gate_recall_seed_sd`,
`branches-20260809-drop-periodogram-clean` to `stage105-propensity`).

**How each outcome reads. Fixed now, so no number below can be re-read later.**

| # | prediction | what confirms it | what falsifies it |
|---|---|---|---|
| **1** | **W2 — the stage's reason to exist.** The control-arm **host-AUC** falls from Arm C to Arm D by more than the max-pairing floor | a fall clearing `1x` the max-pairing floor | a fall inside the floor, or any *rise*. **This is the falsification of the branch's value**: it attacks host-scoring at source, and if it does not move host-scoring it has not done the one thing it was built for |
| **2** | **Recall.** TESS recall @1% FPR is **not** moved beyond its own max-pairing floor | margin inside `1x` | a margin clearing `1x` either way |
| **3** | The presence flag adds no mission separation: Arm D's TESS-vs-Kepler split does not exceed Arm C's beyond its floor | within floor | Arm D separates missions more |
| **4** | The rebuild is not itself an effect: Arm C's TESS recall sits within its floor of `stage105-control`'s 0.2831 | within floor | outside it — then the rebuild moved the baseline and **1–3 are reported against Arm C only**, with the anchor comparison recorded as failed |

**"Unresolved" is a named outcome, not a fallback.** Per **4.1b**, a margin within
**1.5x** its floor is UNRESOLVED — a stop-and-ask, reported as neither confirmed
nor falsified. Given the floor's own sampling spread is roughly 40% of its value
at three draws, **prediction 1 landing between 1x and 1.5x is the single most
likely outcome**, and it is recorded in advance as *unresolved and needing more
draws*, not as a weak pass. It will not be read as a pass.

**What is not claimed.** The branch is present on 41.1% of rows and on no Kepler
or K2 row at all, so a null on the pooled statistics is uninformative about the
branch and must not be reported as evidence against it. **Every prediction above
is on the TESS slice**, which is where the branch exists.

**Nothing promotes on any of this.** `models/registry.json` is untouched,
`ca906040` stays served, and neither arm is written into `models/cv/` under a
name the weekly gate could select — the gate takes the newest
`models/cv/*/cv_summary.json` that is not the control lane, and the Saturday
09:00 job must not pick up an experimental arm.

#### Built 2026-08-17 — the branch exists, neither arm has run

`f0dccf0`. `preprocess/diffimage.py` re-grids; `difference_view`
`(8, 17, 17, 3)` and `difference_quality_view` `(8, 2)` join `VIEW_SHAPES`; the
branch is a 2-D tower under `TimeDistributed` pooled by attention whose logits
read both the encoded stamp and DV's quality for it.

Three things worth having written down:

- **The pool is masked, and the mask is why.** The number of measured sectors is
  how many times TESS looked at the star, so an unmasked pool would make this
  branch's output scale with observation baseline — the confound the label work
  exists to remove, re-entering through the branch built to attack it.
- **A textbook masked softmax returns NaN here.** Masking with `-inf` gives NaN
  when every slot is absent, which is **58.9% of rows**, and a NaN reaching the
  presence gate multiplies to NaN rather than to nothing. A finite offset plus an
  explicit zeroing keeps every row finite; there is a test for it.
- **Sectors are capped at 8, kept highest-quality first.** The count runs 1–43,
  median 3; eight covers 86.0% of present targets whole. Keeping the *best* eight
  rather than the earliest means a 40-sector target's retained quality is higher
  than an 8-sector target's — recorded as a known cost of the cap, and preferred
  to feeding the branch images DV itself flags as untrustworthy.

**The rebuild is cheaper than costed, for a reason worth keeping.** The two new
views come from the DV report, so they depend on neither the bin resolution nor
the ephemeris the per-target cache is keyed on. They are built at assemble time
and the light-curve cache is untouched, turning "rebuild every folded view from
the FITS files" into "re-parse the DV archive". **Against a warm cache that is
minutes rather than ~95 min.**

**The cache on disk is not warm, and that is pre-existing.** Only 18 of the
current labels table's 5,705 rows hit the interim cache: `t0` or `duration` moved
for nearly every row in the 2026-08-14 labels refresh, and the cache is keyed on
the ephemeris. So the first stage-9 rebuild pays the full light-curve cost
anyway. Unrelated to this branch, and recorded because it makes the next rebuild
after any labels refresh cost 95 min rather than the minutes the key implies.

#### Amendment — recorded 2026-08-20, before either arm was launched

**The shard set is the existing view set plus the two DV views, not a cold
rebuild.** The DV-sourced views depend on neither the bin resolution nor the
ephemeris, so `build_viewset.py --views-from` adds them to
`data/processed` and passes the eleven light-curve views through **byte for
byte** — verified, all eleven arrays equal and the scalars frame identical.

**Why this is a better measurement and not a shortcut.** Arms C and D were always
going to share one shard set, so the D−C contrast was valid either way. What
changes is the *anchor*: rebuilding from the light curves would have folded a
rebuild into the comparison against `stage105-control`, which is the confound the
anchor exists to detect. Now arm C's inputs differ from that run's in the two new
arrays and in nothing else.

**Said against my own interest: prediction 4 is now close to tautological** and
must be read that way. It asked whether the rebuild moved the baseline; there is
no rebuild, so a pass tells us almost nothing and only a *failure* would be
informative — it would mean something non-obvious moved. It stays on the list
rather than being quietly dropped, with its weakened status recorded here.

It also cost **47 s** against the ~95 min the item was costed at, and the
2,232 rows carrying a stamp (41.1%) reproduce the independent measurement above
exactly.

**Stops if.** Unchanged, and neither condition fired.

#### Result — the branch is built and measured, and its own falsification test cannot be run (2026-08-20)

Arm C 2 h 02, arm D 2 h 57, both on `models/fold_assignments/stage10_5.json` at
`n_models_per_fold: 3`, both over the same **5,375** groups (51 of 5,426 dropped
by the artefact, as intended). Identical TIC sets across C, D and the anchor,
confirmed before any metric was read.

| TESS, pooled out-of-fold, n=2,367 (1,300 positive) | arm C — branch dropped | **arm D — with branch** | anchor `stage105-control` |
|---|---:|---:|---:|
| ROC-AUC | 0.9215 | 0.9156 | 0.9250 |
| recall @1% FPR | 0.2315 | 0.2146 | 0.2831 |
| Brier | 0.1120 | 0.1135 | 0.1091 |
| ECE | 0.0416 | 0.0250 | 0.0396 |

*Outcomes, read against the pre-registration and nothing else.*

| # | prediction | outcome |
|---|---|---|
| **1** | control-arm **host-AUC** falls from C to D beyond the max-pairing floor | **UNMEASURABLE** — see below. Neither confirmed nor falsified |
| **2** | TESS recall @1% FPR **not** moved beyond its own max-pairing floor | **confirmed** — −0.0169 at **0.20x** the 0.0843 max floor |
| **3** | arm D does not separate the missions more than arm C | **confirmed** — split +0.0342 → +0.0407, a +0.0066 change at **0.21x** the 0.0305 max floor |
| **4** | arm C's TESS recall within its floor of the anchor's 0.2831 | **confirmed** — −0.0515 at **0.70x** the 0.0740 max floor |

**Prediction 1 cannot be evaluated, and the reason predates this stage by eleven
days.** The stage 7i harness sets **every DV-derived column to NaN and
`dv_usable` to False** for every control-arm row — its *limit 1*, pre-registered
2026-08-09, on the ground that no DV report exists at a synthetic ephemeris. A
control-arm host therefore carries an all-absent `difference_view`, the presence
gate zeroes the branch, and its contribution is **exactly 0.0** — measured on the
built model, not argued. Arms C and D are consequently the *same model* on
control-arm inputs, and any host-AUC difference between them would measure
training history rather than the branch.

**So the stage's own falsification test is unrunnable on the instrument it
named, and that is the headline.** Stage 9 was justified as the direct instrument
against **W2**; predictions 2–4 confirm only that the branch **costs nothing** —
it does not degrade recall, does not worsen mission separation, and the rebuild
did not move the baseline. **Nothing here establishes that it delivers anything.**
Reporting the three confirmations without this sentence would be reporting a
stage as successful on its secondary criteria while its primary one was never
measured.

**What is *not* being done.** The harness will not be modified to feed real DV
stamps so that prediction 1 becomes measurable. Changing a pre-registered limit
*after* discovering it blocks a prediction is re-specification, which this
project does not do. It is a legitimate design question — a stamp taken at the
real TCE, shown beside a light curve with no transit in it, is arguably the
sharpest W2 test available — and it is **Ollie's call**, recorded here rather
than taken.

**The amendment's reasoning was wrong, and prediction 4 was informative after
all.** It was recorded as near-tautological on the ground that no rebuild
happened. But arm C lands **−0.0515** from the anchor on TESS recall despite
byte-identical light-curve views — 0.70x its floor, inside, but most of the way
there. The likely mechanism is RNG: `augment_viewset` draws
`tf.random.normal` per view, so adding one noised view shifts the consumption
order and every later draw — augmentation and dropout alike — differs. Two runs
over shard sets differing only by an added view are therefore **independent
draws, not replicates**. Not isolated, so it is offered as the probable cause
rather than a finding.

**Recorded as a standing limit on every cross-run comparison, not only this
one.** The shared fold artefact was built on 2026-08-14 against exactly this
problem — 3.11a records it as the blocking build, *"reusable — stages 9 and 7ii
face the same cross-run comparability problem."* It fixes **which rows land in
which fold**. It does not touch the RNG stream, and the stream is the half that
moves when the view count changes. Cross-run comparability was therefore only
ever half-closed, and the open half was not visible until a run added a view.
Two consequences, both unclosed at the time of writing: any cross-shard-set
reading in this file is **one draw from an unmeasured distribution**, and the
max-pairing floors may be calibrated only for paired reruns, in which case they
are too tight for the cross-set case they are being read against. The structural
fix is `tf.random.stateless_*` keyed on `(example_id, epoch)`, which would make
such runs genuine replicates; it changes augmentation behaviour and so carries
its own rebaseline, and is **not** taken here.

**That is also why the paired design was the right one.** The drop is applied at
the *model*, not the shard set: every `Input` stays in the signature and the
stream yields all thirteen views to both arms, so C and D consume the identical
augmented batches. Only the C-versus-anchor comparison crosses shard sets, and
only it is affected.

**One post-hoc number, labelled as such and not a test.** TESS ECE improves
0.0416 → 0.0250, but at **0.64x** its own max-pairing floor that is not a
measured difference, and no ECE floor was pre-registered. It is recorded because
it is the only cell where arm D leads, and suppressing it would be as selective
as banking it.

**Nothing promotes.** `models/registry.json` untouched, `ca906040` still served.
Both arms are written to `models/stage9/` — deliberately outside
`models/cv/`, which is the glob the weekly gate draws its candidate from.

**Stops if.** Unchanged, and neither condition fired.


### 4.2b The ExoMiner++ readout — measured against the real repository and the paper (2026-08-20)

The NASA ExoMiner repository (`ExoMiner-main`, ExoMiner++ / the 2025 TESS paper
build) was read directly, together with Valizadegan et al. 2025, *AJ* **170**:287.
Everything below is quoted from that code or that paper, not from a summary of
either. Their code is **not vendored** — NASA NOSA licence, and
`models/cnn_branches.py:5` already records that decision.

**1. ExoMiner does not detect. It vets.** `exominer_pipeline/run_pipeline.py`
takes TIC IDs, downloads light-curve FITS **and DV XML** from MAST, and reads its
TCEs out of SPOC's DV XML. There is no period search anywhere in the repository.
"Validates 301 new exoplanets" is *classification of existing candidates*, not
detection. **This project already has `search/bls.py`, which ExoMiner has no
equivalent of.** Detection-plus-vetting is therefore not a gap to close but a
capability the reference implementation does not have, and it is the honest
differentiator for this work.

**2. The difference-image branch was built without its reference frame, and that
is the mechanism behind stage 9's null.** ExoMiner++ feeds its difference branch
**three** `[33, 33, 5]` tensors plus quality; ours feeds **one** `(8, 17, 17, 3)`.

| input | ExoMiner++ | ours |
|---|---|---|
| difference image | `diff_imgs_std_trainset` | channel 0 |
| out-of-transit image | `oot_imgs_std_trainset` | channel 1 |
| **target pixel position** | `target_imgs`, one-hot at the star's pixel with sub-pixel offsets — `src_preprocessing/diff_img/preprocessing/utils_diff_img.py:183` | **absent** |
| **neighbouring TIC positions + magnitudes** | `neighbors_imgs`, built by `diff_img/search_neighbors/` | **absent** |

We carry the difference and the reference image. What is missing is the
**origin**. `diffimage.py::_centred_slice` centres the *bounding box*, which puts
the star near the middle as a placement artefact rather than a measurement, so a
centroid offset is not computable from what the branch is given. This explains
stage 9 without needing prediction 1, which remains unrunnable for the reason
4.2 records.

**3. Their training set is 27x ours, and the unit of analysis differs.** The
paper reports **147,568 unlabeled TCEs** over TESS SPOC 2-min S1–S67, of which
ExoMiner++ calls **7,330** planet candidates. One ExoMiner row is a **TCE from
SPOC's DV XML**; one of ours is a **catalogue disposition**
(`data/catalog.py:35`), with Kepler additionally hard-capped at 1250/1250 in
`conf/data/full.yaml:15`. Their negatives are pipeline detections that vetting
rejected; ours are false positives somebody adjudicated. That difference is W1
and W2 at source, and no architecture reaches it.

**4. Multisource Kepler+TESS training is a deliberate fix for exactly our
defect.** The paper trains on Kepler *and* TESS jointly "to mitigate the impact
of TESS's noisier and more ambiguous labels". W1 was re-measured on 2026-08-20 as
a **TESS** defect (+0.3874 TESS, +0.1025 Kepler, **−0.1490 K2**), so the
published remedy for our largest measured defect is a training-set composition we
do not use.

**5. Where we already agree, and it is not by accident.** Savitzky–Golay
detrending: the paper records replacing a spline filter with SG for TESS, and
`build_viewset.py:43` has used SG at window 401 / polyorder 2 throughout. A
per-bin variance channel: they added one, we carry it as the third channel.
Focal loss: `exominer_plusplus.yaml` ships `loss: crossentropy` with
`focal_class_balancing: false` — configured and not used, which is the conclusion
our own campaign reached independently.

**6. Two inputs they have and we do not, beyond the difference-image tensors.**
A `momentum_dump` branch — spacecraft reaction-wheel desaturation, a TESS-specific
systematic, and per their Table 1 the only model of eleven that uses it. And,
for TESS only, they **removed** the flux-weighted centroid statistics and the
difference-image centroid offset because both are *"not reliable for TESS ...
due to substantial crowding in the photometric apertures"*, replacing them with
target magnitude and RUWE. Our `centroid_view` is scoped to `mean_sky_offset`,
`control_sky_offset` and `ruwe` (`cnn_branches.py`), so we are feeding on TESS
two scalars the reference implementation deliberately dropped for TESS. **Not
acted on here; recorded as a finding with a citation.**

**7. Their public vetting catalog is a spreadsheet.**
`exominer_vetting_pc_catalog_dash-render-web-app/` is **270 lines of Dash**: one
`dash_table.DataTable`, a regex filter, a CSV export button and a logo. No plots,
no light curves, no difference images, no explainability. Our `/score/{tic_id}`
already returns phase-folded global/local/odd/even views, a centroid track, a
periodogram, five diagnostic suites, and a calibrated probability with its
MC-dropout band and per-fold members. **The product gap runs in our favour and
it is large.**
