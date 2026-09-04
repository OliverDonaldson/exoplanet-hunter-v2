> Moved verbatim from `docs/roadmap.md` §4.2d on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

### 4.2d Phase 1 — the target-position channel and the momentum dump

#### Built 2026-08-27 — both inputs exist, neither arm has run

**The origin was measurable all along, and nobody had looked.** 4.2b finding 2
said the branch has "the difference and the reference image and NO origin", and
named `target_imgs` as ExoMiner++'s third tensor. The origin is in the DV report
we already parse: every `differenceImageResults` block carries a
`ticReferenceCentroid` with `row` and `column` in the **same CCD frame as the
pixel list**, at sub-pixel precision. It was simply never read.

**Measured over the whole archive before anything was built**, because a channel
pinned to the placement it is meant to replace would make the whole phase
vacuous. All 53,118 difference images, of which 14,154 are the declined state
3.9-era work already found:

| | |
|---|---:|
| non-declined images carrying a usable target position | **38,964 of 38,964 (100%)** |
| target offset from the bounding-box centre, row | mean −0.066 px, **sd 0.607** |
| target offset from the bounding-box centre, column | mean −0.028 px, **sd 0.637** |
| median radial offset | **0.84 px** |
| stamps whose target is in a **different pixel** than the box centre | **77.8%** |
| stamps more than 1 px from it | 29.6% |
| sub-pixel residual after rounding to a pixel | sd **0.2884** (uniform is 0.2887) |
| largest offset seen | 3.24 px, on a grid 17 wide |

So `_centred_slice` was not an approximation of the target position: it disagrees
with it on more than three quarters of the archive, and the residual after
rounding is *uniform*, meaning a hard one-hot would throw away a full pixel of
position on every stamp.

**`difference_view` is now `(8, 17, 17, 4)`, and `present` stays last.**
Channels are `[difference, out-of-transit, target, present]`. The marker is
inserted *before* the presence flag rather than appended, because
`cnn_branches._gated`, `SectorPresence` and `viewset_augment._augment_view` all
read presence as `[..., -1]` on every view in the project. Appending would have
silently made three modules gate a branch on a position marker.

**The marker is a bilinear deposit, not ExoMiner's one-hot, and the reason is
our grid.** ExoMiner writes a hard 1 at the rounded target pixel and recovers
the sub-pixel part by up-sampling the stamp first. `DIFF_GRID` is one CCD pixel
per cell by construction — the same refusal to resample that `_periodogram_view`
and `_centroid_view` make — so up-sampling would undo the property the module
rests on. Splitting one unit of mass over the four pixels the position falls
between carries the same information at native scale: at a pixel centre it
reduces to their one-hot, and the marker's centroid reproduces DV's own
sub-pixel value exactly. Mass falling off the grid is **dropped, not
renormalised** — a marker summing to less than one says part of the star is out
of frame, which is true, where rescaling would move the implied centroid inwards.

**A stamp with no origin is now absent rather than centred.** `regrid_stamp`
returns None when DV left the position undefined or placed it outside the frame.
On this archive that is 0 stamps, so no row changes state; it is there so that a
future product cannot quietly reintroduce the exact defect this phase exists to
remove. DV's sentinel for the undefined case is **on the uncertainty, not the
value** — a declined sector is written as row 0.0, column 0.0, uncertainty −1.0,
on every one of the 14,154 — so reading the value alone would place the target
2,000 px away with confidence. The same trap `_nan` was written for.

#### The momentum dump — the flag is not in our light curves, and could not be

**Bit 32 is zero on every cadence of all 6,192 cached TESS curves.** They were
downloaded through lightkurve's default quality bitmask, which *removes*
desaturation cadences before the file is written. `_gap_view` already recorded
this in passing — "reading `QUALITY` bit 5 directly gives zero for every target
in the cache" — and built the gap-shaped proxy instead. A branch fed the cached
flag would have been an all-zero input that looked like a feature.

**So the flag is fetched from the spacecraft rather than from the target.** A
momentum dump is an attitude event: the wheels spin down and every target on the
focal plane is flagged in the same cadences. Measured 2026-08-27 on sector 1,
four independent TICs carry **the same 70 flagged timestamps** to 1e-6 d — not a
similar count, the identical list. `scripts/fetch_momentum_dumps.py` therefore
downloads **one** unmasked 120-s curve per sector and writes
`data/tables/momentum_dumps.parquet` as `[sector, time]`, and re-checks the
common-mode property against a second target every tenth sector.

**The cadences the dump removed are put back at the target's own cadence.** The
dump cadences are missing from each target's own time array for the same reason
they are missing everywhere, so a fold over the surviving times finds nothing at
any phase. Each dump is re-expanded over its measured interval at the target's
median cadence, so a 120-s target and a 200-s FFI target get the number of
cadences each would actually have lost rather than the number the representative
curve lost. `momentum_dump_view` is `(201, 2)` = `[dump fraction, present]` over
the same local window as every other local view, where the fraction's denominator
is every cadence the target either has or provably lost.

**No variance channel, against ExoMiner.** Their
`local_momentum_dump_view_var` is the within-bin spread of a 0/1 flag, which for
a Bernoulli mean `p` is `p(1-p)` — a deterministic function of the channel
already there. It would be a second copy of the first, and this project has
spent two stages removing inputs that looked like measurements and were not.

**TESS-only by construction**, like `difference_view`: Kepler and K2 never saw a
TESS reaction wheel, so those 3,027 rows carry presence 0 rather than a measured
zero, and no Kepler FITS is opened to produce one.

#### The epoch had to be recovered, and that is a limit worth stating

The momentum view is folded on the transit ephemeris, and
`viewset_scalars.parquet` records a row's `period` and `duration` but **not its
`t0`**. Rebuilding the light-curve views to recover it is exactly what the 4.2
amendment forbids — it would fold a labels refresh into the comparison, since
the epochs have moved twice since this set was built.

So the epoch is matched back out of the labels tables on an exact `(tic_id,
mission, period, duration)` agreement. **Why that identifies it:** across the
5,478 rows present in both `labels.parquet` and `labels.previous.parquet` with
identical period and duration, `t0` differs on **0**. Period, epoch and duration
move together in a refresh, so a row matching on two of them did not move.
Recovered for **2,380 of 2,399** TESS rows; the other 19 get an absent momentum
view rather than a fold on a guessed epoch.

#### Three build decisions recorded, because each could have gone quietly wrong

- **The target marker is not augmented.** `viewset_augment` protects the last
  channel of every view because noise on `present` flips a gate; the marker is
  now protected the same way, via `_ANNOTATION_CHANNELS`. Noising the star's
  catalogue position would invent a positional uncertainty DV never reported, on
  the one channel the centroid measurement is taken *against*. It also leaves
  the noise draw's shape on a stamp unchanged at `(sectors, 17, 17, 2)`, which
  is what it was before the marker existed.
- **`momentum_dump_view` joins the DV pair in `ASSEMBLED_VIEWS`**, not in the
  per-target light-curve cache. Its source is a side table that can be refetched
  without touching a FITS file. Unlike the DV pair it *is* folded on the
  ephemeris, so the assemble step rebuilds it every time rather than reading it
  back from a cache keyed on one.
- **The control arm is given the dump table and is still denied the DV
  columns.** Stage 7i's limit 1 masks DV because *no DV report exists at a
  synthetic ephemeris*. A desaturation is not a DV product: it happened to the
  spacecraft at a real time, and folding it on a synthetic period is as
  meaningful as folding the star's flux on one. Withholding it would gate the
  new branch off and measure a model the run did not build. **This does not
  make stage 9's prediction 1 measurable** — arms C and D still differ only in
  the difference branch, which is still all-absent on control-arm rows, so they
  are still the same model there.

**What the paired drop is, unchanged from stage 9.** Arm C drops `difference`
and arm D drops nothing, on one shard set. The momentum branch is present in
**both**, so it cancels out of the D−C contrast and the mechanism test is
unconfounded by it. Its own attribution is a stage 7ii question and is not
commissioned here.

#### Phase 1a had to run on 9b57f79's code, and finding out cost one launch

**The seed sweep will not run on today's tree, and that is correct rather than a
defect.** The first launch failed in seconds: `Missing data for input
"momentum_dump_view"`. The model builds an `Input` for every entry in
`VIEW_SHAPES`, so the Phase 1 build makes it a fourteen-view model, and the
stage-9 shard set has thirteen. The stamp's channel count moved too, 3 to 4.

**Had it not failed it would have been the wrong measurement anyway.** Phase 1a
exists to give the seed spread of *the thing stage 9 measured*, so that its
floors are commensurable with stage 9's −0.0515 anchor gap and with the arm C/D
contrast. A run on the new inputs is a draw of a different configuration. So
seeds 43–45 run from a detached **9b57f79** worktree against
`data/processed/viewset_tfrecords_stage9` — same code, same shards, same fold
artefact, same config, seed varied and nothing else. Arm D is seed 42 of exactly
that, which is why only three runs are launched for four draws.

**One provenance wrinkle, recorded rather than left to be noticed.**
`run_config.git_sha` is read from the working tree, so the seed-sweep summaries
name the dirty Phase 1 tree while the code that ran was 9b57f79. Each run
directory carries a `CODE_PROVENANCE.txt` saying so. The arms have no such
problem — they are the working tree, which is the point of them.

#### The floor arithmetic was verified against stage 9 before it was used

`.phase1-scratch/analyse_phase1.py` recomputes the max-pairing floor from each
run's own `predictions.parquet` member columns rather than reading the stored
variance block, so the dv_usable slice can have one too. Run against stage 9's
arms it reproduces the published figures exactly — prediction 2's **0.0843** and
prediction 4's **0.0740**, and the −0.0169 margin at **0.20x** — which is what
licenses using it on Phase 1's.

It also produces the number stage 9 never reported: **stage 9's own arms on the
`dv_usable` slice**, which is where 4.2c's pre-registered statistic lives.

| stage 9, TESS ∩ `dv_usable`, n=2,077 (1,220 positive) | arm C | arm D | D − C | max floor | x |
|---|---:|---:|---:|---:|---:|
| ROC-AUC | 0.9255 | 0.9181 | −0.0075 | 0.0227 | 0.33x |
| recall @1% FPR | 0.2459 | 0.2164 | −0.0295 | 0.0507 | 0.58x |

Both inside their floor, as on the full TESS slice. **This is the comparison
Phase 1's arms have to be read against**, and recording it before Phase 1's
numbers exist is deliberate.



#### Two version-skew failures, and the rule they establish (2026-08-27/28)

Adding a view broke two things that had nothing to do with the branch, and both
failed in the same way for the same reason. Recorded because the rule generalises
to every stage that adds an input.

1. **Phase 1a would not run on the new tree.** `build_cnn_branches` builds an
   `Input` for every entry in `VIEW_SHAPES`, so the Phase 1 build makes it a
   fourteen-view model and the stage-9 shard set has thirteen. It failed in
   seconds on `Missing data for input "momentum_dump_view"`. Had it not failed it
   would have been the wrong measurement anyway — a draw of a different
   configuration, not a replicate of arm D.
2. **The control-arm harness fed a 4-channel stamp to a 3-channel model**, dying
   after **95 minutes** of host building on `expected shape (None, 8, 17, 17, 3),
   found (64, 8, 17, 17, 4)`. The harness *rebuilds* views from the light curves
   and scores them through a run's saved Keras members, so it inherits the
   working tree while the members are frozen at the code that trained them.

**The rule: any harness that rebuilds inputs must be pinned to the code version
of the model it is scoring, not to the working tree.** Both Phase 1a lanes now
run from a detached `9b57f79` worktree; the Phase 1 arms and their control arms
run from the working tree, which is the point of them.

**What was *not* affected, checked rather than assumed.** An import-graph trace
over `training/train.py`, `build_dataset.py`, `shard_views.py`, `control_lane.py`,
`refresh_pipeline.py`, `api/app/main.py` and `scoring/service.py` reaches **none**
of the eight changed modules. The `views_io.py` / `viewset_io.py` separation held
exactly as its docstring claims, so the weekly refresh and the live `/score` path
are untouched by this stage.

#### The weekly refresh had never reached a verdict — fixed 2026-08-28

`outputs/refresh-cron.log` records **two flow runs and zero verdicts** — no
`PROMOTE`, `REJECT` or `UNRESOLVED` string appears anywhere in it — and the
latest died in `validation_gates()` at `validate_data.py --strict` on
**`dv-archive FAIL: 12 expected targets absent`**.

*(Corrected 2026-09-04: this first read "15 runs", from a `grep -c refresh` that
counted every line mentioning the word, task names like `refresh_label_catalogue`
included. `grep -c "Beginning flow run"` gives two. The zero-verdicts claim and
the diagnosis are unaffected.)*

**The gate was right and the data was wrong.** A labels refresh added 12 TESS
targets — 8 `KP`, 4 `FP` — that had never been queried against MAST for a DV
report, and `check_dv_archive` exists precisely to keep "never queried" separate
from "no DV product", because the two are indistinguishable in a presence mask.
`fetch_dv.py` on those twelve took **49 s**; all 7 gates now pass at exit 0.

So **4.1a's "the weekly gate is the one automated decision in the project"** has
been true in design and not in fact for at least a week. Worth stating plainly:
the lane, the floor and the third verdict were all built and none of them had
ever run to completion on the schedule they were built for.

#### The promotion log — built 2026-08-28, and it was a leak rather than a feature

`models/registry.json` records only what is *currently* served, so `/runs`
hardcoded `verdict=None, reason=None` and the console printed "No promotion log
is written yet" on every row. But the verdict was already being computed and
then deleted: `PromotionDecision` carries a three-state verdict, reasons and
alarms; `write_decision` already serialised it; `promotion_gate.py --verdict-out`
already exposed it; and `refresh_pipeline.py` already passed it — **into a
`tempfile.TemporaryDirectory()`**.

The gate now writes `promotion_log.json` into the candidate's own run directory
and `/runs` serves it. `write_decision` was **wrapped rather than extended**, so
`read_decision` stays a faithful inverse of both files. The log additionally
carries the thresholds *as applied*, including `floor_source` — which
distinguishes a measured floor from `LEGACY_RECALL_TOLERANCE` standing in for
one, a distinction the bare number cannot carry.

**Verified on a live gate run, not just in tests.** Gating `models/stage9/arm-d-difference`
returned REJECT and wrote a log whose verdict `/runs` reads back; the registry
was byte-identical before and after. That run also surfaced something Phase 1
will hit: the gate refused a pooled comparison because **`ca906040`'s own summary
carries neither a `per_mission` block nor a variance block**, so gating the Phase 1
arms requires `--champion-summary models/cv/champion-rebaselined-today/cv_summary.json`
or the control lane's. That is what `--champion-summary`'s help text already
anticipates.

**One console defect found and deliberately not fixed here** (`frontend/` is
another session's): `app.pages.js:860` is a two-way ternary, so **4.1b's
`UNRESOLVED` — a named third outcome and an explicit stop-and-ask — would render
as a red reject chip.** The recommended patch is in `docs/console-verdict-chip.md`.
