> Moved verbatim from `docs/roadmap.md` §3.1 on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

### 3.1 Stages 1–3 — housekeeping, ExoMiner-grade inputs, the re-baselined summary

**Stage 1 *(old 0)* — housekeeping, landmines, vendoring.** *(done)*
71 GB of stitched-and-forgotten staging deleted and auto-cleanup added;
`preprocess_only.py` and `score_target.py` deleted (both could silently write
9-dim/no-K2 data or break on a non-9-dim model); the four remaining audit items
fixed (gate cwd, `dvc` resolution, MLflow run naming, the CI gate jobs
`ci.yml` had promised); patched TRICERATOPS vendored.

**Stage 2 *(old 1)* — ExoMiner-grade inputs.** *(done 2026-08-05)*
A 5,423-example view set with eleven branches — global/local flux with variance
and presence channels, odd/even, weak-secondary, centroid, flux-trend, unfolded
per-transit with transit counts, cadence-gap, and the periodogram pair — plus a
3.6 GB DV archive, its scalars table, Gaia RUWE, FFI recovery for the 744
`no_fits` candidates, and a seventh validation gate. `ca906040` served
untouched throughout.

The three things flagged as likely to bite, and what actually happened:

- **Shard size.** Feared 20-50x. Actual **2.6x** at 301/31 (122 MB against
  47 MB), so the `tf.data.cache()` decision needed no revisiting *at that
  resolution*. Restoring 2001/201 moves it — see the run-2 sizing below.
- **Per-branch presence masking.** Necessary exactly as predicted: `dv_usable`
  is 87.4% on TESS and **0% on Kepler and K2**. Every branch carries a presence
  channel and the model gates on it.
- **The DV download.** Sized at 14-56 GB and many hours; actual **3.6 GB** in
  5.3 h. The 2-8 MB/file estimate was the DVR *PDF* and DVT *FITS*, not the
  ~0.34 MB XML. What mattered for runtime was batching the availability query
  (40 TICs per round trip), not scoping sectors.

Two things could not be built as specified. The **momentum-dump branch** reads
`QUALITY` bit 5, which lightkurve's default bitmask strips at download — the
flag is zero on every cadence in the cache — so it measures the hole a dump
leaves instead. And **difference-image stamps are 11-17 px, not a fixed
33x33**; that is Kepler's size, and stage 9 must re-grid.

Full detail, and the merge collision that silently dropped the transit counts
past all seven gates, in `data_provenance.md`.

**Stage 3 *(old A)* — the re-baselined incumbent summary.** *(done 2026-08-08)*
`evaluate.py summarise` writes `models/cv/incumbent-rebaselined/cv_summary.json`
with a `per_mission` block computed from out-of-fold rows only, so the gate
returns a decision instead of falling through to pooled means. Detail in "The
gate cannot engage against the incumbent" above, including the trap in the fix.

**Stage 4 *(old 2(a))* — the model, incrementally.** *(closed — all four arms
rejected)*
Originally specified as one stage with four gated sub-steps: (a) per-diagnostic
branches + scoped scalars + variance channels + joint local conv; (b)
unfolded-flux branch; (c) trend + periodogram branches; (d) difference-image
branch with quality attention, then an Optuna re-tune on the winner. **The
sub-step design was never the implementation path** — `build_cnn_branches` builds
all eleven branches at once — so what survives of it is: (a) is stage 4 and is
closed, (b) and (c) become attribution and are stage 7, (d) is still a genuine
build and is stage 9, and the re-tune is stage 10. Every arm passes the promotion
gate on CV AUC/Brier/ECE, the TESS slice, and injection-recovery completeness.
