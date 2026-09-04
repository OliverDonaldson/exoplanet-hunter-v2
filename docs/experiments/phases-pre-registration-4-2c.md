> Moved verbatim from `docs/roadmap.md` §4.2c on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

### 4.2c Pre-registered — the four phases, and what each is allowed to claim (pre-registered 2026-08-20)

Recorded before any of it is built. Phase ordering is by *what unblocks what*,
not by appetite.

**Phase 0 — the console, brought forward · ~8 h · no compute.** Stage 12 was
"locked last"; it moves first. It is the only work in this file that is
**entirely decoupled from every open model question** — it reads a pinned API
contract that already exists, so nothing it does can be invalidated by stages 9b,
10, or the data work. It makes the project demonstrable at every later point
rather than only at the end.
*Claims allowed*: none about model quality. Phase 0 is presentation of numbers
measured elsewhere, and any screenshot of it must carry `model_version`.
*Stops if*: it starts requiring API changes. The contract in `api/app/schemas.py`
and `frontend/src/api/types.ts` is pinned and changes to it are a different piece
of work.

**Phase 1 — the target-position channel and momentum dump · ~15 h · 3–4 h
compute.** `difference_view` gains a fourth channel carrying DV's target pixel
position; a `momentum_dump` branch is added. Re-run as the same paired
model-level drop stage 9 used, on the same fold artefact.
*Pre-registered reading*: this is a **mechanism test of finding 2 above**, not a
performance bid. If TESS recall @1% FPR on `dv_usable` rows does not move beyond
its floor, the reference-frame explanation is **falsified** and is to be recorded
as such — the branch is then simply not carrying signal, and no third stamp
variant is commissioned.
*Floors*: from the Phase 1a seed sweep below, not from stage 6's 2026-08-09
figure, which predates stage 8's labels.

**Phase 1a — the seed sweep · ~6–9 h compute.** Three seeds, current winning
config, pinned `stage10_5.json` folds, nothing else varied. Delivers a current
seed sd for recall @1% FPR, TESS AUC **and host-AUC**, the last of which has never
had a floor. It also settles the RNG limit recorded in 4.2: if the spread
contains stage 9's −0.0515 anchor gap, that explanation stands.
*Runs before Phase 1 is read, and before stage 10 is started.*

**Phase 2 — the data fix · ~40 h, mostly download and preprocessing.** Tier 1:
lift the Kepler cap to the full DR25 KOI set — the certified-FP path already
exists at `data/catalog.py:287`. Tier 2: move the unit of analysis to the **TCE**,
ingested from DV XML as ExoMiner does.
*Pre-registered reading*: Tier 2 is the only intervention in this file aimed at
W1/W2 **at source**. If Spearman(baseline, label) on TESS does not fall below
+0.30 after it, the selection-effect account of W1 is wrong and the defect is
something else.
*Limit*: this changes the distribution, so every metric measured before it is
re-based, and stage 3's re-baselined summary must be regenerated.

**Phase 3 — stage 10, then 7ii, then stage 11.** Unchanged in content, and all
three read better after Phases 1–2. Stage 10 explicitly **does not start** before
Phase 1a delivers its floors.

**What none of this promotes.** `models/registry.json` stays untouched, and
`ca906040` stays served, until a stage asks for a promotion in writing and it is
granted.
