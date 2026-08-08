# Demo script — the serving stack as it stands, 2026-08-08

Recorded against **`ca906040`**, the promoted incumbent, unchanged since
2026-07-19. Nothing from the stage 4 branch line is in this demo, deliberately:
it has not passed the gate, and this is the system that actually serves.

Re-record this after stage 11 (serving parity + explainability), and show the
two side by side. The second recording gets per-branch contributions, which is
the thing this one visibly lacks.

## Starting it

Two terminals, or two background jobs:

```bash
cd /Users/ollie/Project/v2 && ./scripts-dev/run-api.sh
```

```bash
cd /Users/ollie/Project/v2/frontend && npm run dev
```

API on `:8000`, console on `:5173`. The API prints `loaded 5 folds from run
ca906040cdb74ba6b07353a500244777` when it is ready — that line is worth having
on screen, since it is the proof that what follows is the served model.

First scoring call on a target takes ~30–60 s (light-curve read, detrend, fold,
20 MC-dropout passes). Score one target before recording so the caches are warm,
then record on a second.

## The run, in five beats

### 1. The catalogue — "11,288 candidates"

Open `http://localhost:5173`. Point at the header: **`API status: ok — serving
cnn_dualview-cv-ca906040`**. The console names the model it is talking to.

Sort, filter by disposition, and note the CSV export. This is the triage surface:
an astronomer's job is to decide which of 11,288 objects deserves telescope time.

### 2. Score one live — TOI-1469.01

Search `TOI-1469.01`, click the row. TIC 283722336, a **Known Planet** at
P = 3.093 d.

The point to make out loud: **this is not a lookup.** The API reads the light
curve, detrends it, folds it on the ephemeris, builds the views and runs the
ensemble. The number appears because the model computed it just now.

### 3. The probability, with an error bar

```
prob_calibrated  0.979        the calibrated probability
prob_mean        0.942        raw ensemble mean, before calibration
prob_std         0.207        MC-dropout spread over 20 samples
decision_threshold 0.486      swept, not assumed to be 0.5
per_fold         5 values     one per CV fold
```

Two things worth saying here. The threshold is **0.486**, not 0.5 — it was swept
on validation rather than assumed. And the score is **calibrated**, so 0.98
means about 98 of 100 such candidates are real. ExoMiner's own tooltip concedes
theirs "is NOT a probability"; this one is, and the reliability diagram at the
bottom of the panel is the evidence — **ECE 0.013, Brier 0.079 over 4,818
held-out predictions**.

### 4. The diagnostics — and the model being contradicted

**This is the beat that matters.** The panel shows five independent,
deterministic checks next to the CNN's opinion:

| check | value | reading |
|---|---|---|
| centroid shift | 0.85σ vs 3σ BEB threshold | the dip is on the target star, not a neighbour |
| odd/even depth | 226 / 231 ppm (Δ 0.4σ) | alternating depths would mean an eclipsing binary; these agree |
| odd/even timing | 0.2 / 0.3 min (Δ 0.1σ) | consistent |
| secondary eclipse | 15 ppm @ φ=0.87 (4.5σ vs 2.5σ FA) | present but albedo 8.2 is unphysical, so not an occultation |
| **duration check** | **q=0.012 · q/q_circ=0.41 · a/R\*=11.0** | **fails** |

And the verdict the console prints:

> **Caution** — transit duration is unphysical for this orbit (q=0.0118,
> q/q_circ=0.41, a/R\*=11.0). Calibrated probability 0.98 should be discounted
> accordingly.

**The system tells you not to trust its own model.** The CNN says 0.98; an
independent geometric check says the transit is far too short for a circular
orbit at that period, and the interface says so in plain language rather than
burying it. That is the difference between a score and a vetting tool, and it is
the single best thing to have on camera.

### 5. Blind search — "and it can find the period itself"

Click **Run BLS periodogram (~30 s)**. It ignores the catalogue ephemeris and
searches from scratch.

Result: **best period 3.095 d** against the catalogue's **3.093 d**. An
independent recovery of the ephemeris from the photometry alone, to 0.002 days.

Worth one sentence on scope: detection at survey scale is deferred and the
reasons are in the roadmap — SPOC already detects, and the measured weaknesses
are all on the vetting side. But the machinery is here and works per target.

## Numbers to have on the slide

Captured from this run — the payload is in `results/` if you keep it, or
regenerate with the URL below.

```
GET /score/283722336?period_days=3.0929386&t0_btjd=3606.251960999798&duration_hours=0.8776501692
```

| | |
|---|---|
| model | `cnn_dualview-cv-ca906040`, 20 MC samples, 5 folds |
| calibration | ECE 0.013, Brier 0.079 on 4,818 held-out predictions |
| gate slice (TESS, n=2,367) | ROC-AUC **0.9100**, recall @1% FPR **0.307** |
| all missions (n=5,375) | ROC-AUC **0.9523** |
| catalogue | 11,288 candidates served; 5,705 labelled rows train the model |
| deployment | 2 GB Fly machine, peak 548 MB, ~$1–4/month scale-to-zero |

## What to say about what is *missing*

Do not oversell. Two honest gaps, both of which make the story better rather
than worse:

- **There are no per-branch contributions yet.** The panel shows deterministic
  diagnostics beside the score, but it cannot yet say *which part of the
  evidence moved the model*. That is stage 11, and it is what the second
  recording will add.
- **The branch architecture has been rejected four times.** This demo runs the
  incumbent because nothing has earned promotion. Being able to say "four
  candidate models were rejected, on a criterion fixed in advance" is a stronger
  claim than a marginal AUC win would have been.
