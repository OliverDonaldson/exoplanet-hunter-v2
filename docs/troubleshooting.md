# Troubleshooting

Failure modes, what they mean, and the operational traps that have cost this
project real time.

## 1. Common failures

**A validation gate FAILs.** Not a bug. That is the system refusing bad data
before compute is spent on it. Read the gate's message, fix the data problem,
re-run.

**`command not found: dvc` / `prefect` / similar.** Wrong conda environment.
See [getting-started.md](getting-started.md) §1.

**"file may be corrupt / interrupted download".** Self-healing — the downloader
evicts the bad file and retries. Re-run if it happened mid-build; nothing is
poisoned.

**Promotion REJECTED.** Not a failure. The challenger lost and the champion
keeps serving. The run remains in MLflow for analysis.

**Console shows "API unreachable".** The API is scale-to-zero; the first request
wakes it. If it persists, check `fly logs` for the boot-time `dvc pull`.

## 2. Long-running jobs

These are specific to this machine and have each cost hours.

**Nothing long-running goes in `/private/tmp`.** Use a scratch directory outside
the repo.

**`setsid` does not exist on macOS.** To detach a job:

```bash
screen -dmS <name> caffeinate -dimsu <script>
```

**Do not hold a long shell waiter against a detached job.** Poll a progress file
that records per-run exit codes, and watch for the screen session disappearing.
A grep that only matches success is silent on a crash.

**`caffeinate`'s `PreventSystemSleep` only works on AC power.** Check it reads 1
*after* launching:

```bash
pmset -g assertions | grep PreventSystemSleep
```

**Never edit `pipeline/src/**` while a run is in flight.** Each sequential run in
a script launches a fresh interpreter and will pick up half-finished edits.
Documentation and out-of-repo scratch are safe to edit.

**Run the test suite one process per file.** Repeated heavy fixtures in a single
process slow without bound.

**Size architectures separately.** The two model families differ by more than 2x
per run on identical settings. Measured: 4 h 44 for a 5-fold, 3-member dual-view
run against roughly 2 h per branch arm. An estimate derived from one does not
transfer to the other.

## 3. Data and storage

### 3.1 Safe to delete

`data/raw/` (every mission cache, re-downloads on demand), `mlruns/`, `outputs/`,
`results/`, and
anything `dvc pull` can restore. Build caches such as `.mypy_cache` are always
safe.

### 3.2 Never delete

`models/registry.json`, the `.dvc` pointer files, and `.dvc/config.local` —
which holds the R2 credentials, exists only on this machine, and cannot be
regenerated without minting a new token.

### 3.3 Reclaiming R2 space

A one-off `dvc gc -c --all-commits` deletes every object whose pointer was never
committed on any branch — trial debris by construction — while keeping every
dataset and model version that git history can still check out.

**Fetch all branches first.** `--all-commits` reads *local* git history, so a
pointer that exists only on an unfetched remote branch looks like debris and its
objects are deleted from the bucket.

```bash
git fetch --all
dvc gc -c --all-commits
```

The stricter `-w` (workspace-only) variant also prunes superseded dataset
versions and breaks `git checkout <old> && dvc pull`. Use it only deliberately,
from a clean, fully-pulled checkout.

### 3.4 A DVC pointer move is a data change

Bumping a `.dvc` file changes the data of record. It gets its own commit,
whatever else is in flight. Two lines folded into an unrelated commit is how the
data of record moves without anyone deciding that it should — this has happened
once and is recorded in [roadmap.md](roadmap.md) §5.4.

Note that the shard sets are built from the catalogue at a point in time. After
a catalogue refresh, rebuilt shards will not reproduce earlier runs, and any
result derived from them is a different measurement.

## 4. Reproducing a recorded result

Runs are identified by their directory under `models/cv/`. Every run's
`cv_summary.json` carries `run_config`, including `git_sha`, `git_dirty` and the
`fold_assignment` file if the outer split was pinned.

Two runs are comparable only if they were scored on the same population. A run
that constructed its own split over a different shard set is not comparable to
one that did, however similar the settings look — check `fold_assignment` before
comparing anything.

## 5. Docker

Not needed for local work. The Dockerfiles exist for the deployed API and a
future GPU training image.
