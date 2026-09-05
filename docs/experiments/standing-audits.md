> Moved verbatim from `docs/roadmap.md` §5 on 2026-09-04. Frozen: a correction is a dated note appended under the entry it corrects, never an edit.

## 5. Standing audits

Done, acted on, and kept here so they are not re-run from scratch. Merged from
`plan-2026-08-09.md`; the dates are when each was performed.

### 5.1 ExoMiner re-audit — not warranted, and the test applied

The 2026-08-07 comparison established 10 ranked adoptions, six questions
answered against their source, what this project already does better, and an
explicit "do not copy" list. A re-read would re-derive it, so it is not repeated
here; the outstanding delta is enumerated below.

**What is actually outstanding is the delta, and it is already enumerated:** of
the 10 adoptions, **6 are done** (shared conv tower, paired Wilcoxon, Cohen's *d*,
N-models-per-fold with the presence gate, serialisable registered layers, code
version pinned in the model config) and **4 are open** — declarative normalisation
policy as an artefact, per-example uncertainty published, provenance headers in
every CSV, versioned container with a DOI. All four are **finishing touches**.

**The one thing worth a fresh look, and only when publishing becomes a goal:**
their published TESS catalogue. That is the single genuine gap against them
(distribution — a survey-scale catalogue with per-row uncertainty, a DOI and a
citation ask), it is a publishing task rather than an engineering one, and it is
deliberately not on this plan.

### 5.2 Security audit — done and acted on, 2026-08-09

| check | result |
|---|---|
| hardcoded secrets, keys, tokens across `.py/.yaml/.toml/.json/.ts/.tsx` | **clean** — only `js-tokens` false positives in a lockfile |
| private keys, AWS-style credentials | **none** |
| CORS | **correctly scoped** — explicit origin allowlist, `allow_methods=["GET"]`, no wildcard |
| input bounds on `/score` | **already hardened** — TIC range, period/duration/epoch ceilings, server paths redacted from client errors |
| auth on a public read-only scoring API | absent by design; defensible |
| rate limiting | **was absent — now added**, `app/ratelimit.py`, 12 tests |
| Python dependency CVEs | **30 across 8 packages** — triaged below, **not bulk-upgraded** |
| npm dependency CVEs | **4 (1 moderate, 3 high)** — **deliberately not fixed tonight**, see below |

**W12's severity was overstated when this plan was first written, and the
correction matters.** `/score` already carries `_score_lock` (one score at a
time, so concurrent callers queue rather than thrash the single serving CPU) and
a 128-entry process-lifetime response cache (a repeated TIC is free). What
neither bounds is a caller walking *distinct* TIC IDs — every one is a cache miss
and a fresh MAST download, serialised into a slow drain of wall clock and egress.
So the real exposure is **cost and availability, not a crash**, and the limiter
is the third mitigation rather than the first.

**Python CVEs — triaged by reachability rather than counted.** Bulk-upgrading
this environment is the wrong move: TF 2.17.1 / Keras 3.15.0 on Metal is a
working stack and the non-negotiable about environment integrity exists because
it has been broken before.

| package | advisories | reachable from the served image? |
|---|---:|---|
| `gitpython` | **14** | **no** — MLflow-side; `dvc` in this install does not require it, and it is not in `docker/constraints.txt` |
| `protobuf` 4.25.9 | 1 | **yes — and the fix is blocked.** See below |
| `aiohttp`, `cryptography`, `h2`, `pyasn1`, `setuptools`, `diskcache` | 15 | unpinned in the serving constraints; transitive, low reachability from a read-only scoring path |

> **The one finding worth escalating: `protobuf` cannot be fixed as advised.**
> PYSEC-2026-1805 lists fixes at **5.29.6 / 6.33.5**, and TensorFlow 2.17.1
> requires **`protobuf <5.0.0dev`**. The advisory's remedy is therefore
> uninstallable without moving TensorFlow, which moves the whole training stack.
> Recorded as an **accepted, documented risk pending a TF upgrade** — not
> silently skipped, and not forced.

**npm — not fixed tonight, on purpose.** `npm audit fix` rewrites
`package-lock.json`, and that file currently carries **another session's
uncommitted `animejs` addition**. Running it would either entangle that work in a
security commit or leave a tangled tree — the same class of mistake as the
`git add docs/` trap, in a new place. The advisories are build-time only
(`postcss` and friends; the deployed console is static), so the cost of waiting
is near zero. **One command, Ollie's call, once the frontend work is committed.**

### 5.3 Cleaning audit — done. The repo is clean; the disk is not

| what | size | verdict |
|---|---:|---|
| `data/` | **74 GB** | mostly the 25 GB FITS cache and derived sets. The FITS cache is the harness's compute saving — **do not delete** |
| `mlruns/` | **1.5 GB** | MLflow history; prunable, low value, **not re-derivable** — ask before touching |
| `models/` | 380 MB | run directories. Every one is a baseline or a record |
| tracked files | **257** | small and tidy for a project this size |
| `.git` | 96 KB (worktree) | v2 is a **worktree** of `/Users/ollie/Project` |

The 2026-08-08 sweep already reclaimed 395.9 MB of orphaned interim cache, and
that was done with a disjointness assertion rather than a glob. There is **no
second pile like it** — the earlier one was created by the `_cache_path`
ephemeris key and that cost has been paid once.

**Nothing is proposed for deletion.** Deleting non-re-derivable data is a
stop-and-ask, and none of the above is worth the risk for the space it returns.

### 5.4 The data-of-record moved mid-session, inside a docs commit — 2026-08-15

**Found by audit.** `e337c1c`, whose message describes only stage 10.5's recall
result, also bumps **three DVC pointers**: `data/tables/catalogue.dvc`,
`data/csv/exofop.dvc` and `data/tables/labels.dvc`. Nothing in the message mentions data.

**What actually happened.** A catalogue refresh ran at **09:00 on 2026-08-15**,
partway through the branch-propensity CV run. It rotated `labels.parquet` into
`labels.previous.parquet` and wrote a new `labels.parquet` differing on
**`snr`, 566 rows**. `label`, `mission` and `depth` are **unchanged**.

**No measured number is affected, and this was checked rather than assumed.**
The shard sets predate the refresh — `tfrecords` 2026-07-25,
`viewset_tfrecords` 2026-08-07 — so no CV run read the new file. Stage 10.5's
gate slice comes from `mission`, which did not move; the harness draw depends on
`depth` and `label`, which did not move either. The 3.9b, 3.9c and 3.11c tables
all re-derive unchanged, and the 4.1 draw still reproduces at 580 hosts.

**Two consequences that are real.**

1. **A committed claim stopped reproducing.** The stage-3 label finding in 3.9c
   reads as contradicted against the working tree and is correct against the
   superseded pointer. Annotated in place; both DVC versions are still in the
   local cache.
2. **Catalogue `snr` is aux index 7** in the 8/9-dim layout, so **any future
   shard rebuild will not reproduce these runs.** Anything re-derived from
   rebuilt shards is a different measurement and must be labelled one.

**The rule this earns.** A DVC pointer move is a data change and gets its own
commit, whatever else is in flight. Two lines in a `docs:` commit is how the
data-of-record moves without anyone deciding that it should.

---
