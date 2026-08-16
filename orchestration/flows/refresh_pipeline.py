"""The V2 DAG: refresh → validate → [changed materially?] → train → promote → publish.

Run standalone (no Prefect server needed — the local API spins up per run):

    python orchestration/flows/refresh_pipeline.py                 # full refresh
    python orchestration/flows/refresh_pipeline.py --force-train   # expansion run
    python orchestration/flows/refresh_pipeline.py --no-train      # refresh + gates only

Design decisions, per the architecture doc:

  * **Burst, don't idle** — the training task shells out to the command in
    `$BURST_CMD` when set (e.g. a script that provisions a cloud GPU, runs
    the containerised trainer against R2-synced shards, tears down), and
    falls back to local training otherwise. The *decision* to spend money
    is `exoplanet_hunter.validation.trigger.evaluate_refresh`, unit-tested
    and explicit.
  * **Leakage guard in the DAG** — the refreshed catalogue is diffed against
    the previous one before anything trains; label flips are reported and
    excluded from the trigger count (they belong to the since-confirmed
    holdout).
  * **Gates are the same code CI runs** — validate_data.py and
    promotion_gate.py subprocesses, exit-code semantics, no forked logic.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import requests
from prefect import flow, get_run_logger, task

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
# launchd runs the weekly refresh with a bare PATH, so a plain "dvc" is not
# resolvable — take the one beside the interpreter running this flow.
DVC = shutil.which("dvc") or str(Path(PYTHON).parent / "dvc")

EXOFOP_URLS = {
    "tois.csv": "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv",
    "ctois.csv": "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv",
}

#: The rolling outer split. Every refresh extends it rather than rebuilding it,
#: so a candidate and the incumbent it is gated against partition their shared
#: population identically. See `roll_fold_assignment`.
ROLLING_FOLDS = REPO_ROOT / "models" / "fold_assignments" / "rolling.json"
N_SPLITS = 5
SEED = 42
#: Members per fold on the refresh path. More than one is what gives the
#: candidate a measured noise floor of its own; at one the promotion gate has
#: no floor to read and falls back to a constant already measured as too tight.
N_MODELS_PER_FOLD = int(os.environ.get("REFRESH_N_MODELS_PER_FOLD", "3"))


def _run(cmd: list[str]) -> None:
    """Stream a subprocess from the repo root; non-zero exit fails the task."""
    get_run_logger().info("$ %s", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _notify(message: str) -> None:
    """Best-effort ping to $NOTIFY_WEBHOOK_URL (Discord/Slack-compatible)."""
    url = os.environ.get("NOTIFY_WEBHOOK_URL")
    if not url:
        return
    try:
        requests.post(url, json={"content": message, "text": message}, timeout=10)
    except Exception as exc:
        get_run_logger().warning("notify failed: %s", exc)


# ------------------------------------------------------------------ tasks --


@task(retries=2, retry_delay_seconds=60)
def download_exofop_exports() -> None:
    dest = REPO_ROOT / "data" / "exofop"
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in EXOFOP_URLS.items():
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()
        (dest / name).write_bytes(resp.content)
        get_run_logger().info("fetched %s (%.1f MB)", name, len(resp.content) / 1e6)


@task
def ingest_candidate_catalogue() -> None:
    _run([PYTHON, "pipeline/scripts/ingest_exofop.py"])


@task(retries=2, retry_delay_seconds=120)
def refresh_label_catalogue(data_config: str) -> Path:
    """Rebuild labels.parquet from the TAP services, keeping the previous
    version aside for the leakage guard + trigger.

    Must run the *same* data group the build uses: a capped rebuild here
    silently shrinks the data-of-record and starves the refresh trigger.
    """
    labels = REPO_ROOT / "data" / "labels" / "labels.parquet"
    previous = labels.with_suffix(".previous.parquet")
    if labels.exists():
        shutil.copy(labels, previous)
    _run([PYTHON, "pipeline/scripts/refresh_labels.py", f"data={data_config}"])
    return previous


@task
def validation_gates(previous_labels: Path) -> None:
    cmd = [PYTHON, "pipeline/scripts/validate_data.py", "--strict"]
    if previous_labels.exists():
        cmd += ["--previous-labels", str(previous_labels)]
    _run(cmd)


@task
def decide_training(previous_labels: Path, min_new_labelled: int, force: bool) -> bool:
    from exoplanet_hunter.validation.trigger import evaluate_refresh

    if not previous_labels.exists():
        get_run_logger().info("no previous catalogue — first run always trains")
        return True
    decision = evaluate_refresh(
        pd.read_parquet(previous_labels),
        pd.read_parquet(REPO_ROOT / "data" / "labels" / "labels.parquet"),
        min_new_labelled=min_new_labelled,
        force=force,
    )
    get_run_logger().info("%s", decision)
    return decision.should_train


@task
def preprocess_and_shard(data_config: str) -> None:
    _run([PYTHON, "pipeline/scripts/build_dataset.py", f"data={data_config}"])
    _run([PYTHON, "pipeline/scripts/shard_views.py"])


@task
def roll_fold_assignment() -> Path:
    """Extend the rolling fold map to cover whatever the catalogue now holds.

    Without this the weekly gate compares across two partitions of two
    populations, and part of every margin is only which rows landed where.
    Rebuilding the map each refresh re-partitions everything; pinning a fixed
    one silently drops each week's new targets, because an uncovered group is
    dropped rather than placed somewhere convenient. Extending keeps every
    target in the fold that has always held it out and still lets new ones in.
    """
    import numpy as np
    import pandas as pd
    from exoplanet_hunter.training.splits import (
        build_fold_assignment,
        extend_fold_assignment,
        load_fold_assignment,
        write_fold_assignment,
    )

    # The INTERSECTION of both shard sets, not either one alone. A group only
    # one trainer can see has no out-of-fold score from the other, so the two
    # cannot be compared or ensembled on it — the same reason
    # build_fold_assignment.py takes a shared population. Reading the branch
    # index alone would cover 5,426 groups against the dual-view trainer's
    # 5,380 and quietly drop the difference at training time.
    frames = []
    for rel in ("data/processed/tfrecords", "data/processed/viewset_tfrecords"):
        path = REPO_ROOT / rel / "index.parquet"
        if not path.exists():
            raise FileNotFoundError(f"no shard index at {path} — build the shard set first")
        frames.append(pd.read_parquet(path)[["tic_id", "label"]].drop_duplicates("tic_id"))
    shared = frames[0]
    for frame in frames[1:]:
        shared = shared.merge(frame, on=["tic_id", "label"], how="inner")
    groups = shared["tic_id"].to_numpy(int)
    y = shared["label"].to_numpy(int)
    get_run_logger().info(
        "[folds] shared population: %d groups (%s)",
        len(groups),
        " ∩ ".join(str(len(f)) for f in frames),
    )

    ROLLING_FOLDS.parent.mkdir(parents=True, exist_ok=True)
    if ROLLING_FOLDS.exists():
        existing, _ = load_fold_assignment(ROLLING_FOLDS)
        assignment, added = extend_fold_assignment(
            existing, groups, y, n_splits=N_SPLITS, seed=SEED
        )
        get_run_logger().info(
            "[folds] rolling map extended: %d carried over, %d added", len(existing), added
        )
    else:
        assignment = build_fold_assignment(groups, y, n_splits=N_SPLITS, seed=SEED)
        get_run_logger().info("[folds] no rolling map — built one over %d groups", len(assignment))

    sizes = np.bincount(list(assignment.values()), minlength=N_SPLITS).tolist()
    write_fold_assignment(
        ROLLING_FOLDS,
        assignment,
        n_groups=len(assignment),
        n_folds=N_SPLITS,
        seed=SEED,
        rolling=True,
        fold_sizes=sizes,
    )
    return ROLLING_FOLDS


@task
def train(data_config: str, fold_assignment: Path | None = None) -> None:
    """Local training, or the GPU burst when $BURST_CMD is set.

    The trainer reads shards already on disk, so the data group changes
    nothing about *what* trains — but the MLflow run is named
    cnn-cv-<data.name>, and without the override a full build was recorded
    as "cnn-cv-default".

    Two overrides make the weekly decision readable. `n_models_per_fold` gives
    the candidate its **own** measured noise floor: at one member per fold the
    variance block carries no reseeding spread, `decision_floor` returns None,
    and the gate falls back to `LEGACY_RECALL_TOLERANCE = 0.02` — a bar measured
    at 0.68 sigma, which rejects a candidate whose true recall equals the
    incumbent's about 31% of the time. The fold assignment is what makes two
    successive candidates comparable at all.
    """
    burst = os.environ.get("BURST_CMD")
    overrides = [f"data={data_config}", f"train.n_models_per_fold={N_MODELS_PER_FOLD}"]
    if fold_assignment is not None:
        overrides.append(f"model.cross_validation.fold_assignment={fold_assignment}")
    if burst:
        get_run_logger().info("dispatching to GPU burst: %s", burst)
        _run(shlex.split(burst) + overrides)
    else:
        _run([PYTHON, "-m", "exoplanet_hunter.training.train", *overrides])


@task
def promotion_gate() -> bool:
    """Gate the newest CV run; promote (update registry) if it wins."""
    cv_root = REPO_ROOT / "models" / "cv"
    newest = max(cv_root.glob("*/cv_summary.json"), key=lambda p: p.stat().st_mtime)
    # The incumbent is gated against its RE-BASELINED summary, not the one
    # registry.json names. `ca906040`'s own summary predates the per_mission
    # block, so without this the gate cannot slice its population and refuses to
    # compare at all — REJECT before a metric is read, whatever the candidate
    # scored. 3.7 found that and closed it with this flag; the flow never passed
    # it, so the weekly loop kept hitting the defect 3.7 had already fixed.
    cmd = [
        PYTHON,
        "pipeline/scripts/promotion_gate.py",
        str(newest.relative_to(REPO_ROOT)),
        "--promote",
    ]
    rebaselined = REPO_ROOT / "models" / "cv" / "incumbent-rebaselined" / "cv_summary.json"
    if rebaselined.exists():
        cmd += ["--incumbent-summary", str(rebaselined.relative_to(REPO_ROOT))]
    else:
        get_run_logger().warning(
            "[gate] no re-baselined incumbent summary at %s — the gate will compare "
            "pooled means and may refuse on population mismatch",
            rebaselined,
        )
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    promoted = result.returncode == 0
    get_run_logger().info("promotion gate: %s", "PROMOTED" if promoted else "rejected")
    return promoted


@task
def publish() -> None:
    """Version the refreshed artefacts and sync them to R2. CV runs are
    allowlisted (promoted run + already-tracked dirs), never globbed — a
    tuning campaign leaves dozens of trial checkpoints in models/cv."""
    from exoplanet_hunter.validation.promotion import publishable_cv_dirs

    _run(
        [
            DVC,
            "add",
            "data/csv/exofop",
            "data/tables/catalogue",
            "data/tables/labels",
            "data/processed/views.npz",
            "data/processed/tfrecords",
        ]
    )
    for run_dir in publishable_cv_dirs(REPO_ROOT / "models"):
        _run([DVC, "add", str(run_dir.relative_to(REPO_ROOT))])
    _run([DVC, "push"])


# ------------------------------------------------------------------- flow --


@flow(name="exoplanet-hunter-refresh", log_prints=True)
def refresh_pipeline(
    min_new_labelled: int = 25,
    force_train: bool = False,
    train_enabled: bool = True,
    data_config: str = "full",
) -> None:
    download_exofop_exports()
    ingest_candidate_catalogue()
    previous = refresh_label_catalogue(data_config)
    validation_gates(previous)

    if not train_enabled:
        print("training disabled for this run — refresh + gates only")
        publish()
        return

    if decide_training(previous, min_new_labelled, force_train):
        preprocess_and_shard(data_config)
        folds = roll_fold_assignment()
        train(data_config, fold_assignment=folds)
        promoted = promotion_gate()
        registry = json.loads((REPO_ROOT / "models" / "registry.json").read_text())
        _notify(
            f"exoplanet-hunter refresh: trained; promotion "
            f"{'PROMOTED' if promoted else 'rejected'} — serving run "
            f"{registry['run_id'][:8]} (AUC {registry['test_roc_auc_mean']:.4f})"
        )
    else:
        _notify("exoplanet-hunter refresh: catalogue updated, no retrain warranted")
    publish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-new-labelled", type=int, default=25)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument(
        "--data-config",
        default="full",
        help="Hydra data group for the build; 'full' is the production dataset "
        "(data=default is the old capped build and would replace the shards)",
    )
    args = parser.parse_args()
    refresh_pipeline(
        min_new_labelled=args.min_new_labelled,
        force_train=args.force_train,
        train_enabled=not args.no_train,
        data_config=args.data_config,
    )
