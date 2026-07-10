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

EXOFOP_URLS = {
    "tois.csv": "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv",
    "ctois.csv": "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv",
}


def _run(cmd: list[str]) -> None:
    """Stream a subprocess from the repo root; non-zero exit fails the task."""
    get_run_logger().info("$ %s", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


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
def refresh_label_catalogue() -> Path:
    """Rebuild labels.parquet from the TAP services, keeping the previous
    version aside for the leakage guard + trigger."""
    labels = REPO_ROOT / "data" / "labels" / "labels.parquet"
    previous = labels.with_suffix(".previous.parquet")
    if labels.exists():
        shutil.copy(labels, previous)
    # build_dataset stage 1 only would be ideal; the script is monolithic, so
    # rebuild the catalogue via the library the same way it does.
    _run(
        [
            PYTHON,
            "-c",
            "from pathlib import Path\n"
            "from exoplanet_hunter.data.catalog import CatalogRequest, build_label_catalog\n"
            "from exoplanet_hunter.data.exofop import enrich_catalog_snr\n"
            "cat = build_label_catalog(CatalogRequest(n_confirmed=500, n_false_pos=500, seed=42),"
            " out_dir=Path('data/labels'))\n"
            "cat['mission'] = cat.get('mission', 'TESS')\n"
            "cat = enrich_catalog_snr(cat, Path('data/catalogue/candidates.parquet'))\n"
            "cat.to_parquet('data/labels/labels.parquet', index=False)\n",
        ]
    )
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
def train() -> None:
    """Local training, or the GPU burst when $BURST_CMD is set."""
    burst = os.environ.get("BURST_CMD")
    if burst:
        get_run_logger().info("dispatching to GPU burst: %s", burst)
        _run(shlex.split(burst))
    else:
        _run([PYTHON, "-m", "exoplanet_hunter.training.train"])


@task
def promotion_gate() -> bool:
    """Gate the newest CV run; promote (update registry) if it wins."""
    cv_root = REPO_ROOT / "models" / "cv"
    newest = max(cv_root.glob("*/cv_summary.json"), key=lambda p: p.stat().st_mtime)
    result = subprocess.run(
        [
            PYTHON,
            "pipeline/scripts/promotion_gate.py",
            str(newest.relative_to(REPO_ROOT)),
            "--promote",
        ],
        cwd=REPO_ROOT,
    )
    promoted = result.returncode == 0
    get_run_logger().info("promotion gate: %s", "PROMOTED" if promoted else "rejected")
    return promoted


@task
def publish() -> None:
    """Version the refreshed artefacts and sync them to R2."""
    _run(
        [
            "dvc",
            "add",
            "data/exofop",
            "data/catalogue",
            "data/labels",
            "data/processed/views.npz",
            "data/processed/tfrecords",
        ]
    )
    for run_dir in (REPO_ROOT / "models" / "cv").glob("*/"):
        if (run_dir / "cv_summary.json").exists() and not run_dir.with_suffix(".dvc").exists():
            _run(["dvc", "add", str(run_dir.relative_to(REPO_ROOT)).rstrip("/")])
    _run(["dvc", "push"])


# ------------------------------------------------------------------- flow --


@flow(name="exoplanet-hunter-refresh", log_prints=True)
def refresh_pipeline(
    min_new_labelled: int = 25,
    force_train: bool = False,
    train_enabled: bool = True,
    data_config: str = "default",
) -> None:
    download_exofop_exports()
    ingest_candidate_catalogue()
    previous = refresh_label_catalogue()
    validation_gates(previous)

    if not train_enabled:
        print("training disabled for this run — refresh + gates only")
        publish()
        return

    if decide_training(previous, min_new_labelled, force_train):
        preprocess_and_shard(data_config)
        train()
        promotion_gate()
    publish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-new-labelled", type=int, default=25)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument(
        "--data-config",
        default="default",
        help="Hydra data group for the build (e.g. 'full' for the expansion run)",
    )
    args = parser.parse_args()
    refresh_pipeline(
        min_new_labelled=args.min_new_labelled,
        force_train=args.force_train,
        train_enabled=not args.no_train,
        data_config=args.data_config,
    )
