"""Cross-validate the per-diagnostic branch model over the view-set shards.

    python pipeline/scripts/train_branches.py \
        --shards data/processed/viewset_tfrecords \
        --out models/cv/branches-$(date +%Y%m%d)

Writes `cv_summary.json`. Comparing it to the incumbent is `promotion_gate.py`,
and promoting anything is a separate, deliberate step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from exoplanet_hunter.training.train_branches import CVConfig, run_cv
from exoplanet_hunter.utils import get_logger, set_global_seed

log = get_logger(__name__)


class _Config:
    """Attribute access over the model YAML, matching the Hydra node shape."""

    def __init__(self, data: dict) -> None:
        self.__dict__.update(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, default=Path("data/processed/viewset_tfrecords"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--model-config", type=Path, default=Path("pipeline/conf/model/cnn_branches.yaml")
    )
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    raw = yaml.safe_load(args.model_config.read_text()) if args.model_config.exists() else {}
    cv = raw.get("cross_validation", {})
    training = raw.get("training", {})
    config = CVConfig(
        n_splits=args.n_splits or int(cv.get("n_splits", 5)),
        val_frac=float(cv.get("val_frac_within_fold", 0.2)),
        epochs=args.epochs or int(training.get("epochs", 40)),
        batch_size=int(training.get("batch_size", 32)),
        patience=int(training.get("patience", 8)),
        learning_rate=float(training.get("learning_rate", 1e-3)),
        seed=args.seed,
    )
    log.info("[train-branches] %s -> %s  (%s)", args.shards, args.out, config)
    run_cv(args.shards, args.out, config=config, model_cfg=_Config(raw))


if __name__ == "__main__":
    main()
