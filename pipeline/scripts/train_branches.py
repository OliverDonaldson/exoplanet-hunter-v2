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

from exoplanet_hunter.datasets.viewset_augment import AugmentConfig
from exoplanet_hunter.training.train_branches import CVConfig, run_cv
from exoplanet_hunter.utils import get_logger, set_global_seed

log = get_logger(__name__)


class _Config:
    """Attribute access over the model YAML, matching the Hydra node shape."""

    def __init__(self, data: dict) -> None:
        self.__dict__.update(data)


def _augment_config(raw: dict) -> AugmentConfig | None:
    """The declared augmentation, or None when the config disables it.

    A config with no `augmentation` block gets the defaults rather than nothing:
    silently training without augmentation is what made run 1's comparison
    against the incumbent unlike-for-like.
    """
    declared = raw.get("augmentation", {})
    if declared.get("enabled", True) is False:
        return None
    return AugmentConfig(
        **{k: float(v) for k, v in declared.items() if k in AugmentConfig.__dataclass_fields__}
    )


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
    parser.add_argument(
        "--n-models-per-fold",
        type=int,
        default=None,
        help="models averaged per fold; >1 also makes seed variance measurable "
        "separately from fold difficulty (see summary.variance)",
    )
    parser.add_argument(
        "--init-filters",
        type=int,
        default=None,
        help="override the declared per-branch filter count. This is the capacity arm, so "
        "that capacity is tested within one architecture rather than across two",
    )
    args = parser.parse_args()

    if not args.model_config.exists():
        # Falling back to `{}` silently trains every architecture default, and
        # nothing in cv_summary.json would record which model actually ran.
        raise SystemExit(f"no model config at {args.model_config} — run from the repository root")
    set_global_seed(args.seed)
    raw = yaml.safe_load(args.model_config.read_text())
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
        n_models_per_fold=args.n_models_per_fold or int(training.get("n_models_per_fold", 1)),
        augment=_augment_config(raw),
    )
    if args.init_filters is not None:
        # Overridden in the resolved config rather than the YAML, so the run
        # records what it actually built (run_config.model_config) while the
        # declared architecture on disk stays the main line's.
        raw["init_filters"] = args.init_filters
        log.info("[train-branches] init_filters overridden to %d", args.init_filters)

    log.info("[train-branches] %s -> %s  (%s)", args.shards, args.out, config)
    run_cv(args.shards, args.out, config=config, model_cfg=_Config(raw))


if __name__ == "__main__":
    main()
