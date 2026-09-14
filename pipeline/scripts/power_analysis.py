"""P2.1 — what effect is detectable at this n, and which metric should gate.

Three measurements over the Phase 1 arms, which differ in one thing and share
their rows, so the contrast is what a promotion gate actually reads:

  1. stability — each candidate metric's paired-bootstrap sd and its
     seed-to-seed spread across the arm's members;
  2. power, uniform — which metric detects a graded degradation of the whole
     ranking, relative to its own noise;
  3. power, localised — the same, with the degradation confined to the top of
     the ranking, which is the case pAUC over FPR in [0, 0.1] exists for.

    python pipeline/scripts/power_analysis.py                # all three
    python pipeline/scripts/power_analysis.py --n-boot 2000  # the recorded run

Read-only: opens the two predictions files and nothing else. Tracked rather
than run from a scratch directory, because change-log.md records an earlier
reproduction script that was not, and its authorship could not be verified.
Result: docs/experiments/p2-1-power-analysis-2026-09-14.md.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]

#: Two-sided alpha 0.05 at 80% power: z(0.975) + z(0.80).
Z_80_POWER = 2.802
#: Members per fold the clean run is planned at (#78 part B).
PLANNED_MEMBERS = 5

Metric = Callable[[np.ndarray, np.ndarray], float]


def recall_at_fpr(y: np.ndarray, p: np.ndarray, fpr: float = 0.01) -> float:
    """The served definition, from `api/app/routes/model.py::_confusion_at_fpr`:
    the threshold is the (1 - fpr) quantile of the negatives, strictly exceeded.
    Reproduced here so these numbers are comparable with the console's."""
    neg, pos = p[y == 0], p[y == 1]
    if neg.size == 0 or pos.size == 0:
        return float("nan")
    return float((pos > float(np.quantile(neg, 1.0 - fpr))).sum()) / float(pos.size)


def _guarded(fn: Metric) -> Metric:
    def inner(y: np.ndarray, p: np.ndarray) -> float:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(fn(y, p))

    return inner


#: pAUC is McClish-standardised, so it shares AUC's scale and 0.5 floor. The
#: ranking by stability does not depend on that choice.
METRICS: dict[str, Metric] = {
    "recall @1% FPR": lambda y, p: recall_at_fpr(y, p, 0.01),
    "recall @5% FPR": lambda y, p: recall_at_fpr(y, p, 0.05),
    "pAUC FPR<=0.1": _guarded(lambda y, p: roc_auc_score(y, p, max_fpr=0.1)),
    "ROC-AUC": _guarded(roc_auc_score),
    "PR-AUC": _guarded(average_precision_score),
}

MEMBERS = [f"member_score_{j}" for j in range(3)]


def load_arms(arm_a: Path, arm_b: Path) -> pd.DataFrame:
    """Inner-join on the identifying columns so every bootstrap draw is paired."""
    keep = ["tic_id", "mission", "label", "dv_usable", "score", *MEMBERS]
    on = ["tic_id", "mission", "label", "dv_usable"]
    joined = pd.read_parquet(arm_a)[keep].merge(
        pd.read_parquet(arm_b)[keep], on=on, suffixes=("_a", "_b")
    )
    if joined.empty:
        raise ValueError(f"{arm_a} and {arm_b} share no rows — the contrast is not paired")
    return joined


def paired_bootstrap(
    y: np.ndarray, pa: np.ndarray, pb: np.ndarray, fn: Metric, n_boot: int, seed: int
) -> float:
    """sd of metric(b) - metric(a) under row resampling.

    One resampled index for both arms, so this is the contrast's own
    variability and not the sum of two independent ones.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[i] = fn(y[idx], pb[idx]) - fn(y[idx], pa[idx])
    return float(np.std(draws, ddof=1))


def seed_spread(y: np.ndarray, members: np.ndarray, fn: Metric) -> float:
    """Spread across one arm's members: identical data and architecture, so
    what is left is the training seed."""
    return float(np.std([fn(y, members[:, j]) for j in range(members.shape[1])], ddof=1))


def stability(df: pd.DataFrame, label: str, n_boot: int, seed: int) -> None:
    y = df["label"].to_numpy(dtype=int)
    pa, pb = df["score_a"].to_numpy(float), df["score_b"].to_numpy(float)
    members = df[[f"{m}_a" for m in MEMBERS]].to_numpy(dtype=float)
    n_neg = int((y == 0).sum())

    print(f"\n{'=' * 92}")
    print(f"{label}: n={len(y)}  positives={int((y == 1).sum())}  negatives={n_neg}")
    print(
        f"  negatives above the 1% FPR cut: {max(1, round(0.01 * n_neg))}   above 5%: "
        f"{max(1, round(0.05 * n_neg))}"
    )
    print("=" * 92)
    header = (
        f"{'metric':<16}{'arm C':>9}{'arm D':>9}{'D-C':>9}{'boot sd':>10}"
        f"{'seed sd':>9}{'sd @5':>9}{'MDE @5':>9}"
    )
    print(header)
    print("-" * len(header))
    for name, fn in METRICS.items():
        va, vb = fn(y, pa), fn(y, pb)
        boot = paired_bootstrap(y, pa, pb, fn, n_boot, seed)
        sd_seed = seed_spread(y, members, fn)
        # Averaging M members shrinks the seed component by sqrt(M); the
        # sampling component does not move, because it is the same rows.
        total5 = float(np.hypot(boot, sd_seed / np.sqrt(PLANNED_MEMBERS)))
        print(
            f"{name:<16}{va:>9.4f}{vb:>9.4f}{vb - va:>9.4f}{boot:>10.4f}"
            f"{sd_seed:>9.4f}{total5:>9.4f}{Z_80_POWER * total5:>9.4f}"
        )


def _degrade(p: np.ndarray, w: float, top_frac: float, seed: int) -> np.ndarray:
    """Rank-space Gaussian jitter, optionally confined to the top `top_frac`.

    Monotonic in w and identical to the served ranking at w = 0, so the only
    thing that changes between arms of this comparison is ranking quality.
    """
    pct = rankdata(p) / len(p)
    out = pct.copy()
    sel = pct >= (1.0 - top_frac) if top_frac < 1.0 else np.ones(len(p), dtype=bool)
    out[sel] = out[sel] + np.random.default_rng(seed).normal(0.0, w, int(sel.sum()))
    return out


def power(
    df: pd.DataFrame, n_boot: int, seed: int, top_frac: float, ws: list[float], n_reps: int
) -> None:
    """Mean z over `n_reps` independent degradations, not one.

    A single realisation of the jitter gives a noisy z — enough to reorder the
    metrics between runs, which is how this was caught. The observed change is
    averaged over repetitions; the bootstrap sd is estimated once per cell,
    being far the more stable of the two.
    """
    y = df["label"].to_numpy(dtype=int)
    p = df["score_a"].to_numpy(dtype=float)
    scope = "the whole ranking" if top_frac >= 1.0 else f"the top {top_frac:.0%} only"
    print(f"\n--- degradation applied to {scope}, mean of {n_reps} realisations ---")
    print("z = mean |change| / paired bootstrap sd of that change; 1.96 is significance")
    header = f"{'metric':<16}" + "".join(f"{'w=' + format(w, 'g'):>10}" for w in ws)
    print(header)
    print("-" * len(header))
    rng = np.random.default_rng(seed)
    for name, fn in METRICS.items():
        cells = []
        for w in ws:
            changes = [
                abs(fn(y, p) - fn(y, _degrade(p, w, top_frac, seed + r))) for r in range(n_reps)
            ]
            q = _degrade(p, w, top_frac, seed)
            draws = np.empty(n_boot)
            for i in range(n_boot):
                idx = rng.integers(0, len(y), len(y))
                draws[i] = fn(y[idx], p[idx]) - fn(y[idx], q[idx])
            sd = float(np.std(draws, ddof=1))
            cells.append(float(np.mean(changes)) / sd if sd > 0 else float("nan"))
        print(f"{name:<16}" + "".join(f"{c:>10.2f}" for c in cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a", type=Path, default=ROOT / "models/phase1/arm-c-control")
    parser.add_argument("--arm-b", type=Path, default=ROOT / "models/phase1/arm-d-difference")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-reps", type=int, default=25, help="degradation realisations per cell")
    args = parser.parse_args()

    df = load_arms(args.arm_a / "predictions.parquet", args.arm_b / "predictions.parquet")
    print(f"arm C = {args.arm_a.name}, arm D = {args.arm_b.name}, paired rows {len(df)}")
    print(f"bootstrap draws {args.n_boot}, seed {args.seed}, members per arm {len(MEMBERS)}")
    print(f"'sd @5' and 'MDE @5' are at {PLANNED_MEMBERS} members per fold")

    tess = df[df["mission"] == "TESS"]
    dv = tess[tess["dv_usable"].astype(bool)]
    stability(tess, "TESS — the gating slice", args.n_boot, args.seed)
    stability(dv, "TESS and dv_usable — the Phase 1 contrast slice", args.n_boot, args.seed)
    stability(df, "all missions pooled", args.n_boot, args.seed)

    print(f"\n{'=' * 92}\nPower on the TESS slice\n{'=' * 92}")
    power(tess, args.n_boot // 4, args.seed, 1.0, [0.02, 0.05, 0.1, 0.2], args.n_reps)
    for top in (0.05, 0.10, 0.20):
        power(tess, args.n_boot // 4, args.seed, top, [0.1], args.n_reps)


if __name__ == "__main__":
    main()
