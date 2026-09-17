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
#: The canonical mission map. Dual-view predictions files carry no `mission`
#: column, so the gating slice has to be joined on rather than read off.
MISSION_TABLE = ROOT / "data/tables/labels/labels.parquet"
#: Read from a run's member checkpoint names: `cv_summary.json` carries no model
#: name for runs that predate `run_config`.
ARCHITECTURES = ("cnn_dualview", "cnn_branches")
#: Where multi-member runs live. `models/stage9/` holds single-member arms.
CENSUS_ROOTS = (ROOT / "models/cv", ROOT / "models/phase1")

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


def architecture_of(run_dir: Path) -> str:
    """The architecture a run trained, from its fold-0 member checkpoint names.

    Raises rather than guessing. A seed sd pooled across architectures is the
    category error `docs/index.md` rule 7 forbids, and the branch and dual-view
    sds differ by 1.7x (2026-09-17 census), so a wrong label is a wrong MDE.
    """
    fold0 = run_dir / "fold_0"
    found = {a for a in ARCHITECTURES if any(fold0.glob(f"model_*_{a}.keras"))}
    if len(found) != 1:
        raise ValueError(
            f"{run_dir}: expected exactly one architecture in {fold0}, found {sorted(found)}"
        )
    return found.pop()


def mission_map() -> pd.DataFrame:
    """tic_id -> mission, from the labels table rather than a sibling arm."""
    if not MISSION_TABLE.exists():
        raise FileNotFoundError(f"no mission map at {MISSION_TABLE}; run the label build first")
    return pd.read_parquet(MISSION_TABLE)[["tic_id", "mission"]].drop_duplicates()


def seed_census(missions: pd.DataFrame, slice_mission: str = "TESS") -> pd.DataFrame:
    """Per-run seed sd on one mission slice, for every multi-member run on disk.

    P2.1 read one arm's three members and got a 2-df estimate whose per-arm
    values span 6x. This walks every run that wrote `member_score_*` columns.
    """
    rows = []
    for root in CENSUS_ROOTS:
        for path in sorted(root.glob("*/predictions.parquet")):
            frame = pd.read_parquet(path)
            members = [c for c in frame.columns if c.startswith("member_score_")]
            if len(members) < 2:
                continue
            if "mission" not in frame.columns:
                frame = frame.merge(missions, on="tic_id", how="inner")
            y_col = "y_true" if "y_true" in frame.columns else "label"
            sl = frame[frame["mission"] == slice_mission]
            if len(sl) < 500 or sl[y_col].nunique() < 2:
                continue
            y = sl[y_col].to_numpy(dtype=int)
            row = {
                "run": path.parent.name,
                "arch": architecture_of(path.parent),
                "members": len(members),
                "n": len(sl),
            }
            for name, fn in METRICS.items():
                draws = [fn(y, sl[m].to_numpy(dtype=float)) for m in members]
                row[name] = float(np.std(draws, ddof=1))
            rows.append(row)
    if not rows:
        raise ValueError(f"no multi-member run scored on {slice_mission}; the census is empty")
    return pd.DataFrame(rows)


def pooled_sd(census: pd.DataFrame, metric: str) -> tuple[float, int]:
    """Variance-pooled sd and its degrees of freedom, sum (M-1) over runs."""
    dof = int((census["members"] - 1).sum())
    var = float(((census["members"] - 1) * census[metric] ** 2).sum() / dof)
    return float(np.sqrt(var)), dof


def report_census(census: pd.DataFrame, slice_mission: str) -> None:
    """Per-run seed sds, then the sd pooled by architecture with its df."""
    print(f"\n{'=' * 92}")
    print(f"SEED CENSUS — every multi-member run on disk, {slice_mission} slice")
    print("=" * 92)
    header = f"{'run':<42}{'arch':<10}{'M':>3}{'n':>6}" + "".join(f"{k:>16}" for k in METRICS)
    print(header)
    print("-" * len(header))
    for _, r in census.iterrows():
        print(
            f"{r['run']:<42}{r['arch']:<10}{r['members']:>3}{r['n']:>6}"
            + "".join(f"{r[k]:>16.4f}" for k in METRICS)
        )

    print(f"\n{'-' * 92}\npooled seed sd, variance-pooled over the runs above")
    print("-" * 92)
    print(f"{'architecture':<16}{'runs':>6}{'df':>5}" + "".join(f"{k:>16}" for k in METRICS))
    print("-" * 92)
    for arch, group in [*census.groupby("arch"), ("ALL POOLED", census)]:
        cells = []
        for metric in METRICS:
            sd, _ = pooled_sd(group, metric)
            cells.append(f"{sd:>16.4f}")
        sd_line = f"{arch:<16}{len(group):>6}{int((group['members'] - 1).sum()):>5}" + "".join(
            cells
        )
        print(sd_line)


def report_mde(census: pd.DataFrame, boots: dict[str, float]) -> None:
    """The bar a challenger must clear, by architecture, at 3/5/10 members."""
    print(f"\n{'=' * 92}")
    print("MDE at 80% power — pooled census sd, measured bootstrap sd, two independent arms")
    print("=" * 92)
    header = f"{'architecture':<16}{'M':>4}" + "".join(f"{k:>16}" for k in METRICS)
    print(header)
    print("-" * len(header))
    for arch, group in [*census.groupby("arch"), ("ALL POOLED", census)]:
        for members in (3, PLANNED_MEMBERS, 10):
            cells = []
            for metric in METRICS:
                sd, _ = pooled_sd(group, metric)
                cells.append(f"{contrast_mde(sd, boots[metric], members):>16.4f}")
            print(f"{arch:<16}{members:>4}" + "".join(cells))


def contrast_mde(sd_seed: float, sd_boot: float, members: int) -> float:
    """Smallest contrast detectable at 80% power with `members` per arm.

    Two seed terms, not one: the quantity under test is a difference of two run
    means, which is what `promotion.py::decision_floor` has read since 4.1b and
    what P2.1's `hypot(boot, sd_seed / sqrt(M))` did not. Measured 2026-09-17,
    the member-paired contrast sd is 0.0175 against 0.0130 predicted under
    independence, so the arms' seed draws do not cancel and both terms stand.
    """
    return Z_80_POWER * float(np.hypot(sd_boot, sd_seed * np.sqrt(2.0 / members)))


def stability(df: pd.DataFrame, label: str, n_boot: int, seed: int) -> dict[str, float]:
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
        f"{'seed sd':>9}{'MDE P2.1':>10}{'MDE 2-arm':>11}"
    )
    print(header)
    print("-" * len(header))
    boots: dict[str, float] = {}
    for name, fn in METRICS.items():
        va, vb = fn(y, pa), fn(y, pb)
        boot = boots[name] = paired_bootstrap(y, pa, pb, fn, n_boot, seed)
        sd_seed = seed_spread(y, members, fn)
        # `MDE P2.1` carries one seed term, which is what the 2026-09-14 file
        # printed; `MDE 2-arm` is the correction — a contrast has two. Both are
        # shown so the published figure stays locatable beside the right one.
        one_term = Z_80_POWER * float(np.hypot(boot, sd_seed / np.sqrt(PLANNED_MEMBERS)))
        two_term = contrast_mde(sd_seed, boot, PLANNED_MEMBERS)
        print(
            f"{name:<16}{va:>9.4f}{vb:>9.4f}{vb - va:>9.4f}{boot:>10.4f}"
            f"{sd_seed:>9.4f}{one_term:>10.4f}{two_term:>11.4f}"
        )
    return boots


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
    parser.add_argument("--no-census", action="store_true", help="skip the multi-run seed census")
    args = parser.parse_args()

    df = load_arms(args.arm_a / "predictions.parquet", args.arm_b / "predictions.parquet")
    print(f"arm C = {args.arm_a.name}, arm D = {args.arm_b.name}, paired rows {len(df)}")
    print(f"bootstrap draws {args.n_boot}, seed {args.seed}, members per arm {len(MEMBERS)}")
    print(f"MDE columns are at {PLANNED_MEMBERS} members per fold; 'MDE 2-arm' is the correct one")

    census = None if args.no_census else seed_census(mission_map())
    if census is not None:
        report_census(census, "TESS")

    tess = df[df["mission"] == "TESS"]
    dv = tess[tess["dv_usable"].astype(bool)]
    boots = stability(tess, "TESS — the gating slice", args.n_boot, args.seed)
    if census is not None:
        report_mde(census, boots)
    stability(dv, "TESS and dv_usable — the Phase 1 contrast slice", args.n_boot, args.seed)
    stability(df, "all missions pooled", args.n_boot, args.seed)

    print(f"\n{'=' * 92}\nPower on the TESS slice\n{'=' * 92}")
    power(tess, args.n_boot // 4, args.seed, 1.0, [0.02, 0.05, 0.1, 0.2], args.n_reps)
    for top in (0.05, 0.10, 0.20):
        power(tess, args.n_boot // 4, args.seed, top, [0.1], args.n_reps)


if __name__ == "__main__":
    main()
