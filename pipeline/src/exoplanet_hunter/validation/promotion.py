"""Model promotion gate: beat the baseline before you cheer — machine-enforced.

A freshly trained CV run produces `cv_summary.json` (written by the trainer).
This module decides whether that run replaces the incumbent in
`models/registry.json`:

  * primary: mean CV test ROC-AUC must be strictly higher than the
    incumbent's;
  * calibration guard: mean CV Brier must not degrade by more than
    `brier_tolerance` — a model that ranks better but calibrates worse is
    not an upgrade for a system whose whole point is trustworthy
    probabilities;
  * reliability guard: mean CV ECE must not degrade by more than
    `ece_tolerance` — Brier alone is blind to this, since a discrimination
    gain can pay for arbitrary miscalibration. Skipped when either summary
    predates the `test_ece` field;
  * shortlist guard: recall at 1% FPR must not fall by more than
    `recall_tolerance`. AUC scores ranking at every threshold; the follow-up
    shortlist this system exists to produce lives at exactly one. The stage 4
    branch model is the case that motivated it — TESS AUC within 0.002
    of the incumbent while recall at 1% FPR fell 0.307 -> 0.238;
  * the first-ever run promotes automatically (there is no bar yet — the RF
    baseline becomes the incumbent as soon as it's registered).

**Which population decides.** When both summaries carry a `per_mission` block
the gate reads `GATE_MISSION`, not the aggregate. TESS is 100% of the
deployment population; the all-mission aggregate's weights are a sampling
decision (Kepler is drawn at exactly 1,250/1,250 by construction) and it is
reported but never gates. `DIAGNOSTIC_MISSIONS` are mandatory reported slices:
a drop beyond `mission_alarm` does not block promotion, but it is surfaced in
the decision so it cannot be promoted past in silence. Summaries without the
block fall back to the aggregate, which is what every run before 2026-08-05
carries.

The registry is a plain JSON pointer, not MLflow state: the serving path and
the CI gate both read it without a tracking-server dependency.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

REGISTRY_NAME = "registry.json"

#: Fewest paired folds at which a signed-rank p-value is worth printing. With
#: five pairs the two-sided Wilcoxon floor is p=0.0625, so it cannot reach 0.05
#: however lopsided the result — reporting one there invites reading "not
#: significant" as evidence of no effect. Below this the paired delta and the
#: effect size are the honest summary.
MIN_PAIRS_FOR_P_VALUE = 6

#: The deployment population — every scored candidate is TESS.
GATE_MISSION = "TESS"
#: Reported on every run, alarmed, never blocking. K2 is 9.7% of training and
#: was unbenchmarked until 2026-08-05; Kepler is a balanced draw.
DIAGNOSTIC_MISSIONS = ("Kepler", "K2")
#: Pooled slice reported alongside the gate and explicitly excluded from it.
AGGREGATE_SLICE = "all"


@dataclass(frozen=True)
class PairedFolds:
    """A candidate against an incumbent on the same folds, fold by fold.

    Comparing two run means asks whether one number exceeds another when each
    carries ~0.011 of training noise. Pairing on fold index removes fold
    difficulty from the comparison entirely, which is the larger of the two
    variance sources and the one that is identical between the runs.
    """

    deltas: list[float]
    effect_size: float
    p_value: float | None
    #: Whether fold *k* provably held out the same rows in both runs. False when
    #: a summary predates `run_config` or the folds differ in size — pairing
    #: still removes most of the fold-difficulty variance, but it is no longer
    #: exact and the reading should be quoted with that attached.
    exact: bool = True

    @property
    def wins(self) -> int:
        return sum(d > 0 for d in self.deltas)

    @property
    def mean(self) -> float:
        return float(sum(self.deltas) / len(self.deltas))

    def __str__(self) -> str:
        out = (
            f"paired over {len(self.deltas)} folds: mean {self.mean:+.4f}, "
            f"won {self.wins}/{len(self.deltas)}, d={self.effect_size:+.2f}"
        )
        if self.p_value is not None:
            out += f", p={self.p_value:.3f}"
        return out + ("" if self.exact else " (folds only approximately matched)")


def paired_folds(
    candidate: dict[str, Any], incumbent: dict[str, Any], metric: str = "test_roc_auc"
) -> PairedFolds | None:
    """Fold-by-fold deltas, or None when the two runs' folds are not comparable.

    Pairing is only meaningful if fold *k* held out the same rows in both runs,
    so it requires a matching fold count and, where both summaries record one,
    a matching seed and split count.
    """
    import numpy as np

    cand_folds, inc_folds = candidate.get("folds") or [], incumbent.get("folds") or []
    if not cand_folds or len(cand_folds) != len(inc_folds):
        return None
    cand_cfg, inc_cfg = candidate.get("run_config", {}), incumbent.get("run_config", {})
    if cand_cfg and inc_cfg:
        keys = ("seed", "n_splits")
        if any(cand_cfg.get(k) != inc_cfg.get(k) for k in keys if k in cand_cfg or k in inc_cfg):
            return None
    # Fold sizes are the only per-fold membership evidence a summary carries.
    # Equal sizes do not prove the same rows, but unequal sizes disprove it —
    # run 1 and run 2 differ here because three FFI rows arrived between them.
    sizes_match = [c.get("n_test") for c in cand_folds] == [i.get("n_test") for i in inc_folds]
    exact = bool(cand_cfg and inc_cfg and sizes_match)

    deltas = np.array(
        [float(c[metric]) - float(i[metric]) for c, i in zip(cand_folds, inc_folds, strict=True)]
    )
    if not np.isfinite(deltas).all():
        return None
    spread = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
    mean = float(deltas.mean())
    if spread > 1e-12:
        effect = mean / spread
    elif mean == 0.0:
        # Every fold identical: no effect, not an unmeasurable one. `inf * 0` is
        # nan, which loses every downstream inequality and reads as "could not
        # tell" for the one case that is certain.
        effect = 0.0
    else:
        # Every fold moved by the same non-zero amount — infinitely consistent.
        effect = math.copysign(math.inf, mean)

    p_value = None
    if len(deltas) >= MIN_PAIRS_FOR_P_VALUE and np.any(deltas != 0):
        from scipy.stats import wilcoxon

        p_value = float(wilcoxon(deltas).pvalue)
    return PairedFolds(deltas.tolist(), effect, p_value, exact)


@dataclass(frozen=True)
class DecisionFloor:
    """The smallest margin this run can resolve, from its own recorded variance.

    Stage 6 measured the floors and fixed the rule that turns them into a
    threshold: **`2 x seed_sd / sqrt(n_models_per_fold)`**. It yielded ~0.007 on
    gate AUC and **~0.034 on gate recall @1% FPR**, and the roadmap states the
    consequence plainly — *a recall @1% FPR margin under ~0.034 is not a
    decision*.

    The gate did not read any of it. It compared AUC with a strict `>` and no
    band at all, and rejected on recall at a hardcoded 0.02 — **tighter than the
    0.034 floor the same project had measured**, so a candidate whose true recall
    equalled the incumbent's was rejected by reseeding noise about a third of the
    time. Both numbers predate stage 6 and neither was revisited when it landed.

    A floor is not a licence to promote a worse model: the gate stays
    conservative and a tie still leaves the incumbent served. What the floor
    changes is what the decision is allowed to *claim* — a margin inside it is
    reported as level, not as having been beaten.
    """

    auc: float | None
    recall: float | None
    source: str

    def times(self, margin: float, floor: float | None) -> str:
        """`margin` as a multiple of its floor, for the decision text."""
        if floor is None or floor <= 0.0:
            return "no measured floor"
        return f"{abs(margin) / floor:.1f}x the {floor:.4f} floor"


#: Applied when a summary carries no `variance` block — every run before
#: 2026-08-08. Not a measurement, and the decision text says so.
LEGACY_RECALL_TOLERANCE = 0.02


#: Pooled median `pooled_gate_recall_seed_sd` over the 11 runs on disk that
#: measured one (0.0029-0.0632, a 22.1x spread). Used ONLY for the incumbent
#: term, and only until an incumbent is re-baselined carrying its own variance
#: block — no incumbent summary on disk has one. Pre-registered in roadmap 4.1b.
POOLED_RECALL_SEED_SD = 0.0353
#: The same, for AUC. Both are priors standing in for a measurement, and every
#: decision text that leans on one has to say so.
POOLED_SEED_SD = 0.0081


def _variance(summary: dict[str, Any]) -> tuple[dict[str, Any], int | None]:
    variance = summary.get("summary", {}).get("variance") or {}
    n_models = variance.get("n_models_per_fold")
    return variance, int(n_models) if n_models and n_models >= 1 else None


def decision_floor(
    summary: dict[str, Any], incumbent: dict[str, Any] | None = None
) -> DecisionFloor:
    """Resolvable margins, by stage 6's rule applied to what is being compared.

    The quantity under test is a **difference of two run means**, so the
    tolerance is `2 x se(delta)` with

        se(delta) = sqrt( sd_cand^2 / n_cand + sd_inc^2 / n_inc )

    not `2 x sd_cand / sqrt(n_cand)`, which is the se of the candidate's mean
    alone. Reading only the candidate ignored the incumbent's noise entirely —
    stage 6's caveat 1 says plainly that it should not — and let a noisier
    candidate earn itself a wider band to clear. Across the 11 runs on disk that
    quantity spans 22.1x.

    A noisier candidate still earns a wider band under this rule, correctly:
    its mean is genuinely less well known. What stops that being exploitable is
    `Verdict.UNRESOLVED`, which catches exactly the case where a margin has
    become comparable to its own floor. Adopted and pre-registered in 4.1b.
    """
    variance, n_models = _variance(summary)
    if n_models is None:
        return DecisionFloor(None, None, "no variance block — run with --n-models-per-fold")

    inc_variance, inc_n = _variance(incumbent or {})
    borrowed: list[str] = []

    def floor(key: str, pooled: float) -> float | None:
        sd = variance.get(key)
        if not sd:
            return None
        term = float(sd) ** 2 / n_models
        inc_sd = inc_variance.get(key)
        if inc_sd and inc_n:
            term += float(inc_sd) ** 2 / inc_n
        else:
            # The incumbent measured nothing, so a prior stands in for it. Named
            # in the source string rather than folded in silently: substituting
            # an assumption for a measurement without saying so is this
            # project's own recurring defect class.
            term += pooled**2 / n_models
            borrowed.append(key)
        return 2.0 * math.sqrt(term)

    auc = floor("seed_sd", POOLED_SEED_SD)
    recall = floor("pooled_gate_recall_seed_sd", POOLED_RECALL_SEED_SD)
    source = f"2 x se(candidate - incumbent), n={n_models}"
    if borrowed:
        source += (
            f"; incumbent variance unmeasured for {sorted(set(borrowed))} — "
            f"pooled prior used (recall {POOLED_RECALL_SEED_SD}, auc {POOLED_SEED_SD})"
        )
    return DecisionFloor(auc=auc, recall=recall, source=source)


class Verdict(StrEnum):
    """Three states, because stage 6's caveat 2 describes three.

    A `bool` made a margin at 0.8x its floor read the same as one at 0.1x. The
    roadmap's own words: *"If the result lands within a factor of ~1.5 of a
    threshold, the honest reading is unresolved — that is a stop-and-ask."*
    """

    PROMOTE = "PROMOTE"
    UNRESOLVED = "UNRESOLVED"
    REJECT = "REJECT"


#: A margin between `floor / 1.5` and `floor * 1.5` is not resolvable against a
#: floor estimated from three draws, whose own sampling spread is ~40% of its
#: value (stage 6, caveat 2).
UNRESOLVED_BAND = 1.5


def unresolved_against(margin: float, floor: float | None) -> bool:
    """Is `margin` too close to `floor` for the comparison to decide?"""
    if floor is None or floor <= 0.0:
        return False
    return floor / UNRESOLVED_BAND <= abs(margin) <= floor * UNRESOLVED_BAND


@dataclass
class PromotionDecision:
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)
    #: Reported slices whose AUC fell beyond `mission_alarm`. Advisory by
    #: default; blocking under `strict`, which is what an unattended loop uses —
    #: "owes a written explanation in the roadmap before promotion" is a
    #: condition no unattended run can satisfy.
    alarms: list[str] = field(default_factory=list)

    @property
    def promoted(self) -> bool:
        """Only PROMOTE promotes. UNRESOLVED is explicitly not a promotion."""
        return self.verdict is Verdict.PROMOTE

    def __str__(self) -> str:
        out = f"{self.verdict.value}: " + "; ".join(self.reasons)
        return out + ("\nALARM: " + "; ".join(self.alarms) if self.alarms else "")


def _mean(summary: dict[str, Any], metric: str) -> float:
    return float(summary["summary"][metric]["mean"])


def _mean_or_none(summary: dict[str, Any], metric: str) -> float | None:
    entry = summary.get("summary", {}).get(metric)
    return float(entry["mean"]) if entry else None


def _gate_slice(summary: dict[str, Any]) -> dict[str, float] | None:
    """The deployment slice's metrics, or None on a summary without the block.

    A summary that carries `per_mission` but no `GATE_MISSION` is not the same
    case as one that predates the block, and returning None for both is how the
    fallback gets entered while reporting that the summary is simply old. Worse,
    `_population_mismatch` compares mission *sets*, so two summaries that both
    lack TESS agree with each other and never trigger the refusal — leaving the
    gate deciding on pooled means with nothing saying so. Refuse instead.
    """
    per_mission = summary.get("per_mission")
    if not per_mission:
        return None
    gate = per_mission.get(GATE_MISSION)
    if gate is None:
        raise ValueError(
            f"summary carries a per_mission block with {sorted(per_mission)} but no "
            f"{GATE_MISSION}, which is 100% of the deployment population — refusing to "
            "fall back to pooled means, which would read as a like-for-like comparison"
        )
    return gate


def _population_mismatch(candidate: dict[str, Any], incumbent: dict[str, Any]) -> str | None:
    """Why these two summaries may not be measured on comparable rows.

    A cv_summary records metrics over whatever population its run scored, and
    nothing in the file says which rows those were. The incumbent `ca906040`
    was scored on 4,818 rows with zero K2; a current run scores 5,426 including
    527 K2 the incumbent's build predates. Comparing the two pooled means reads
    as a model difference and is partly a population difference.
    """
    cand_missions = set(candidate.get("per_mission", {})) - {AGGREGATE_SLICE}
    inc_missions = set(incumbent.get("per_mission", {})) - {AGGREGATE_SLICE}
    if not cand_missions or not inc_missions:
        return "one summary carries no per_mission block, so the rows behind each mean are unknown"
    only_candidate = sorted(cand_missions - inc_missions)
    only_incumbent = sorted(inc_missions - cand_missions)
    parts = [
        f"only the candidate scored {', '.join(only_candidate)}" if only_candidate else "",
        f"only the incumbent scored {', '.join(only_incumbent)}" if only_incumbent else "",
    ]
    return "; ".join(p for p in parts if p) or None


def _gate_population_drift(
    cand_slice: dict[str, float] | None, inc_slice: dict[str, float] | None
) -> str | None:
    """Whether the two gate slices are measured over the same number of rows.

    `_population_mismatch` compares mission *sets*, so two summaries that both
    carry TESS agree with each other however much their TESS membership differs.
    Measured 2026-08-08, the re-baselined incumbent gates on 2,367 TESS rows and
    run 2 on 2,399 — a 32-row gap that reads as a model difference. Row counts
    are the only membership evidence a summary carries, and equal counts do not
    prove equal rows; unequal counts disprove it. Alarmed rather than blocking,
    because the catalogue legitimately grows between runs.
    """
    if cand_slice is None or inc_slice is None:
        return None
    # `.get`, not `[]`: this is an advisory alarm, and a summary predating the
    # row count should lose the alarm rather than the whole comparison.
    cand_n, inc_n = cand_slice.get("n"), inc_slice.get("n")
    if cand_n is None or inc_n is None or cand_n == inc_n:
        return None
    cand_n, inc_n = int(cand_n), int(inc_n)
    return (
        f"the {GATE_MISSION} slice holds {cand_n} rows for the candidate and {inc_n} for the "
        f"incumbent ({cand_n - inc_n:+d}) — the gating comparison is not row-for-row"
    )


def _reproducibility_warning(candidate: dict[str, Any]) -> str | None:
    """Whether the candidate's numbers can be traced to a known code state.

    Alarmed, not blocking: a deliberate promotion from a working tree is the
    operator's call. But promoting a run whose code state contradicts its
    recorded commit means the served model cannot be rebuilt from the
    repository, and nothing downstream would ever surface that.

    Only a *recorded* claim is checked. `git_dirty` absent means the summary
    predates the field; `git_dirty: None` means provenance ran and git could not
    be reached, which is a different statement from a clean tree and is the one
    place this distinction earns its keep.
    """
    config = candidate.get("run_config", {})
    if "git_dirty" not in config:
        # A summary written before the field existed cannot be held to it, and
        # alarming on every legacy artefact is how an alarm list becomes noise
        # that gets skimmed. Absence is quiet; a recorded claim is checked.
        return None
    if config["git_dirty"] is None:
        return (
            "git was unreachable when this run was recorded, so whether it was trained "
            "from a clean tree is unknown"
        )
    if config["git_dirty"]:
        return (
            f"trained from a dirty working tree at {config.get('git_sha', 'an unknown commit')} "
            "— the recorded commit does not describe the code that ran"
        )
    return None


def _diagnostics(
    candidate: dict[str, Any], incumbent: dict[str, Any], alarm: float
) -> tuple[list[str], list[str]]:
    """Report every non-gating slice; alarm on any AUC drop beyond `alarm`."""
    reasons, alarms = [], []
    for mission in (*DIAGNOSTIC_MISSIONS, AGGREGATE_SLICE):
        cand = candidate.get("per_mission", {}).get(mission)
        inc = incumbent.get("per_mission", {}).get(mission)
        if cand is None or inc is None:
            continue
        delta = cand["roc_auc"] - inc["roc_auc"]
        label = "reported, never gates" if mission == AGGREGATE_SLICE else "diagnostic"
        reasons.append(
            f"{mission} AUC {cand['roc_auc']:.4f} vs {inc['roc_auc']:.4f} ({delta:+.4f}, {label})"
        )
        if mission != AGGREGATE_SLICE and delta < -alarm:
            alarms.append(
                f"{mission} AUC fell {delta:+.4f} (>{alarm}) — this does not block "
                "promotion but requires a written explanation in the roadmap first"
            )
    return reasons, alarms


def evaluate_promotion(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
    *,
    brier_tolerance: float = 0.005,
    ece_tolerance: float = 0.01,
    recall_tolerance: float | None = None,
    mission_alarm: float = 0.02,
    allow_unmatched_populations: bool = False,
) -> PromotionDecision:
    """Compare a candidate cv_summary against the incumbent's.

    Gates on `GATE_MISSION` when both summaries carry a `per_mission` block.
    Without it there is no way to check the two means cover the same rows, so
    the fallback refuses rather than comparing populations that may differ —
    see `_population_mismatch`. `allow_unmatched_populations` overrides that
    for a deliberate cross-population read.

    `recall_tolerance` defaults to **this run's own measured floor** rather than
    a constant — see `DecisionFloor`. Passing a number overrides it, which is
    what the pre-stage-6 tests do deliberately.
    """
    if incumbent is None:
        return PromotionDecision(
            Verdict.PROMOTE, ["first registered model — becomes the incumbent"]
        )

    cand_slice, inc_slice = _gate_slice(candidate), _gate_slice(incumbent)
    cand_recall: float | None = None
    inc_recall: float | None = None
    cand_ece: float | None
    inc_ece: float | None
    if cand_slice is not None and inc_slice is not None:
        cand_auc, inc_auc = cand_slice["roc_auc"], inc_slice["roc_auc"]
        cand_brier, inc_brier = cand_slice["brier"], inc_slice["brier"]
        cand_ece, inc_ece = cand_slice["ece"], inc_slice["ece"]
        cand_recall, inc_recall = cand_slice["recall_at_1pct_fpr"], inc_slice["recall_at_1pct_fpr"]
        header = f"gated on {GATE_MISSION} (n={cand_slice['n']:.0f}), the deployment population"
    else:
        cand_auc, inc_auc = _mean(candidate, "test_roc_auc"), _mean(incumbent, "test_roc_auc")
        cand_brier, inc_brier = _mean(candidate, "test_brier"), _mean(incumbent, "test_brier")
        cand_ece, inc_ece = (
            _mean_or_none(candidate, "test_ece"),
            _mean_or_none(incumbent, "test_ece"),
        )
        header = "gated on pooled CV means — a summary here predates the per_mission block"

    floor = decision_floor(candidate, incumbent)
    if recall_tolerance is None:
        recall_tolerance = floor.recall if floor.recall is not None else LEGACY_RECALL_TOLERANCE

    mismatch = _population_mismatch(candidate, incumbent)
    paired = paired_folds(candidate, incumbent)
    gate_size = _gate_population_drift(cand_slice, inc_slice)
    reasons = [
        header,
        f"ROC-AUC {cand_auc:.4f} vs incumbent {inc_auc:.4f}",
        f"Brier {cand_brier:.4f} vs incumbent {inc_brier:.4f}",
    ]
    diagnostics, alarms = _diagnostics(candidate, incumbent, mission_alarm)
    unresolved: list[str] = []
    reasons += diagnostics
    if gate_size:
        alarms.append(gate_size)
    if provenance := _reproducibility_warning(candidate):
        alarms.append(provenance)

    # Every comparison below is an inequality, and NaN loses all of them — so a
    # degenerate run (a single-class fold, an empty slice, a blown-up loss)
    # would sail through every guard and promote itself. Checked first, on both
    # sides: a NaN incumbent metric is a corrupt registry, and promoting past it
    # unnoticed is the same failure.
    unmeasured = [
        f"{name} ({side})"
        for name, cand, inc in (
            ("ROC-AUC", cand_auc, inc_auc),
            ("Brier", cand_brier, inc_brier),
            ("ECE", cand_ece, inc_ece),
            ("recall @1% FPR", cand_recall, inc_recall),
        )
        for side, value in (("candidate", cand), ("incumbent", inc))
        if value is not None and not math.isfinite(value)
    ]
    if unmeasured:
        reasons.append(f"not measurable: {', '.join(unmeasured)}")
        return PromotionDecision(Verdict.REJECT, reasons, alarms)

    # Gating on the deployment slice compares one mission the two runs both
    # scored, so a mission only one of them has is worth reporting but does not
    # invalidate the comparison. The pooled fallback has no such footing: it
    # compares each run's mean over its own rows, and this project has been
    # doing exactly that against an incumbent whose build predates K2.
    if mismatch:
        gated_on_slice = cand_slice is not None and inc_slice is not None
        reasons.append(f"populations differ: {mismatch}")
        if gated_on_slice:
            alarms.append(
                f"populations differ ({mismatch}) — the pooled slice is not like-for-like"
            )
        elif not allow_unmatched_populations:
            reasons.append(
                "refusing to promote on a pooled comparison whose rows cannot be "
                "matched — re-baseline the incumbent on the current view set, or "
                "pass allow_unmatched_populations=True to read it anyway"
            )
            return PromotionDecision(Verdict.REJECT, reasons, alarms)

    if cand_auc <= inc_auc:
        # A tie and a loss are not the same statement, and saying "does not beat"
        # for a delta of 0.0001 asserts a measurement that does not exist. The
        # gate's answer is identical either way — the incumbent keeps serving,
        # because churning a deployed model on noise is not an upgrade — but the
        # *record* has to distinguish them, or a level result reads for ever
        # after as a defeat. This is the direction of the asymmetry that matters:
        # every other criterion below grants the candidate a tolerance band, so
        # without this the incumbent silently wins every tie it is handed.
        margin = inc_auc - cand_auc
        if floor.auc is not None and margin <= floor.auc:
            reasons.append(
                f"level on ROC-AUC — the {margin:.4f} gap is inside this run's own "
                f"{floor.auc:.4f} floor ({floor.source}), so it is not a measured "
                "difference. The incumbent stays served because a tie is not a reason "
                "to replace a deployed model, not because the candidate lost"
            )
        else:
            reasons.append(
                f"does not beat the incumbent's CV score (-{margin:.4f}, "
                f"{floor.times(margin, floor.auc)})"
            )
        return PromotionDecision(Verdict.REJECT, reasons, alarms)
    if cand_brier > inc_brier + brier_tolerance:
        reasons.append(f"calibration degraded beyond tolerance (+{brier_tolerance})")
        return PromotionDecision(Verdict.REJECT, reasons, alarms)

    if cand_ece is not None and inc_ece is not None:
        reasons.append(f"ECE {cand_ece:.4f} vs incumbent {inc_ece:.4f}")
        if cand_ece > inc_ece + ece_tolerance:
            reasons.append(f"reliability degraded beyond tolerance (+{ece_tolerance})")
            return PromotionDecision(Verdict.REJECT, reasons, alarms)
    else:
        reasons.append("ECE guard skipped — summary predates the test_ece field")

    if cand_recall is not None and inc_recall is not None:
        margin = cand_recall - inc_recall
        measured = floor.recall is not None
        reasons.append(
            f"recall @1% FPR {cand_recall:.3f} vs incumbent {inc_recall:.3f} "
            f"({margin:+.4f}, {floor.times(margin, floor.recall)}; "
            f"tolerance {recall_tolerance:.4f}"
            + ("" if measured else ", legacy constant — this run reported no variance block")
            + ")"
        )
        # Order matters and is pre-registered: a margin 1.02x its own floor is
        # not a confident rejection, so the band is tested BEFORE the threshold.
        # Only a margin clear of the band on the wrong side rejects.
        if margin < -recall_tolerance and not unresolved_against(margin, recall_tolerance):
            reasons.append(f"shortlist recall degraded beyond tolerance (-{recall_tolerance:.4f})")
            return PromotionDecision(Verdict.REJECT, reasons, alarms)
        if unresolved_against(margin, recall_tolerance):
            # Not a rejection and not a promotion: the margin and the floor it is
            # read against are the same size, and the floor is three draws wide.
            unresolved.append(
                f"shortlist recall margin {margin:+.4f} is within {UNRESOLVED_BAND}x of its "
                f"{recall_tolerance:.4f} floor — too close to call from three draws"
            )
    else:
        reasons.append("recall guard skipped — summary predates the per_mission block")

    if paired is not None:
        reasons.append(str(paired))
        # The mean can clear the incumbent while the candidate loses most folds,
        # which is what winning on training noise looks like. Alarmed rather than
        # blocking: at five folds the signed-rank test cannot reach p<0.05, so
        # refusing on it would refuse every real improvement too.
        if paired.wins * 2 <= len(paired.deltas):
            alarms.append(
                f"the candidate won only {paired.wins}/{len(paired.deltas)} folds head to head "
                "— the mean improvement is not reproduced fold by fold"
            )
        elif abs(paired.effect_size) < 1.0:
            alarms.append(
                f"paired effect size d={paired.effect_size:+.2f} is inside the fold-to-fold "
                "spread — repeat the run before reading this as an improvement"
            )

    if unresolved_against(cand_auc - inc_auc, floor.auc):
        unresolved.append(
            f"gate AUC margin {cand_auc - inc_auc:+.4f} is within {UNRESOLVED_BAND}x of its "
            f"{floor.auc:.4f} floor — too close to call from three draws"
        )

    if unresolved:
        # REJECT dominates UNRESOLVED dominates PROMOTE. Reaching here means no
        # criterion rejected, so the only question left is whether any of them
        # could be resolved at all.
        reasons.extend(unresolved)
        reasons.append(
            "no criterion rejected, but the decision is not resolvable against a floor "
            "this wide — a stop-and-ask, per stage 6 caveat 2"
        )
        return PromotionDecision(Verdict.UNRESOLVED, reasons, alarms)

    reasons.append("beats incumbent with calibration and shortlist recall intact")
    return PromotionDecision(Verdict.PROMOTE, reasons, alarms)


# ------------------------------------------------------------------ registry --


def load_registry(models_dir: Path) -> dict[str, Any] | None:
    path = models_dir / REGISTRY_NAME
    return json.loads(path.read_text()) if path.exists() else None


def load_incumbent_summary(
    models_dir: Path, summary_path: Path | None = None
) -> dict[str, Any] | None:
    """The incumbent's summary — the registry's, or an explicit re-baseline.

    **`summary_path` is the other half of stage 3.** The 2026-08-07 audit found
    that the registry's `ca906040` summary carries no `per_mission` block, so
    `_gate_slice` returns None, `_population_mismatch` fires and every candidate
    is refused *before a single metric is compared*. Stage 3 produced the
    re-baselined summary that fixes it — `models/cv/incumbent-rebaselined/` —
    and the audit's own note said `promotion_gate.py` had no flag to point at
    one. That note was never actioned: the roadmap recorded stage 3 as "the gate
    returns decisions again instead of refusing" while the gate went on refusing
    everything, and every branch rejection since was decided by hand.

    Passing a path here rather than editing `models/registry.json` is deliberate.
    The registry names what is *served*; the re-baseline is a measurement of that
    same model on the current view set. Writing the latter into the former would
    make the serving pointer depend on an evaluation artefact, and touching the
    registry at all is a stop-and-ask in this project.
    """
    if summary_path is not None:
        return json.loads(summary_path.read_text())
    registry = load_registry(models_dir)
    if registry is None:
        return None
    # Registry paths are repo-root-relative ("models/cv/<run>/..."); resolve
    # against models_dir's parent, not the caller's cwd, so
    # `promotion_gate.py --models-dir` works from anywhere.
    path = Path(registry["cv_summary"])
    if not path.is_absolute():
        path = models_dir.parent / path
    return json.loads(path.read_text())


def publishable_cv_dirs(models_dir: Path) -> list[Path]:
    """CV run dirs the publish step may `dvc add`: the registry's promoted run
    plus dirs that already carry a `.dvc` pointer. An allowlist, not a glob —
    a tuning campaign litters `models/cv/` with cheap trial checkpoints that
    also have a `cv_summary.json` and must never be pushed to the remote.
    """
    registry = load_registry(models_dir)
    promoted = registry["run_id"] if registry else None
    cv_root = models_dir / "cv"
    if not cv_root.is_dir():
        return []
    dirs = []
    for run_dir in sorted(cv_root.iterdir()):
        if not (run_dir / "cv_summary.json").is_file():
            continue
        pointer = cv_root / f"{run_dir.name}.dvc"
        if run_dir.name == promoted or pointer.exists():
            dirs.append(run_dir)
    return dirs


def assert_servable(cv_dir: Path) -> None:
    """Every fold must hold weights and a calibrator before the registry names it.

    Stage 4 run 1 wrote no checkpoints at all — a metrics-only run that would
    promote cleanly here and fail at serve time, long after the decision. The
    registry is a pointer to something that has to exist.
    """
    folds = sorted(cv_dir.glob("fold_*"))
    if not folds:
        raise ValueError(f"{cv_dir} holds no fold_* directories — nothing to serve")
    incomplete = [
        fold.name
        for fold in folds
        if not any(fold.glob("*.keras")) or not any(fold.glob("*calibrator*.joblib"))
    ]
    if incomplete:
        raise ValueError(
            f"{cv_dir}: {incomplete} lack weights or a calibrator bundle — "
            "this run cannot be served and must not be promoted"
        )


def promote(models_dir: Path, run_id: str, cv_summary_path: Path) -> dict[str, Any]:
    """Point the registry at a new best run. Caller decides via evaluate_promotion."""
    assert_servable(cv_summary_path.parent)
    summary = json.loads(cv_summary_path.read_text())
    registry = {
        "run_id": run_id,
        "cv_summary": str(cv_summary_path),
        "cv_dir": str(cv_summary_path.parent),
        "test_roc_auc_mean": _mean(summary, "test_roc_auc"),
        "test_brier_mean": _mean(summary, "test_brier"),
        "promoted_at": datetime.now(UTC).isoformat(),
    }
    ece_mean = _mean_or_none(summary, "test_ece")
    if ece_mean is not None:
        registry["test_ece_mean"] = ece_mean
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / REGISTRY_NAME).write_text(json.dumps(registry, indent=2) + "\n")
    return registry
