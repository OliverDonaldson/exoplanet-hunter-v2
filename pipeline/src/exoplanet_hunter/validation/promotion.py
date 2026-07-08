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
  * the first-ever run promotes automatically (there is no bar yet — the RF
    baseline becomes the incumbent as soon as it's registered).

The registry is a plain JSON pointer, not MLflow state: the serving path and
the CI gate both read it without a tracking-server dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REGISTRY_NAME = "registry.json"


@dataclass
class PromotionDecision:
    promoted: bool
    reasons: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        verdict = "PROMOTE" if self.promoted else "REJECT"
        return f"{verdict}: " + "; ".join(self.reasons)


def _mean(summary: dict[str, Any], metric: str) -> float:
    return float(summary["summary"][metric]["mean"])


def evaluate_promotion(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
    *,
    brier_tolerance: float = 0.005,
) -> PromotionDecision:
    """Compare a candidate cv_summary against the incumbent's."""
    if incumbent is None:
        return PromotionDecision(True, ["first registered model — becomes the incumbent"])

    cand_auc = _mean(candidate, "test_roc_auc")
    inc_auc = _mean(incumbent, "test_roc_auc")
    cand_brier = _mean(candidate, "test_brier")
    inc_brier = _mean(incumbent, "test_brier")

    reasons = [
        f"ROC-AUC {cand_auc:.4f} vs incumbent {inc_auc:.4f}",
        f"Brier {cand_brier:.4f} vs incumbent {inc_brier:.4f}",
    ]
    if cand_auc <= inc_auc:
        reasons.append("does not beat the incumbent's CV score")
        return PromotionDecision(False, reasons)
    if cand_brier > inc_brier + brier_tolerance:
        reasons.append(f"calibration degraded beyond tolerance (+{brier_tolerance})")
        return PromotionDecision(False, reasons)
    reasons.append("beats incumbent with calibration intact")
    return PromotionDecision(True, reasons)


# ------------------------------------------------------------------ registry --


def load_registry(models_dir: Path) -> dict[str, Any] | None:
    path = models_dir / REGISTRY_NAME
    return json.loads(path.read_text()) if path.exists() else None


def load_incumbent_summary(models_dir: Path) -> dict[str, Any] | None:
    registry = load_registry(models_dir)
    if registry is None:
        return None
    return json.loads(Path(registry["cv_summary"]).read_text())


def promote(models_dir: Path, run_id: str, cv_summary_path: Path) -> dict[str, Any]:
    """Point the registry at a new best run. Caller decides via evaluate_promotion."""
    summary = json.loads(cv_summary_path.read_text())
    registry = {
        "run_id": run_id,
        "cv_summary": str(cv_summary_path),
        "cv_dir": str(cv_summary_path.parent),
        "test_roc_auc_mean": _mean(summary, "test_roc_auc"),
        "test_brier_mean": _mean(summary, "test_brier"),
        "promoted_at": datetime.now(UTC).isoformat(),
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / REGISTRY_NAME).write_text(json.dumps(registry, indent=2))
    return registry
