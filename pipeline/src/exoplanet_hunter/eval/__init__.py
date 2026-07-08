"""Evaluation: classification/calibration metrics and per-candidate vetting figures.

The V1 attention-diagnostic module (a report-era research artefact) is not
ported; it stays available in V1 history if ever needed.
"""

from exoplanet_hunter.eval.metrics import (
    ClassificationMetrics,
    calibration_curve,
    classification_metrics,
)
from exoplanet_hunter.eval.vetting import CandidateReport, vetting_figure

__all__ = [
    "CandidateReport",
    "ClassificationMetrics",
    "calibration_curve",
    "classification_metrics",
    "vetting_figure",
]
