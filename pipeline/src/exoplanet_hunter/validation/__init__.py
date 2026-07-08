"""Validation gates: catalogue schemas, refresh leakage guard, promotion gate.

Built in `feat/validation-gates`. The gates run in three places: after a
catalogue refresh (schemas + leakage guard), after a views build (array
checks), and after a training run (promotion gate). CI exercises them via
unit tests; the orchestrator branch wires them into the live DAG.
"""

from exoplanet_hunter.validation.leakage import (
    assert_refresh_safe,
    diff_label_catalogues,
    quarantine_tics,
)
from exoplanet_hunter.validation.promotion import (
    PromotionDecision,
    evaluate_promotion,
    load_incumbent_summary,
    load_registry,
    promote,
)
from exoplanet_hunter.validation.schemas import (
    candidate_catalogue_schema,
    check_views,
    label_catalogue_schema,
)

__all__ = [
    "PromotionDecision",
    "assert_refresh_safe",
    "candidate_catalogue_schema",
    "check_views",
    "diff_label_catalogues",
    "evaluate_promotion",
    "label_catalogue_schema",
    "load_incumbent_summary",
    "load_registry",
    "promote",
    "quarantine_tics",
]
