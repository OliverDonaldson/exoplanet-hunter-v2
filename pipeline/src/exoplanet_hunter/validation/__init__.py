"""Validation gates: catalogue schemas, refresh leakage guard, promotion gate.

Built in `feat/validation-gates`. The gates run in three places: after a
catalogue refresh (schemas + leakage guard), after a views build (array
checks), and after a training run (promotion gate). CI exercises them via
unit tests; the orchestrator branch wires them into the live DAG.
"""

from exoplanet_hunter.validation.leakage import (
    assert_refresh_safe,
    diff_label_catalogues,
    drop_quarantined,
    load_quarantine,
    quarantine_tics,
    record_quarantine,
)
from exoplanet_hunter.validation.promotion import (
    ACKNOWLEDGED_ALARMS,
    PROMOTION_LOG_NAME,
    VERDICT_BY_EXIT_CODE,
    VERDICT_EXIT_CODES,
    PairedFolds,
    PromotionDecision,
    Verdict,
    decision_floor,
    evaluate_promotion,
    load_champion_summary,
    load_incumbent_summary,
    load_registry,
    paired_folds,
    promote,
    publishable_cv_dirs,
    read_decision,
    unacknowledged_alarms,
    write_decision,
    write_promotion_log,
)
from exoplanet_hunter.validation.schemas import (
    candidate_catalogue_schema,
    check_dv_archive,
    check_view_set,
    check_views,
    label_catalogue_schema,
)
from exoplanet_hunter.validation.shrink import check_catalogue_shrink
from exoplanet_hunter.validation.trigger import RefreshDecision, evaluate_refresh

__all__ = [
    "ACKNOWLEDGED_ALARMS",
    "PROMOTION_LOG_NAME",
    "VERDICT_BY_EXIT_CODE",
    "VERDICT_EXIT_CODES",
    "PairedFolds",
    "PromotionDecision",
    "RefreshDecision",
    "Verdict",
    "assert_refresh_safe",
    "candidate_catalogue_schema",
    "check_catalogue_shrink",
    "check_dv_archive",
    "check_view_set",
    "check_views",
    "decision_floor",
    "diff_label_catalogues",
    "drop_quarantined",
    "evaluate_promotion",
    "evaluate_refresh",
    "label_catalogue_schema",
    "load_champion_summary",
    "load_incumbent_summary",
    "load_quarantine",
    "load_registry",
    "paired_folds",
    "promote",
    "publishable_cv_dirs",
    "quarantine_tics",
    "read_decision",
    "record_quarantine",
    "unacknowledged_alarms",
    "write_decision",
    "write_promotion_log",
]
