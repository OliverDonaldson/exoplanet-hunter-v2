"""Tests for the GPU-burst trigger ("dataset changed materially")."""

import pandas as pd

from exoplanet_hunter.validation.trigger import evaluate_refresh


def catalogue(n: int, start: int = 1, label: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mission": ["TESS"] * n,
            "tic_id": range(start, start + n),
            "label": [label] * n,
            "disposition": ["CP" if label == 1 else "FP"] * n,
        }
    )


def test_small_refresh_skips_training():
    old = catalogue(100)
    new = pd.concat([old, catalogue(10, start=1000)], ignore_index=True)
    decision = evaluate_refresh(old, new, min_new_labelled=25)
    assert not decision.should_train
    assert decision.n_new_confirmed == 10
    assert "not worth a GPU burst" in str(decision)


def test_material_refresh_trains():
    old = catalogue(100)
    new = pd.concat(
        [old, catalogue(20, start=1000), catalogue(10, start=2000, label=0)],
        ignore_index=True,
    )
    decision = evaluate_refresh(old, new, min_new_labelled=25)
    assert decision.should_train
    assert decision.n_new_confirmed == 20
    assert decision.n_new_false_pos == 10


def test_force_trains_regardless():
    old = catalogue(100)
    decision = evaluate_refresh(old, old.copy(), force=True)
    assert decision.should_train
    assert "expansion run" in str(decision)


def test_flips_are_counted_but_do_not_trigger():
    old = catalogue(100)
    new = old.copy()
    new.loc[:39, "label"] = 0  # 40 flips — quarantine material, not training data
    decision = evaluate_refresh(old, new, min_new_labelled=25)
    assert not decision.should_train
    assert decision.n_flips == 40
    assert decision.n_new_targets == 0
