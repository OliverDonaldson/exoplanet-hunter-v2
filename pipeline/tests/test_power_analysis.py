"""P2.1's harness — that it measures what the console measures.

The power analysis compares candidate metrics against `recall @1% FPR`, which
is also the statistic the Model page serves. If the two definitions drift, the
experiment file's numbers stop being comparable with the console's and nobody
finds out, so the equivalence is pinned here rather than assumed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


power = _load(ROOT / "pipeline" / "scripts" / "power_analysis.py", "_power_analysis")


@pytest.fixture
def served_recall():
    """The API's own implementation, imported rather than copied."""
    sys.path.insert(0, str(ROOT / "api"))
    from app.routes.model import _recall_at_fpr

    return _recall_at_fpr


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_recall_matches_the_served_definition(served_recall, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 400)
    p = rng.random(400)
    assert power.recall_at_fpr(y, p, 0.01) == pytest.approx(served_recall(y, p, 0.01))


def test_recall_is_nan_when_a_class_is_absent():
    """The served version returns None; this one returns NaN so it can sit in a
    numeric table. Both refuse to answer, which is the property that matters."""
    y = np.ones(20, dtype=int)
    assert np.isnan(power.recall_at_fpr(y, np.random.default_rng(0).random(20)))


def test_degradation_is_identity_at_zero():
    """w = 0 must be the served ranking, or every z is measured against the
    wrong baseline."""
    p = np.random.default_rng(0).random(200)
    from scipy.stats import rankdata

    assert np.array_equal(
        rankdata(power._degrade(p, 0.0, 1.0, seed=1)),
        rankdata(p),
    )


@pytest.mark.parametrize("w", [0.05, 0.2, 0.5])
def test_degradation_is_monotone_in_w(w):
    """A larger w has to mean a worse ranking, or the power curve is not a curve."""
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 800)
    p = rng.random(800) + y * 0.7
    from sklearn.metrics import roc_auc_score

    small = roc_auc_score(y, power._degrade(p, w, 1.0, seed=5))
    large = roc_auc_score(y, power._degrade(p, w * 3, 1.0, seed=5))
    assert large <= small


def test_only_the_named_top_fraction_moves():
    """A degradation confined to the top must leave the tail's order alone,
    or the localised result is measuring a global change."""
    from scipy.stats import rankdata

    p = np.random.default_rng(0).random(1000)
    out = power._degrade(p, 0.3, 0.05, seed=2)
    pct = rankdata(p) / len(p)
    tail = pct < 0.95
    assert np.array_equal(rankdata(out[tail]), rankdata(p[tail]))


def test_unpaired_arms_raise_rather_than_report(tmp_path):
    """A contrast over arms that share no rows is not a contrast. Guards raise."""
    import pandas as pd

    cols = {
        "tic_id": [1, 2],
        "mission": ["TESS"] * 2,
        "label": [0, 1],
        "dv_usable": [True, True],
        "score": [0.1, 0.9],
        **{f"member_score_{j}": [0.1, 0.9] for j in range(3)},
    }
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    pd.DataFrame(cols).to_parquet(a)
    pd.DataFrame({**cols, "tic_id": [3, 4]}).to_parquet(b)
    with pytest.raises(ValueError, match="not paired"):
        power.load_arms(a, b)


def test_pooled_sd_recovers_a_known_variance():
    """Variance-pooling equal-sized groups is the root mean of their variances."""
    import pandas as pd

    census = pd.DataFrame({"members": [3, 3], "ROC-AUC": [0.02, 0.04]})
    sd, dof = power.pooled_sd(census, "ROC-AUC")
    assert dof == 4
    assert sd == pytest.approx(np.sqrt((0.02**2 + 0.04**2) / 2))


def test_pooled_sd_weights_by_degrees_of_freedom():
    """A run with more members carries more of the pooled estimate."""
    import pandas as pd

    heavy = pd.DataFrame({"members": [11, 3], "ROC-AUC": [0.01, 0.10]})
    sd, dof = power.pooled_sd(heavy, "ROC-AUC")
    assert dof == 12
    assert sd < np.sqrt((0.01**2 + 0.10**2) / 2)


def test_contrast_mde_carries_two_seed_terms():
    """The published P2.1 figure used one; a difference of two means needs two."""
    one_term = power.Z_80_POWER * np.hypot(0.0021, 0.0093 / np.sqrt(5))
    two_term = power.contrast_mde(0.0093, 0.0021, 5)
    assert two_term > one_term
    assert two_term == pytest.approx(0.0175, abs=5e-4)


def test_contrast_mde_falls_as_members_rise():
    assert power.contrast_mde(0.01, 0.002, 10) < power.contrast_mde(0.01, 0.002, 5)


def test_architecture_is_read_not_guessed(tmp_path):
    """Two architectures in one run directory is unreadable, so it raises."""
    fold = tmp_path / "fold_0"
    fold.mkdir(parents=True)
    with pytest.raises(ValueError, match="expected exactly one architecture"):
        power.architecture_of(tmp_path)
    (fold / "model_0_cnn_dualview.keras").touch()
    assert power.architecture_of(tmp_path) == "cnn_dualview"
    (fold / "model_0_cnn_branches.keras").touch()
    with pytest.raises(ValueError, match="expected exactly one architecture"):
        power.architecture_of(tmp_path)
