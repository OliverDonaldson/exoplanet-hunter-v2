"""Tests for `eval.scoring` — the run-scoring layer and its protocol invariant.

`eval/scoring.py` states in its own module docstring that mixing out-of-fold and
zero-shot scores across one population is a comparability defect. That invariant
was enforced nowhere until 2026-08-08, which is the class of defect this project
keeps finding: a stated rule with no code behind it.

`score_run` — the model-loading half — was itself untested until later the same
day, while `summarise_scored` had coverage. Its tests are at the bottom.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from exoplanet_hunter.datasets.tfrecords import ShardMetadata, load_index, write_tfrecord_shards
from exoplanet_hunter.datasets.views_io import ViewArrays
from exoplanet_hunter.eval.scoring import (
    Protocol,
    legacy_aux,
    read_views,
    score_run,
    summarise_scored,
)
from exoplanet_hunter.training.calibration import PlattScaler


def scored(**overrides) -> pd.DataFrame:
    """A scored frame: TESS and Kepler out-of-fold, K2 zero-shot."""
    rows = {
        "tic_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "label": [1, 0, 1, 0, 1, 0, 1, 0],
        "score": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
        "mission": ["TESS"] * 4 + ["Kepler"] * 2 + ["K2"] * 2,
        "protocol": [Protocol.OUT_OF_FOLD] * 6 + [Protocol.ZERO_SHOT] * 2,
    }
    return pd.DataFrame({**rows, **overrides})


def test_slices_are_out_of_fold_only_and_zero_shot_is_reported_apart():
    result = summarise_scored(scored(), source="test")
    assert set(result["per_mission"]) == {"TESS", "Kepler", "all"}
    assert result["per_mission"]["all"]["n"] == 6
    assert set(result["zero_shot"]) == {"K2"}
    assert result["zero_shot"]["K2"]["n"] == 2


def test_a_zero_shot_block_of_pure_negatives_cannot_reach_the_gate():
    """The live case, measured 2026-08-08: the re-baselined champion carries
    243 zero-shot Kepler rows with no positives at all. Pooling them into the
    out-of-fold Kepler slice measures a population no model was asked about."""
    frame = scored()
    frame.loc[frame["mission"] == "Kepler", "protocol"] = Protocol.ZERO_SHOT
    frame.loc[frame["mission"] == "Kepler", "label"] = 0

    result = summarise_scored(frame, source="test")
    assert "Kepler" not in result["per_mission"]
    assert result["zero_shot"]["Kepler"]["n_positive"] == 0
    assert result["per_mission"]["all"]["n"] == 4


def test_an_unresolved_mission_is_refused_rather_than_dropped():
    frame = scored()
    frame.loc[0, "mission"] = None
    with pytest.raises(ValueError, match="carry no mission"):
        summarise_scored(frame, source="test")


def test_unresolved_rows_can_be_excluded_but_are_recorded():
    frame = scored()
    frame.loc[0, "mission"] = None
    result = summarise_scored(frame, source="test", exclude_unresolved=True)
    assert result["provenance"]["excluded_unresolved"] == [1]
    assert result["per_mission"]["all"]["n"] == 5


def test_the_aggregate_must_equal_the_missions_that_make_it_up():
    result = summarise_scored(scored(), source="test")
    per_mission = result["per_mission"]
    counted = sum(v["n"] for k, v in per_mission.items() if k != "all")
    assert counted == per_mission["all"]["n"]


def test_a_frame_with_no_held_out_rows_cannot_gate():
    frame = scored()
    frame["protocol"] = Protocol.ZERO_SHOT
    with pytest.raises(ValueError, match="not a held-out measurement"):
        summarise_scored(frame, source="test")


def test_a_frame_without_the_gate_mission_is_refused():
    frame = scored()
    frame = frame[frame["mission"] != "TESS"]
    with pytest.raises(ValueError, match="cannot gate"):
        summarise_scored(frame, source="test")


def test_a_missing_column_names_itself():
    with pytest.raises(ValueError, match=r"\['protocol'\]"):
        summarise_scored(scored().drop(columns=["protocol"]), source="test")


def test_the_summary_block_the_pooled_fallback_reads_is_present():
    result = summarise_scored(scored(), source="test")
    assert set(result["summary"]) == {"test_roc_auc", "test_brier", "test_ece", "variance"}
    assert result["summary"]["test_roc_auc"]["mean"] == result["per_mission"]["all"]["roc_auc"]


def ensembled(n_tess: int = 60, n_members: int = 3) -> pd.DataFrame:
    """An out-of-fold TESS slice scored by several members that disagree.

    Wide enough that a 1% FPR cut lands on more than one negative: on a handful
    of rows every member returns the same recall, and a spread of exactly zero
    is read as no measurement rather than as a noiseless run.
    """
    rng = np.random.default_rng(0)
    label = np.tile([1, 0], n_tess // 2)
    score = np.where(label == 1, rng.uniform(0.35, 0.95, n_tess), rng.uniform(0.05, 0.65, n_tess))
    frame = pd.DataFrame(
        {
            "tic_id": np.arange(n_tess),
            "label": label,
            "score": score,
            "mission": "TESS",
            "protocol": Protocol.OUT_OF_FOLD,
        }
    )
    for member in range(n_members):
        frame[f"member_score_{member}"] = np.clip(score + rng.normal(0, 0.06, n_tess), 0, 1)
    return frame


def test_a_summary_rebuilt_from_member_scores_carries_the_recall_floor():
    """The point of rebuilding a summary without retraining: the gate sizes its
    recall tolerance from the run's own reseeding spread, and the per-member
    scores that measure it are already in the prediction set."""
    variance = summarise_scored(ensembled(), source="test")["summary"]["variance"]
    assert variance["n_models_per_fold"] == 3
    assert variance["pooled_gate_recall_n_draws"] == 3
    assert variance["pooled_gate_recall_seed_sd"] > 0.0


def test_the_floor_it_reports_is_one_the_gate_can_size_a_tolerance_from():
    from exoplanet_hunter.validation import decision_floor

    assert decision_floor(summarise_scored(ensembled(), source="test")).recall is not None


def test_a_single_model_run_reports_the_block_and_measures_nothing_in_it():
    """The block is emitted either way. One that appeared only when the
    measurement succeeded would make a missing key and a null read the same to a
    person and differently to a program — and a run with no member columns must
    still leave the gate exactly where it was before this existed."""
    from exoplanet_hunter.validation import decision_floor

    result = summarise_scored(scored(), source="test")
    variance = result["summary"]["variance"]
    assert variance["n_models_per_fold"] == 0
    assert variance["pooled_gate_recall_seed_sd"] is None
    floor = decision_floor(result)
    assert floor.recall is None and floor.auc is None
    assert "no variance block" in floor.source


def test_no_folds_block_so_pairing_reports_nothing_rather_than_something_wrong():
    """The champion's folds are a different partition from any candidate's, so
    pairing on fold index would compare fold k of one split against fold k of
    another. `paired_folds` returns None on a missing block."""
    from exoplanet_hunter.validation.promotion import paired_folds

    result = summarise_scored(scored(), source="test")
    assert "folds" not in result
    assert paired_folds({"folds": [{"test_roc_auc": 0.9}]}, result) is None


# ------------------------------------------------------------- score_run --
#
# Every guard below stands between a plausible number and a wrong one, which is
# the only reason any of them exists.

GLOBAL_BINS, LOCAL_BINS, AUX_DIM = 64, 16, 13
N_ROWS = 12


def views_and_labels(n: int = N_ROWS, *, seed: int = 0):
    """A legacy dual-view shard set plus the catalogue rows it joins against."""
    rng = np.random.default_rng(seed)
    tic_ids = np.arange(1, n + 1, dtype=np.int64)
    views = ViewArrays(
        global_views=rng.normal(size=(n, GLOBAL_BINS)).astype(np.float32),
        local_views=rng.normal(size=(n, LOCAL_BINS)).astype(np.float32),
        labels=np.array([1, 0] * (n // 2), dtype=np.int8),
        tic_ids=tic_ids,
        aux_features=rng.normal(size=(n, AUX_DIM)).astype(np.float32),
    )
    labels = pd.DataFrame({"tic_id": tic_ids, "snr": rng.uniform(5.0, 50.0, n)})
    return views, labels


def constant_model(probability: float):
    """A dual-view model that ignores its inputs and emits one value.

    Constant on purpose: it makes the *routing* observable. Which fold scored a
    row is otherwise invisible in the output, and routing is what `OUT_OF_FOLD`
    has to get right.
    """
    import tensorflow as tf

    g = tf.keras.Input(shape=(GLOBAL_BINS,), name="global_view")
    lo = tf.keras.Input(shape=(LOCAL_BINS,), name="local_view")
    aux = tf.keras.Input(shape=(9,), name="aux_features")
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="prediction")(
        tf.keras.layers.Concatenate()([g, lo, aux])
    )
    model = tf.keras.Model([g, lo, aux], out)
    dense = model.get_layer("prediction")
    kernel, _ = dense.get_weights()
    logit = float(np.log(probability / (1.0 - probability)))
    dense.set_weights([np.zeros_like(kernel), np.array([logit], dtype=np.float32)])
    return model


def write_run(run_dir, aux, fold_probabilities: dict[int, float]):
    """A run directory: one checkpoint and calibrator bundle per fold."""
    for fold, probability in fold_probabilities.items():
        fold_dir = run_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        constant_model(probability).save(fold_dir / "cnn_dualview.keras")
        pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(aux)
        joblib.dump(
            {"aux_pipeline": pipeline, "calibrator": PlattScaler(1.0, 0.0)},
            fold_dir / "cnn_calibrator.joblib",
        )
    return run_dir


@pytest.fixture
def scoring_fixture(tmp_path):
    """Shards, catalogue and a two-fold run, wired the way `score_run` expects."""
    views, labels = views_and_labels()
    shard_dir = tmp_path / "shards"
    write_tfrecord_shards(views, shard_dir, examples_per_shard=8)
    aux = legacy_aux(load_index(shard_dir), labels, AUX_DIM)
    run_dir = write_run(tmp_path / "run", aux, {0: 0.25, 1: 0.75})
    return shard_dir, run_dir, labels, views


def test_legacy_aux_rebuilds_index_seven_rather_than_slicing():
    """The 9-dim and 13-dim layouts disagree at index 7 — catalogue SNR there,
    `pink_snr` in the shards. Slicing 13 -> 9 feeds one lane into the other and
    returns a confident wrong number."""
    index = pd.DataFrame(
        {"tic_id": [1, 2], **{f"aux_{k}": [float(k), float(k) + 100] for k in range(AUX_DIM)}}
    )
    labels = pd.DataFrame({"tic_id": [1, 2], "snr": [11.0, 22.0]})

    aux = legacy_aux(index, labels, AUX_DIM)
    assert aux.shape == (2, 9)
    np.testing.assert_array_equal(aux[:, 7], [11.0, 22.0])
    assert aux[0, 7] != index.loc[0, "aux_7"], "index 7 was taken from the shard, not the catalogue"
    np.testing.assert_array_equal(aux[:, :7], index[[f"aux_{k}" for k in range(7)]].to_numpy())
    np.testing.assert_array_equal(aux[:, 8], index["aux_8"].to_numpy())


def test_legacy_aux_leaves_a_missing_catalogue_snr_as_nan():
    """All 527 K2 rows have no catalogue SNR. NaN lets the fold's own pipeline
    impute it, which is the path a non-TOI already takes at serve time; a zero
    would be a measurement."""
    index = pd.DataFrame(
        {"tic_id": [1, 2], **{f"aux_{k}": [float(k), float(k)] for k in range(AUX_DIM)}}
    )
    labels = pd.DataFrame({"tic_id": [1], "snr": [11.0]})
    aux = legacy_aux(index, labels, AUX_DIM)
    assert aux[0, 7] == 11.0
    assert np.isnan(aux[1, 7])


@pytest.mark.parametrize(
    ("protocol", "fold_of"),
    [(Protocol.OUT_OF_FOLD, None), (Protocol.ZERO_SHOT, pd.Series({1: 0}))],
)
def test_the_protocol_and_fold_map_must_agree(tmp_path, protocol, fold_of):
    """Supplying `fold_of` under ZERO_SHOT is refused rather than ignored: the
    caller clearly meant one thing and asked for the other."""
    with pytest.raises(ValueError, match="needs fold_of"):
        score_run(tmp_path, tmp_path, labels=pd.DataFrame(), protocol=protocol, fold_of=fold_of)


def test_a_run_with_no_folds_raises(tmp_path, scoring_fixture):
    shard_dir, _, labels, _ = scoring_fixture
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match=r"no fold_\* directories"):
        score_run(empty, shard_dir, labels=labels, protocol=Protocol.ZERO_SHOT)


def test_out_of_fold_scores_each_row_with_the_fold_that_held_it_out(scoring_fixture):
    """The protocol's whole content. Each fold's model emits its own constant,
    so the returned score names the fold that produced it."""
    shard_dir, run_dir, labels, views = scoring_fixture
    assignment = {int(t): int(i % 2) for i, t in enumerate(views.tic_ids)}
    fold_of = pd.Series(assignment)

    result = score_run(
        run_dir, shard_dir, labels=labels, protocol=Protocol.OUT_OF_FOLD, fold_of=fold_of
    )
    assert result.protocol is Protocol.OUT_OF_FOLD
    assert len(result.predictions) == len(views.tic_ids)
    expected = {0: 0.25, 1: 0.75}
    for row in result.predictions.itertuples():
        assert row.fold == assignment[row.tic_id]
        assert row.score == pytest.approx(expected[row.fold], abs=1e-4)


def test_a_row_whose_fold_has_no_checkpoint_raises_rather_than_scoring(scoring_fixture):
    """`fold_of` comes from predictions.parquet and `folds` from globbing the
    directory; nothing ties them together. Uninitialised memory returned as a
    calibrated probability is the failure this replaced."""
    shard_dir, run_dir, labels, views = scoring_fixture
    fold_of = pd.Series({int(t): 7 for t in views.tic_ids})  # no fold_7 on disk
    with pytest.raises(RuntimeError, match="no checkpoint"):
        score_run(run_dir, shard_dir, labels=labels, protocol=Protocol.OUT_OF_FOLD, fold_of=fold_of)


def test_zero_shot_refuses_a_run_whose_trained_rows_are_unknown(scoring_fixture):
    """Averaging every fold over a row one of them trained on is not a held-out
    measurement, and nothing downstream could tell."""
    shard_dir, run_dir, labels, _ = scoring_fixture
    with pytest.raises(ValueError, match="contamination filter cannot run"):
        score_run(run_dir, shard_dir, labels=labels, protocol=Protocol.ZERO_SHOT)


def test_zero_shot_proceeds_when_the_caller_takes_responsibility(scoring_fixture):
    shard_dir, run_dir, labels, views = scoring_fixture
    result = score_run(
        run_dir,
        shard_dir,
        labels=labels,
        protocol=Protocol.ZERO_SHOT,
        allow_untracked_rows=True,
    )
    assert result.protocol is Protocol.ZERO_SHOT
    assert len(result.predictions) == len(views.tic_ids)
    # The mean of the two folds' constants, and the fold column says "no fold".
    assert result.predictions["score"].to_numpy() == pytest.approx(0.5, abs=1e-4)
    assert (result.predictions["fold"] == -1).all()


def test_zero_shot_drops_the_rows_the_run_trained_on(scoring_fixture):
    """The contamination filter, from the run's own predictions.parquet."""
    shard_dir, run_dir, labels, views = scoring_fixture
    trained_on = views.tic_ids[:5]
    pd.DataFrame({"tic_id": trained_on, "score": 0.5}).to_parquet(run_dir / "predictions.parquet")

    result = score_run(run_dir, shard_dir, labels=labels, protocol=Protocol.ZERO_SHOT)
    assert not set(result.predictions["tic_id"]) & set(trained_on.tolist())
    assert len(result.predictions) == len(views.tic_ids) - len(trained_on)


def test_read_views_refuses_a_stream_that_does_not_match_the_index(scoring_fixture):
    """Scores are joined back positionally, so a reordered stream would attach
    every probability to the wrong star."""
    shard_dir, _, _, _ = scoring_fixture
    index = load_index(shard_dir)
    metadata = ShardMetadata.load(shard_dir)
    read_views(shard_dir, metadata, index)  # the honest ordering is fine

    with pytest.raises(RuntimeError, match="does not match the index"):
        read_views(shard_dir, metadata, index.iloc[::-1].reset_index(drop=True))
