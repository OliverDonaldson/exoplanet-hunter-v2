"""Tests for the TFRecord shard set and the tf.data input pipeline."""

import numpy as np
import pytest
import tensorflow as tf

from exoplanet_hunter.datasets import (
    AugmentConfig,
    ShardMetadata,
    Split,
    ViewArrays,
    augment_views,
    aux_constants_from_pipeline,
    fit_aux_pipeline,
    list_shards,
    load_index,
    make_dataset,
    make_split_table,
    tf_aux_transform,
    write_tfrecord_shards,
)

GLOBAL_BINS, LOCAL_BINS, AUX_DIM = 64, 16, 9


def synthetic_views(n: int = 30, seed: int = 0, with_aux: bool = True) -> ViewArrays:
    rng = np.random.default_rng(seed)
    aux = rng.normal(size=(n, AUX_DIM)).astype(np.float32)
    aux[:, 8] = np.abs(aux[:, 8]) * 100  # centroid_snr is non-negative, heavy-tailed
    aux[rng.random(size=aux.shape) < 0.1] = np.nan  # ExoFOP-style missingness
    return ViewArrays(
        global_views=rng.normal(size=(n, GLOBAL_BINS)).astype(np.float32),
        local_views=rng.normal(size=(n, LOCAL_BINS)).astype(np.float32),
        labels=(rng.random(n) < 0.4).astype(np.int8),
        # 3 rows per TIC — exercises the group-routing logic.
        tic_ids=np.repeat(np.arange(100, 100 + n // 3), 3).astype(np.int64),
        aux_features=aux if with_aux else None,
    )


@pytest.fixture(scope="module")
def shard_set(tmp_path_factory):
    views = synthetic_views()
    out = tmp_path_factory.mktemp("shards")
    metadata = write_tfrecord_shards(views, out, examples_per_shard=8)
    return views, out, metadata


def test_roundtrip_preserves_values_and_order(shard_set):
    views, out, metadata = shard_set
    assert metadata.n_shards == 4  # 30 examples / 8 per shard
    ds = make_dataset(list_shards(out), metadata, batch_size=64, cache=False)
    inputs, labels = next(iter(ds))
    np.testing.assert_allclose(
        inputs["global_view"].numpy().squeeze(-1), views.global_views, rtol=1e-6
    )
    np.testing.assert_allclose(labels.numpy(), views.labels.astype(np.float32))


def test_index_matches_views(shard_set):
    views, out, _ = shard_set
    index = load_index(out)
    np.testing.assert_array_equal(index["tic_id"].to_numpy(), views.tic_ids)
    np.testing.assert_array_equal(index["label"].to_numpy(), views.labels)
    aux_cols = [c for c in index.columns if c.startswith("aux_")]
    np.testing.assert_allclose(
        index[aux_cols].to_numpy(dtype=np.float32), views.aux_features, rtol=1e-6
    )


def test_split_filter_routes_by_tic_and_preserves_order(shard_set):
    views, out, metadata = shard_set
    unique_tics = np.unique(views.tic_ids)
    codes_by_tic = {t: int(Split.TRAIN) for t in unique_tics}
    for t in unique_tics[-3:]:
        codes_by_tic[t] = int(Split.VAL)
    row_codes = np.array([codes_by_tic[t] for t in views.tic_ids], dtype=np.int64)
    table = make_split_table(views.tic_ids, row_codes)

    for split in (Split.TRAIN, Split.VAL):
        expected_rows = np.where(row_codes == int(split))[0]
        ds = make_dataset(
            list_shards(out), metadata, split_table=table, split=split, batch_size=64, cache=False
        )
        _, labels = next(iter(ds))
        assert len(labels) == len(expected_rows)
        np.testing.assert_allclose(labels.numpy(), views.labels[expected_rows].astype(np.float32))

    # Leakage guard: no TIC ID crosses splits.
    train_tics = set(views.tic_ids[row_codes == int(Split.TRAIN)])
    val_tics = set(views.tic_ids[row_codes == int(Split.VAL)])
    assert not train_tics & val_tics


def test_tf_aux_transform_matches_sklearn(shard_set):
    views, _, _ = shard_set
    pipeline = fit_aux_pipeline(views.aux_features[:20])
    constants = aux_constants_from_pipeline(pipeline)

    sk_out = pipeline.transform(views.aux_features).astype(np.float32)
    tf_out = np.stack(
        [tf_aux_transform(tf.constant(row), constants).numpy() for row in views.aux_features]
    )
    np.testing.assert_allclose(tf_out, sk_out, rtol=1e-4, atol=1e-5)


def test_aux_pipeline_survives_all_nan_column():
    """Regression: an all-NaN aux column (snr on a small fresh build) must not
    change the aux dimension — the imputer used to drop it, leaving the tf
    replay with 7 constants against 8 features."""
    rng = np.random.default_rng(4)
    aux = rng.normal(size=(40, 8)).astype(np.float32)
    aux[:, 7] = np.nan  # dead column, as produced by the 2026-07-09 fresh build
    aux[rng.random(size=aux.shape) < 0.1] = np.nan

    pipeline = fit_aux_pipeline(aux[:30])
    constants = aux_constants_from_pipeline(pipeline)
    assert constants.aux_dim == 8  # dimension preserved

    sk_out = pipeline.transform(aux).astype(np.float32)
    assert sk_out.shape == (40, 8)
    np.testing.assert_allclose(sk_out[:, 7], 0.0)  # dead column -> constant 0

    tf_out = np.stack([tf_aux_transform(tf.constant(row), constants).numpy() for row in aux])
    np.testing.assert_allclose(tf_out, sk_out, rtol=1e-4, atol=1e-5)


def test_augment_preserves_shape_and_is_stochastic():
    tf.random.set_seed(7)
    g = tf.random.normal((GLOBAL_BINS, 1))
    l = tf.random.normal((LOCAL_BINS, 1))
    cfg = AugmentConfig()
    g1, l1 = augment_views(g, l, cfg)
    g2, _ = augment_views(g, l, cfg)
    assert g1.shape == g.shape and l1.shape == l.shape
    assert not np.allclose(g1.numpy(), g2.numpy())  # fresh draws per call
    # Magnitude sanity: augmented signal stays close to the original.
    assert float(tf.reduce_mean(tf.abs(g1 - g))) < 0.5


def test_augmented_dataset_differs_across_epochs(shard_set):
    _, out, metadata = shard_set
    ds = make_dataset(
        list_shards(out),
        metadata,
        batch_size=64,
        augment=AugmentConfig(),
        cache=True,
        seed=3,
    )
    epoch1 = next(iter(ds))[0]["global_view"].numpy()
    epoch2 = next(iter(ds))[0]["global_view"].numpy()
    assert not np.allclose(epoch1, epoch2)  # augmentation runs after cache()


@pytest.mark.slow
def test_end_to_end_mini_training(tmp_path):
    """One epoch of the real trainer over synthetic shards (CPU, ~1 min)."""
    from hydra import compose, initialize_config_dir

    from exoplanet_hunter.training.train import run

    views = synthetic_views(n=60, seed=1)
    shard_dir = tmp_path / "data" / "processed" / "tfrecords"
    write_tfrecord_shards(views, shard_dir, examples_per_shard=16)

    conf_dir = str((tmp_path / "conf").resolve())
    import shutil
    from pathlib import Path

    shutil.copytree(Path(__file__).resolve().parents[1] / "conf", conf_dir)
    with initialize_config_dir(version_base="1.3", config_dir=conf_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"paths.root={tmp_path}",
                f"mlflow.tracking_uri=sqlite:///{tmp_path}/mlflow.db",
                "train.epochs=1",
                "train.batch_size=16",
                "model.cross_validation.n_splits=2",
                "model.global_view.conv_blocks=[8]",
                "model.local_view.conv_blocks=[8]",
                "model.head.fc_units=[16]",
                "model.attention.num_heads=2",
                "model.attention.key_dim=4",
            ],
        )
    auc = run(cfg)
    assert 0.0 <= auc <= 1.0
    # Both folds saved a model + a bundle with the serving-contract keys.
    import joblib

    bundles = sorted((tmp_path / "models" / "cv").rglob("cnn_calibrator.joblib"))
    assert len(bundles) == 2
    bundle = joblib.load(bundles[0])
    assert {"calibrator", "platt_a", "platt_b", "threshold", "aux_pipeline", "aux_dim"} <= set(
        bundle
    )


def test_rewrite_clears_stale_shards(tmp_path):
    """Regression (2026-07-12): rebuilding into the same directory must not
    leave the previous set's shards behind — a different example count names
    shards differently, and the survivors poison readers with a mixed schema."""
    write_tfrecord_shards(synthetic_views(n=30), tmp_path, examples_per_shard=8)  # 4 shards
    metadata = write_tfrecord_shards(synthetic_views(n=12), tmp_path, examples_per_shard=8)
    shards = list_shards(tmp_path)
    assert len(shards) == metadata.n_shards == 2  # no of-00004 stragglers
    ds = make_dataset(shards, metadata, batch_size=64, cache=False)
    _, labels = next(iter(ds))
    assert len(labels) == 12  # parses cleanly end to end


def test_metadata_roundtrip(shard_set):
    _, out, metadata = shard_set
    assert ShardMetadata.load(out) == metadata
