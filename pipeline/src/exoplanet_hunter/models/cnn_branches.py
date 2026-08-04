"""Per-diagnostic branch CNN — stage 2(a) of the ExoMiner rebuild.

One conv tower per view, each with its own scoped scalars, then a late-fusion
head. Reimplemented from the ExoMiner/ExoMiner++ papers; their branch structure
is credited, their code is not vendored (NASA NOSA licence).

Two things are structural rather than stylistic:

**Presence masks gate their branch.** Every view carries a `present` channel
and the DV/RUWE scalars carry a mask. A branch with no data contributes zero to
the fusion rather than a learned bias on zeros, which is what stops a missing
branch poisoning every row of a mission.

**Scoped scalars join after their own tower.** A scalar concatenated into a
global feature vector is the 13-dim aux null again; attached to the branch it
qualifies, it can modulate that branch's evidence.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import Model, layers

from exoplanet_hunter.datasets.viewset_io import VIEW_SHAPES

#: view name -> indices into the scalar vector that belong to that branch.
#: Names index `viewset_tfrecords.FEATURE_COLUMNS`.
BRANCH_SCALARS: dict[str, tuple[str, ...]] = {
    "global_view": (),
    "local_view": (),
    "odd_view": ("odd_even_statistic",),
    "even_view": (),
    "secondary_view": ("weak_secondary_max_mes",),
    "trend_view": (),
    "centroid_view": ("mean_sky_offset", "control_sky_offset", "ruwe"),
    "unfolded_view": (
        "observed_transit_count",
        "expected_transit_count",
        "transit_completeness",
    ),
    "gap_view": (),
    "periodogram_view": (),
    "periodogram_masked_view": (),
}

#: Scalars with no view of their own; one small dense tower each.
SCALAR_BRANCHES: dict[str, tuple[str, ...]] = {
    "detection": (
        "max_multiple_event_sigma",
        "robust_statistic",
        "bootstrap_significance",
        "summary_quality_fraction",
    ),
    "ghost": ("ghost_core_statistic", "ghost_halo_statistic"),
}


def _conv_tower(
    x: tf.Tensor,
    *,
    blocks: int,
    filters: int,
    kernel_size: int,
    pool_size: int,
    name: str,
) -> tf.Tensor:
    """Conv-BN-ReLU blocks with max pooling, ending in global average pooling."""
    for block in range(blocks):
        x = layers.Conv1D(
            filters * (2**block),
            kernel_size,
            padding="same",
            name=f"{name}_conv{block}",
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn{block}")(x)
        x = layers.Activation("relu", name=f"{name}_relu{block}")(x)
        if x.shape[1] is not None and x.shape[1] >= pool_size * 2:
            x = layers.MaxPooling1D(pool_size, name=f"{name}_pool{block}")(x)
    return layers.GlobalAveragePooling1D(name=f"{name}_gap")(x)


def _branch(
    view: tf.Tensor,
    scalars: tf.Tensor | None,
    *,
    blocks: int,
    filters: int,
    kernel_size: int,
    pool_size: int,
    units: int,
    name: str,
) -> tf.Tensor:
    """One diagnostic's tower: conv over the view, then its scoped scalars."""
    if len(view.shape) == 4:
        # Unfolded view is (transits, bins, channels): convolve each transit
        # with shared weights, then pool across transits, so the branch sees
        # transit-to-transit variation rather than an average.
        transits = view.shape[1]
        x = layers.Reshape((transits, view.shape[2] * view.shape[3]), name=f"{name}_flatten")(view)
    else:
        x = view
    x = _conv_tower(
        x, blocks=blocks, filters=filters, kernel_size=kernel_size, pool_size=pool_size, name=name
    )
    if scalars is not None:
        x = layers.Concatenate(name=f"{name}_with_scalars")([x, scalars])
    x = layers.Dense(units, activation="relu", name=f"{name}_fc")(x)
    return x


def _gated(x: tf.Tensor, view: tf.Tensor, name: str) -> tf.Tensor:
    """Zero a branch's contribution when its view holds no measured bins.

    The presence channel is the last one on every view. A branch with nothing
    measured must contribute nothing, not a learned bias on a zero tensor.
    """
    present = layers.Lambda(
        lambda t: tf.cast(
            tf.reduce_max(t[..., -1], axis=list(range(1, len(t.shape) - 1))) > 0.0, tf.float32
        )[:, None],
        name=f"{name}_present",
    )(view)
    return layers.Multiply(name=f"{name}_gate")([x, present])


def build_cnn_branches(
    model_cfg: Any,
    *,
    scalar_columns: list[str],
    mask_columns: list[str],
) -> Model:
    """Construct the per-diagnostic branch model as a Keras Functional `Model`.

    Parameters
    ----------
    model_cfg      : the `model` Hydra group.
    scalar_columns : order of the `scalars` feature vector in the shard set.
    mask_columns   : order of the `masks` vector (presence flags for DV / RUWE).
    """
    blocks = int(getattr(model_cfg, "conv_blocks", 2))
    filters = int(getattr(model_cfg, "init_filters", 16))
    kernel_size = int(getattr(model_cfg, "kernel_size", 5))
    pool_size = int(getattr(model_cfg, "pool_size", 4))
    branch_units = int(getattr(model_cfg, "branch_units", 32))
    head_units = list(getattr(model_cfg, "head_units", [256, 128]))
    dropout = float(getattr(model_cfg, "dropout", 0.3))

    index = {name: i for i, name in enumerate(scalar_columns)}
    inputs: dict[str, tf.Tensor] = {
        name: layers.Input(shape=shape, name=name) for name, shape in VIEW_SHAPES.items()
    }
    scalar_in = layers.Input(shape=(len(scalar_columns),), name="scalars")
    mask_in = layers.Input(shape=(len(mask_columns),), name="masks")
    inputs["scalars"] = scalar_in
    inputs["masks"] = mask_in

    def _slice(names: tuple[str, ...]) -> tf.Tensor | None:
        wanted = [index[n] for n in names if n in index]
        if not wanted:
            return None
        return layers.Lambda(lambda t: tf.gather(t, wanted, axis=1), name=f"pick_{names[0]}")(
            scalar_in
        )

    embeddings: list[tf.Tensor] = []
    for name in VIEW_SHAPES:
        view = inputs[name]
        embedding = _branch(
            view,
            _slice(BRANCH_SCALARS.get(name, ())),
            blocks=blocks,
            filters=filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            units=branch_units,
            name=name,
        )
        embeddings.append(_gated(embedding, view, name))

    for name, columns in SCALAR_BRANCHES.items():
        picked = _slice(columns)
        if picked is None:
            continue
        embeddings.append(layers.Dense(branch_units, activation="relu", name=f"{name}_fc")(picked))

    # The masks ride into the head directly: the model needs to know a scalar
    # branch was fed zeros because nothing was measured.
    embeddings.append(mask_in)

    x = layers.Concatenate(name="fusion")(embeddings)
    for i, units in enumerate(head_units):
        x = layers.Dense(units, name=f"head_fc{i}")(x)
        x = layers.LeakyReLU(negative_slope=0.1, name=f"head_act{i}")(x)
        # training=True keeps MC-dropout uncertainty working at inference, as
        # the dual-view model does.
        x = layers.Dropout(dropout, name=f"head_drop{i}")(x, training=True)
    output = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    return Model(inputs=inputs, outputs=output, name="cnn_branches")
