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

import keras
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
    # `secondary_phase` is the phase the candidate secondary sits at, and it is
    # the one DV-adjacent scalar measured on all three missions — it was written
    # to every shard and read by no branch until 2026-08-07.
    "secondary_view": ("weak_secondary_max_mes", "secondary_phase"),
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

#: The mask column that vouches for each scalar-only branch. Both towers are fed
#: entirely from the DV report, which is absent on 100% of Kepler and K2 rows and
#: 12.8% of TESS — so without a gate they emit `relu(bias)`, a learned constant,
#: into fusion for 56% of the training set.
SCALAR_BRANCH_MASK: dict[str, str] = {"detection": "dv_usable", "ghost": "dv_usable"}


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


@keras.saving.register_keras_serializable(package="exoplanet_hunter")
class PresenceFlag(layers.Layer):
    """1.0 when a view holds at least one measured bin, 0.0 when it holds none.

    A registered layer rather than a `Lambda` over a Python lambda: Keras
    refuses to deserialise the latter without `safe_mode=False`, which would
    make every checkpoint unloadable without disabling a safety check — and the
    checkpoint is the artefact that gets promoted and served.
    """

    def call(self, view: tf.Tensor) -> tf.Tensor:
        bin_axes = list(range(1, len(view.shape) - 1))
        return tf.cast(tf.reduce_max(view[..., -1], axis=bin_axes) > 0.0, tf.float32)[:, None]

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return (input_shape[0], 1)


@keras.saving.register_keras_serializable(package="exoplanet_hunter")
class PickColumns(layers.Layer):
    """Gather a branch's scoped scalars out of the shared scalar vector."""

    def __init__(self, indices: list[int], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.indices = list(indices)

    def call(self, scalars: tf.Tensor) -> tf.Tensor:
        return tf.gather(scalars, self.indices, axis=1)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return (input_shape[0], len(self.indices))

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "indices": self.indices}


def _gated(x: tf.Tensor, view: tf.Tensor, name: str) -> tf.Tensor:
    """Zero a branch's contribution when its view holds no measured bins.

    The presence channel is the last one on every view. A branch with nothing
    measured must contribute nothing, not a learned bias on a zero tensor.
    """
    present = PresenceFlag(name=f"{name}_present")(view)
    return layers.Multiply(name=f"{name}_gate")([x, present])


def _mask_gated(
    x: tf.Tensor, mask_in: tf.Tensor, mask_index: dict[str, int], column: str, name: str
) -> tf.Tensor:
    """Zero a scalar-only branch when the report its inputs come from is absent.

    `_gated` reads a view's own presence channel. A scalar-only tower has no
    view, so it is gated on the mask column that vouches for its inputs instead.
    Same invariant, different source — a branch with nothing measured must
    contribute nothing rather than a learned bias on a zero tensor.
    """
    if column not in mask_index:
        raise ValueError(
            f"branch {name!r} gates on mask column {column!r}, which this shard set "
            f"does not carry (has: {sorted(mask_index)})"
        )
    flag = PickColumns([mask_index[column]], name=f"{name}_mask")(mask_in)
    return layers.Multiply(name=f"{name}_gate")([x, flag])


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
        missing = [n for n in names if n not in index]
        if missing:
            # Skipping silently leaves the branch with no scalars at all and it
            # still trains to a plausible AUC. BRANCH_SCALARS and the shard set's
            # scalar_columns are separate literals; this is what ties them.
            raise ValueError(
                f"scalars {missing} are declared on a branch but absent from this shard set "
                f"(has: {sorted(index)})"
            )
        if not names:
            return None
        return PickColumns([index[n] for n in names], name=f"pick_{names[0]}")(scalar_in)

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

    mask_index = {name: i for i, name in enumerate(mask_columns)}
    for name, columns in SCALAR_BRANCHES.items():
        picked = _slice(columns)
        if picked is None:
            continue
        tower = layers.Dense(branch_units, activation="relu", name=f"{name}_fc")(picked)
        embeddings.append(_mask_gated(tower, mask_in, mask_index, SCALAR_BRANCH_MASK[name], name))

    # The masks still ride into the head directly. The gate above removes the
    # branch's contribution; this is what tells the head the removal happened,
    # so "absent" and "measured, and it came out at the median" stay distinct.
    embeddings.append(mask_in)

    x = layers.Concatenate(name="fusion")(embeddings)
    for i, units in enumerate(head_units):
        x = layers.Dense(units, name=f"head_fc{i}")(x)
        x = layers.LeakyReLU(negative_slope=0.1, name=f"head_act{i}")(x)
        # training=None, so the call-time flag decides: deterministic under
        # predict(), stochastic when `mc_dropout_predict` asks for training=True.
        # This said training=True until 2026-08-07, with a comment claiming it
        # matched the dual-view model — that model uses None, and
        # `mc_dropout_predict` documents None as the contract it needs.
        x = layers.Dropout(dropout, name=f"head_drop{i}")(x, training=None)
    output = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    return Model(inputs=inputs, outputs=output, name="cnn_branches")
