"""Per-diagnostic branch CNN — stage 4 (old 2(a)) of the ExoMiner rebuild.

One conv tower per view, each with its own scoped scalars, then a late-fusion
head. Reimplemented from the ExoMiner/ExoMiner++ papers; their branch structure
is credited, their code is not vendored (NASA NOSA licence).

Three things are structural rather than stylistic:

**Presence masks gate their branch.** Every view carries a `present` channel
and the DV/RUWE scalars carry a mask. A branch with no data contributes zero to
the fusion rather than a learned bias on zeros, which is what stops a missing
branch poisoning every row of a mission.

**Scoped scalars join after their own tower.** A scalar concatenated into a
global feature vector is the 13-dim aux null again; attached to the branch it
qualifies, it can modulate that branch's evidence.

**Pooling across transits is masked.** The unfolded stack is zero-padded to
`MAX_TRANSITS` and 30.4% of the training set carries at least one padded slot,
so an unmasked pool would make the branch's output scale with how many transits
a target happened to catch — which correlates with the label. See
`_unfolded_branch` and `MaskedTransitPool`.
"""

from __future__ import annotations

from collections.abc import Iterable
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
    # `odd_even_statistic` qualifies the *pair*, so it is scoped to the contrast
    # branch (CONTRAST_SCALARS) rather than to either half on its own.
    "odd_view": (),
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
    # `mean_sky_offset` and `control_sky_offset` are DV's own summary of these
    # stamps and would sit naturally here, but they are already scoped to
    # `centroid_view`. Moving them would change two branches at once and make
    # this branch's contribution unattributable, so they stay where they are.
    "difference_view": (),
    # Nothing DV publishes qualifies a momentum dump — the flag is a fact about
    # the spacecraft, and the branch's whole content is where the flagged
    # cadences fall relative to the transit.
    "momentum_dump_view": (),
}

#: Views that feed another branch rather than owning one. `difference_quality_
#: view` is how the difference branch weights its sectors; built as a branch of
#: its own it would put DV's opinion of the data straight into fusion as
#: evidence about the target.
ATTENTION_VIEWS: frozenset[str] = frozenset({"difference_quality_view"})

#: Views that are the same measurement — phase-folded flux at LOCAL_BINS — and
#: are meant to be compared against each other. They share one conv tower, so a
#: depth difference the head reads between odd and even transits is a difference
#: in the data and not partly a difference between two independently-learned
#: sets of kernels. `centroid_view` has the same shape but carries a pixel shift
#: in units of its own scatter, so it is not comparable and keeps its own tower.
SHARED_LOCAL_VIEWS: tuple[str, ...] = ("local_view", "odd_view", "even_view", "secondary_view")
#: The pair whose *difference* is the diagnostic. Fusion takes `odd - even`
#: rather than the two embeddings: an eclipsing binary shows alternating depths,
#: and a subtraction is only meaningful under tied weights.
CONTRAST_PAIR: tuple[str, str] = ("odd_view", "even_view")
#: The fusion branch the pair produces. Named rather than spelled inline because
#: `BRANCH_NAMES` and the drop mechanism both have to refer to it.
CONTRAST_BRANCH = "odd_even"
#: Scoped onto the contrast branch rather than either half of the pair.
CONTRAST_SCALARS: tuple[str, ...] = ("odd_even_statistic",)

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

#: Every branch that reaches fusion, **derived** from the constants that build
#: them rather than listed. A hand-written list is one more literal to drift out
#: of step with `VIEW_SHAPES`, and the drop mechanism validates against this: a
#: name that silently fell out would turn a declared ablation into a no-op that
#: still trains to a plausible AUC.
BRANCH_NAMES: frozenset[str] = frozenset(
    [name for name in SHARED_LOCAL_VIEWS if name not in CONTRAST_PAIR]
    + [CONTRAST_BRANCH]
    + [
        name
        for name in VIEW_SHAPES
        if name not in SHARED_LOCAL_VIEWS and name not in ATTENTION_VIEWS
    ]
    + list(SCALAR_BRANCHES)
)

#: How stage 7's leave-one-out attribution groups those branches. A family is
#: dropped as a unit where its members are the same measurement and a partial
#: drop would answer no clean question: the periodogram pair is one diagnostic
#: masked two ways, and the flux family shares a conv tower, so removing
#: `local_view` alone would also change what `odd - even` is measured against.
#:
#: `global_view` is kept apart from the flux family despite being folded flux —
#: it is 2001 bins through its own tower, so it is separately attributable and
#: separating it costs one run.
#:
#: These partition `BRANCH_NAMES` exactly; `test_the_families_partition_every_
#: branch` is what keeps that true as branches are added.
BRANCH_FAMILIES: dict[str, tuple[str, ...]] = {
    "flux": ("local_view", CONTRAST_BRANCH, "secondary_view"),
    "global": ("global_view",),
    "unfolded": ("unfolded_view",),
    "trend": ("trend_view",),
    "periodogram": ("periodogram_view", "periodogram_masked_view"),
    "centroid": ("centroid_view",),
    "gap": ("gap_view",),
    "difference": ("difference_view",),
    # Its own family, not folded into "gap". `_gap_view` measures the hole a
    # dump leaves and cannot tell it from any other lost cadence; this measures
    # the dumps themselves. Dropping them together would answer neither
    # question, and `docs/roadmap.md` 4.2c reads them separately.
    "momentum": ("momentum_dump_view",),
    "scalar_only": tuple(SCALAR_BRANCHES),
}


def resolve_dropped_branches(spec: object) -> frozenset[str]:
    """Branch names to leave out of fusion, from a family or branch spec.

    Leaving a branch out has to be a *declared experiment* rather than a code
    edit: `run_config.model_config` records whatever `model_cfg` carries, so a
    run's own summary states which branches it was missing. Editing
    `build_cnn_branches` to comment one out records nothing, and stage 4 already
    produced four runs whose architecture was not recoverable from the artefact.

    Accepts family names (`"periodogram"`) and individual branch names
    (`"gap_view"`). Both raise on anything unrecognised rather than silently
    dropping nothing — an ablation that quietly ablated nothing would report a
    delta of zero and read as "this branch does not matter", which is the
    opposite of what happened.
    """
    if spec is None or isinstance(spec, bool):
        return frozenset()
    if isinstance(spec, str):
        items: list[str] = [spec]
    elif isinstance(spec, Iterable):
        items = [str(item) for item in spec]
    else:
        # A bare scalar — `drop_branches: 3` in the YAML — would otherwise reach
        # the loop below as a one-element sequence or not at all.
        raise ValueError(f"drop_branches must be a name or a list of names, got {spec!r}")

    dropped: set[str] = set()
    unknown: list[str] = []
    for item in items:
        name = str(item)
        if name in BRANCH_FAMILIES:
            dropped.update(BRANCH_FAMILIES[name])
        elif name in BRANCH_NAMES:
            dropped.add(name)
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(
            f"unknown branch or family {sorted(unknown)}; "
            f"families are {sorted(BRANCH_FAMILIES)}, branches are {sorted(BRANCH_NAMES)}"
        )
    if dropped >= BRANCH_NAMES:
        raise ValueError(
            f"dropping {sorted(dropped)} leaves no branch feeding fusion — the model would "
            f"score every row from the presence masks alone"
        )
    return frozenset(dropped)


#: Slots a target must have measured before its transit-to-transit spread is a
#: measurement rather than an artefact of having one transit.
MIN_TRANSITS_FOR_SPREAD = 2
#: Added under the spread's square root. `sqrt` has an infinite derivative at
#: zero, and a masked variance is *exactly* zero — not merely small — whenever
#: fewer than two slots are measured: 17 training rows have one filled slot and
#: 5 have none. Without it those rows return NaN gradients, and a NaN gradient
#: takes the fold, not the batch.
SPREAD_EPSILON = 1.0e-6


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


def _conv_tower_model(
    input_shape: tuple[int, ...],
    *,
    blocks: int,
    filters: int,
    kernel_size: int,
    pool_size: int,
    name: str,
) -> Model:
    """The conv stack as a reusable `Model`, so `TimeDistributed` can tie weights."""
    view = layers.Input(shape=input_shape, name=f"{name}_input")
    encoded = _conv_tower(
        view,
        blocks=blocks,
        filters=filters,
        kernel_size=kernel_size,
        pool_size=pool_size,
        name=name,
    )
    return Model(view, encoded, name=name)


#: Conv kernel over a difference-image stamp. Three, not the 5 the phase-folded
#: views use: those run over 201-2001 bins, a stamp is 17 px across, and a 5x5
#: kernel spans nearly a third of it — wide enough that a centroid shift and the
#: star itself fall inside one receptive field.
DIFF_KERNEL_SIZE = 3
#: Pooling stride for the same reason. 17 -> 8 -> 4 over two blocks still
#: resolves which side of the aperture the flux moved to, which is the question.
DIFF_POOL_SIZE = 2


def _conv_tower_2d(
    input_shape: tuple[int, ...],
    *,
    blocks: int,
    filters: int,
    name: str,
) -> Model:
    """Conv-BN-ReLU blocks over a 2-D stamp, as a reusable `Model`.

    Separate from `_conv_tower` rather than generalised over rank: the 1-D towers
    are the project's main measurement path, and threading a rank switch through
    them to serve one branch would put every existing branch's behaviour on a
    code path this one exercises.
    """
    stamp = layers.Input(shape=input_shape, name=f"{name}_input")
    x = stamp
    for block in range(blocks):
        x = layers.Conv2D(
            filters * (2**block),
            DIFF_KERNEL_SIZE,
            padding="same",
            name=f"{name}_conv{block}",
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn{block}")(x)
        x = layers.Activation("relu", name=f"{name}_relu{block}")(x)
        if x.shape[1] is not None and x.shape[1] >= DIFF_POOL_SIZE * 2:
            x = layers.MaxPooling2D(DIFF_POOL_SIZE, name=f"{name}_pool{block}")(x)
    return Model(stamp, layers.GlobalAveragePooling2D(name=f"{name}_gap")(x), name=name)


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
        return _unfolded_branch(
            view,
            scalars,
            blocks=blocks,
            filters=filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            units=units,
            name=name,
        )
    if len(view.shape) != 3:
        # A view of an unhandled rank would otherwise reach Conv1D and either
        # fail somewhere unrecognisable or convolve the wrong axis, which is the
        # defect this function was rebuilt to remove.
        raise ValueError(
            f"branch {name!r} got a rank-{len(view.shape)} view; a branch takes "
            "(bins, channels) or (transits, bins, channels)"
        )
    x = _conv_tower(
        view,
        blocks=blocks,
        filters=filters,
        kernel_size=kernel_size,
        pool_size=pool_size,
        name=name,
    )
    if scalars is not None:
        x = layers.Concatenate(name=f"{name}_with_scalars")([x, scalars])
    x = layers.Dense(units, activation="relu", name=f"{name}_fc")(x)
    return x


def _unfolded_branch(
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
    """Per-transit tower under `TimeDistributed`, then a masked pool across transits.

    What the branch was always documented to do, and did not. Until 2026-08-08
    it reshaped `(transits, bins, channels)` to `(transits, bins * channels)`
    and convolved along the **transit** axis, so the 201 phase bins became 603
    unordered channels, the transit shape was destroyed before the first
    convolution, and no weights were shared across phase. The branch could not
    see what a single transit looks like — only how flattened transit vectors
    varied down the sequence. It also spent **48,256 of the model's 215,281
    parameters** on that one 603-channel convolution.

    **Its own tower, not the shared flux one.** ExoMiner stacks the unfolded
    transits into the same `TimeDistributed` call as the folded local family,
    and the shapes here would allow it. `BatchNormalization` under
    `TimeDistributed` computes its statistics over batch × stack entries, so
    twenty low-SNR single transits would set the normalisation for the four
    high-SNR folded views as well — moving `local_view`, the odd/even contrast
    and `secondary_view` in the same change that fixes this branch, and
    confounding the attribution stage 7 (old D) exists to do.

    **Three statistics, not an average.** A mean across transits is the folded
    view again, which is the thing this branch exists to avoid. `mean` is the
    reference, `max` catches a single anomalous transit, and the spread is the
    transit-to-transit variation an eclipsing binary or a background blend
    shows and a planet does not.
    """
    _, bins, channels = view.shape[1:]
    tower = _conv_tower_model(
        (bins, channels),
        blocks=blocks,
        filters=filters,
        kernel_size=kernel_size,
        pool_size=pool_size,
        name=f"{name}_tower",
    )
    encoded = layers.TimeDistributed(tower, name=f"{name}_td")(view)
    measured = TransitPresence(name=f"{name}_transits_present")(view)
    parts = [
        MaskedTransitPool(name=f"{name}_pool")([encoded, measured]),
        SpreadMeasurable(name=f"{name}_spread_measurable")(measured),
    ]
    if scalars is not None:
        parts.append(scalars)
    x = layers.Concatenate(name=f"{name}_with_scalars")(parts)
    return layers.Dense(units, activation="relu", name=f"{name}_fc")(x)


def _difference_branch(
    view: tf.Tensor,
    quality: tf.Tensor,
    scalars: tf.Tensor | None,
    *,
    blocks: int,
    filters: int,
    units: int,
    name: str,
) -> tf.Tensor:
    """Per-sector 2-D tower, pooled across sectors by attention over DV's quality.

    A star's difference image is measured once per sector, and the sectors are
    not equally worth reading: DV publishes a quality metric per image, and
    17.5% of them are flagged invalid. A plain mean over sectors would let one
    bad image drag a good one, and there is no fixed number of them to average —
    the count runs from 1 to 43, with a median of 3.

    So the pool is an attention over sectors whose logits see both the encoded
    stamp and DV's own quality for it. Weighting on quality *alone* would ignore
    what the image shows; on the embedding alone it would ignore that DV already
    said which images to distrust.

    **The mask, not a mean.** 58.9% of the set has no difference image at all,
    and unmeasured sectors are zero-padded slots. Attention over them would put
    the padding's learned embedding into the pool, and — because the number of
    measured sectors is how many times TESS looked at the star — make this
    branch's output scale with observation baseline, which is the confound the
    whole project has been pulling out of its metrics.
    """
    tower = _conv_tower_2d(
        tuple(view.shape[2:]),
        blocks=blocks,
        filters=filters,
        name=f"{name}_tower",
    )
    encoded = layers.TimeDistributed(tower, name=f"{name}_td")(view)
    measured = SectorPresence(name=f"{name}_sectors_present")(view)
    logits = layers.TimeDistributed(layers.Dense(1, name=f"{name}_score"), name=f"{name}_score_td")(
        layers.Concatenate(name=f"{name}_scored")([encoded, quality])
    )
    parts: list[tf.Tensor] = [MaskedAttentionPool(name=f"{name}_pool")([encoded, logits, measured])]
    if scalars is not None:
        parts.append(scalars)
    pooled = parts[0] if len(parts) == 1 else layers.Concatenate(name=f"{name}_with_scalars")(parts)
    return layers.Dense(units, activation="relu", name=f"{name}_fc")(pooled)


@keras.saving.register_keras_serializable(package="exoplanet_hunter")
class SectorPresence(layers.Layer):
    """Which sector slots hold a measured stamp.

    Reads the stamp's own presence channel, so the three states DV distinguishes
    stay distinguished: no report, a sector DV declined to measure, and a sector
    measured and flat. Only the third has a presence channel set, and only the
    third is evidence that the star did not move.
    """

    def call(self, view: tf.Tensor) -> tf.Tensor:
        return tf.cast(tf.reduce_max(view[..., -1], axis=(2, 3)) > 0.0, tf.float32)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return (input_shape[0], input_shape[1])


@keras.saving.register_keras_serializable(package="exoplanet_hunter")
class MaskedAttentionPool(layers.Layer):
    """Softmax over measured slots only, then the weighted sum of their embeddings.

    Takes `[encoded, logits, measured]` — `(batch, slots, width)` embeddings,
    `(batch, slots, 1)` attention logits and the `(batch, slots)` flags — and
    returns `(batch, width)`.

    **Finite when nothing is measured, which is the common case here.** Masking
    by adding `-inf` to absent slots is the textbook form and returns NaN when
    every slot is absent — true for 58.9% of this set. The branch is gated on
    presence downstream and a gate multiplies, so `NaN * 0` would poison the
    whole model rather than contribute nothing. A large finite offset plus an
    explicit zeroing keeps every row finite.
    """

    def __init__(self, offset: float = 1.0e9, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.offset = float(offset)

    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        encoded, logits, measured = inputs
        mask = measured[..., None]
        shifted = logits - (1.0 - mask) * self.offset
        # Subtracting the row max before exponentiating: the logits are learned
        # and unbounded, and exp() of a large one overflows to inf in float32
        # long before training diverges enough to notice.
        shifted = shifted - tf.reduce_max(shifted, axis=1, keepdims=True)
        weights = tf.exp(shifted) * mask
        weights = weights / tf.maximum(tf.reduce_sum(weights, axis=1, keepdims=True), 1.0e-9)
        return tf.reduce_sum(encoded * weights, axis=1)

    def compute_output_shape(self, input_shape: list[tuple]) -> tuple:
        encoded = input_shape[0]
        return (encoded[0], encoded[2])

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "offset": self.offset}


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
class StackViews(layers.Layer):
    """Stack same-shaped views on a new axis, for a `TimeDistributed` tower.

    Registered rather than a `Lambda`, for the reason in `PresenceFlag`.
    """

    def call(self, views: list[tf.Tensor]) -> tf.Tensor:
        return tf.stack(views, axis=1)

    def compute_output_shape(self, input_shape: list[tuple]) -> tuple:
        first = input_shape[0]
        return (first[0], len(input_shape), *first[1:])


@keras.saving.register_keras_serializable(package="exoplanet_hunter")
class PickView(layers.Layer):
    """One view's feature vector out of the shared tower's stacked output."""

    def __init__(self, index: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.index = int(index)

    def call(self, encoded: tf.Tensor) -> tf.Tensor:
        return encoded[:, self.index, :]

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return (input_shape[0], input_shape[2])

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "index": self.index}


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


@keras.saving.register_keras_serializable(package="exoplanet_hunter")
class TransitPresence(layers.Layer):
    """Which transit slots hold at least one measured bin.

    `preprocess.viewset._unfolded` allocates `zeros((MAX_TRANSITS, bins, 3))`
    and fills only the epochs that caught a cadence, so an unused slot is all
    zeros — presence channel included. Nothing downstream can tell a padded
    slot from a measured one unless it reads that channel, and padding is
    neither rare nor label-neutral: **30.4% of the training set carries at
    least one padded slot**, 16% have fewer than ten of twenty filled, and on
    TESS the 25th percentile of occupancy is 0.50. On K2 planet hosts average
    12.4 filled slots against 17.2 for false positives.

    So an unmasked pool would divide by twenty regardless and make the branch's
    output scale with occupancy — reintroducing the observation-baseline
    confound (roadmap, stage 8) through the one branch built to measure
    transits rather than hosts.
    """

    def call(self, view: tf.Tensor) -> tf.Tensor:
        return tf.cast(tf.reduce_max(view[..., -1], axis=-1) > 0.0, tf.float32)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return (input_shape[0], input_shape[1])


@keras.saving.register_keras_serializable(package="exoplanet_hunter")
class MaskedTransitPool(layers.Layer):
    """Mean, max and spread of the per-transit embeddings, over measured slots only.

    Takes `[encoded, measured]` — `(batch, transits, width)` embeddings and the
    `(batch, transits)` flags from `TransitPresence` — and returns
    `(batch, 3 * width)`.

    Every statistic stays finite when nothing is measured. The branch is gated
    on the view's presence channel downstream, and a gate multiplies: `NaN * 0`
    is `NaN`, so a pool that returned NaN on an empty stack would poison the
    whole model rather than contributing nothing. Five training rows have zero
    filled slots.
    """

    def __init__(self, epsilon: float = SPREAD_EPSILON, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        encoded, measured = inputs
        mask = measured[..., None]
        count = tf.reduce_sum(mask, axis=1)
        divisor = tf.maximum(count, 1.0)

        mean = tf.reduce_sum(encoded * mask, axis=1) / divisor
        # Masked *after* subtracting, so an unmeasured slot cannot contribute
        # its distance from the mean. Summing squared deviations keeps the
        # variance non-negative by construction — `E[x²] - E[x]²` goes slightly
        # negative in float32 and returns NaN from the sqrt.
        deviation = (encoded - mean[:, None, :]) * mask
        variance = tf.reduce_sum(tf.square(deviation), axis=1) / divisor
        spread = tf.sqrt(variance + self.epsilon)

        # Lowest representable rather than zero: `max` over `encoded * mask`
        # would be correct only while the tower ends in a ReLU, and would go
        # silently wrong the day it does not.
        #
        # A *scalar* `lowest`, broadcast by `tf.where`, rather than a
        # `tf.fill(tf.shape(encoded), ...)`: filling from a dynamic shape leaves
        # the result with no static shape, the concat below inherits that, and
        # `_ConcatGradV2` then aborts the process — not an exception — when the
        # gradient runs.
        lowest = tf.constant(encoded.dtype.min, dtype=encoded.dtype)
        largest = tf.reduce_max(tf.where(mask > 0.0, encoded, lowest), axis=1)
        largest = tf.where(count > 0.0, largest, tf.zeros_like(largest))

        return tf.concat([mean, largest, spread], axis=-1)

    def compute_output_shape(self, input_shape: list[tuple]) -> tuple:
        encoded = input_shape[0]
        return (encoded[0], 3 * encoded[2])

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "epsilon": self.epsilon}


@keras.saving.register_keras_serializable(package="exoplanet_hunter")
class SpreadMeasurable(layers.Layer):
    """1.0 when enough transit slots were measured for a spread to mean anything.

    A spread of zero from one transit and a spread of zero from twenty
    identical transits are the same float with opposite meanings — the first is
    *unmeasured*, the second is the strongest evidence the branch can offer. On
    its own that is this project's recurring defect: a plausible number where
    the honest answer is "no measurement". The head cannot recover the
    distinction from the scoped scalars either, because `observed_transit_count`
    is the true count and the stack is capped at `MAX_TRANSITS`.
    """

    def __init__(self, minimum: int = MIN_TRANSITS_FOR_SPREAD, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.minimum = int(minimum)

    def call(self, measured: tf.Tensor) -> tf.Tensor:
        count = tf.reduce_sum(measured, axis=1, keepdims=True)
        return tf.cast(count >= float(self.minimum), tf.float32)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        return (input_shape[0], 1)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "minimum": self.minimum}


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

    def _head(features: tf.Tensor, scoped: tf.Tensor | None, name: str) -> tf.Tensor:
        if scoped is not None:
            features = layers.Concatenate(name=f"{name}_with_scalars")([features, scoped])
        return layers.Dense(branch_units, activation="relu", name=f"{name}_fc")(features)

    embeddings: list[tf.Tensor] = []

    # A declared experiment, not a code edit — and it lands in
    # `run_config.model_config`, so the run's own summary says what it was
    # missing. Every `Input` stays in the signature whether or not its branch is
    # built: the shard stream always yields all eleven views, and an ablation
    # that also changed the input contract would not be a controlled comparison.
    dropped = resolve_dropped_branches(getattr(model_cfg, "drop_branches", ()))

    # One tower over the whole phase-folded-flux family. Four independent towers
    # meant an odd-versus-even difference was partly a difference between two
    # sets of kernels, which is the one comparison this branch exists to make.
    shared_branches = [
        name for name in (*SHARED_LOCAL_VIEWS, CONTRAST_BRANCH) if name not in CONTRAST_PAIR
    ]
    if any(name not in dropped for name in shared_branches):
        shared = _conv_tower_model(
            VIEW_SHAPES[SHARED_LOCAL_VIEWS[0]],
            blocks=blocks,
            filters=filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            name="local_shared",
        )
        encoded = layers.TimeDistributed(shared, name="local_shared_td")(
            StackViews(name="local_stack")([inputs[v] for v in SHARED_LOCAL_VIEWS])
        )
        features = {
            view: PickView(i, name=f"{view}_features")(encoded)
            for i, view in enumerate(SHARED_LOCAL_VIEWS)
        }

        for name in SHARED_LOCAL_VIEWS:
            if name in CONTRAST_PAIR or name in dropped:
                continue
            head = _head(features[name], _slice(BRANCH_SCALARS.get(name, ())), name)
            embeddings.append(_gated(head, inputs[name], name))

        if CONTRAST_BRANCH not in dropped:
            odd, even = CONTRAST_PAIR
            contrast = _head(
                layers.Subtract(name="odd_even_difference")([features[odd], features[even]]),
                _slice(CONTRAST_SCALARS),
                CONTRAST_BRANCH,
            )
            # Both halves have to be measured for their difference to mean
            # anything, so the contrast is gated on each in turn.
            embeddings.append(
                _gated(_gated(contrast, inputs[odd], f"{odd}_pair"), inputs[even], CONTRAST_BRANCH)
            )

    for name in VIEW_SHAPES:
        if name in SHARED_LOCAL_VIEWS or name in ATTENTION_VIEWS or name in dropped:
            continue
        view = inputs[name]
        if name == "difference_view":
            embedding = _difference_branch(
                view,
                inputs["difference_quality_view"],
                _slice(BRANCH_SCALARS.get(name, ())),
                blocks=blocks,
                filters=filters,
                units=branch_units,
                name=name,
            )
        else:
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
        if name in dropped:
            continue
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
