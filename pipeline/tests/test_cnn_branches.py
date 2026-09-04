"""Per-diagnostic branch model: wiring, presence gating, scoped scalars."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from exoplanet_hunter.datasets.viewset_io import VIEW_SHAPES
from exoplanet_hunter.datasets.viewset_tfrecords import FEATURE_COLUMNS, MASK_COLUMNS
from exoplanet_hunter.models.cnn_branches import (
    BRANCH_FAMILIES,
    BRANCH_NAMES,
    BRANCH_SCALARS,
    CONTRAST_BRANCH,
    CONTRAST_SCALARS,
    SCALAR_BRANCHES,
    SHARED_LOCAL_VIEWS,
    SPREAD_EPSILON,
    MaskedTransitPool,
    build_cnn_branches,
    resolve_dropped_branches,
)
from exoplanet_hunter.preprocess.diffimage import TARGET_CHANNEL

SCALARS = list(FEATURE_COLUMNS)
MASKS = list(MASK_COLUMNS)


@pytest.fixture(scope="module")
def model():
    return build_cnn_branches(SimpleNamespace(), scalar_columns=SCALARS, mask_columns=MASKS)


def make_batch(n: int = 4, *, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    batch = {}
    for name, shape in VIEW_SHAPES.items():
        arr = rng.normal(size=(n, *shape)).astype(np.float32)
        arr[..., -1] = (rng.random((n, *shape[:-1])) > 0.3).astype(np.float32)
        batch[name] = arr
    batch["scalars"] = rng.normal(size=(n, len(SCALARS))).astype(np.float32)
    batch["masks"] = np.ones((n, len(MASKS)), dtype=np.float32)
    return batch


def test_model_takes_every_view_plus_scalars_and_masks(model):
    names = {t.name.split(":")[0] for t in model.inputs}
    assert set(VIEW_SHAPES) <= names
    assert {"scalars", "masks"} <= names


def test_output_is_a_probability(model):
    out = model(make_batch(), training=False).numpy()
    assert out.shape == (4, 1)
    assert ((out >= 0) & (out <= 1)).all()


def test_scoring_is_deterministic_but_mc_dropout_still_works(model):
    """Head dropout must follow the call-time flag rather than being wired on.

    Built with `training=True` — as this model was until 2026-08-07 — every
    score is a stochastic draw, so a fold's AUC becomes a sample rather than a
    measurement. It does not even buy MC-dropout: `mc_dropout_predict` passes
    `training=True` itself and documents `training=None` as the contract.
    """
    batch = make_batch()
    fixed = [model(batch, training=False).numpy() for _ in range(3)]
    assert all(np.array_equal(fixed[0], other) for other in fixed[1:])

    sampled = [model(batch, training=True).numpy() for _ in range(6)]
    assert any(not np.array_equal(sampled[0], other) for other in sampled[1:])


@pytest.mark.parametrize(
    "view", ["centroid_view", "trend_view", "unfolded_view", "difference_view"]
)
def test_an_absent_branch_contributes_exactly_zero(model, view):
    # The whole point of the presence channel: a branch with nothing measured
    # must contribute nothing, not a learned bias on a zero tensor. Without
    # this, a mission that lacks a branch has every row poisoned.
    gate = model.get_layer(f"{view}_gate")
    probe = tf.keras.Model(inputs=model.inputs, outputs=gate.output)

    batch = make_batch()
    assert np.abs(probe(batch, training=False).numpy()).sum() > 0

    batch[view] = batch[view].copy()
    batch[view][..., -1] = 0.0
    assert np.abs(probe(batch, training=False).numpy()).sum() == pytest.approx(0.0)


def test_a_present_branch_is_not_gated_off(model):
    gate = model.get_layer("global_view_gate")
    probe = tf.keras.Model(inputs=model.inputs, outputs=gate.output)
    batch = make_batch()
    batch["global_view"][..., -1] = 1.0
    assert np.abs(probe(batch, training=False).numpy()).sum() > 0


def test_scoped_scalars_reach_their_own_branch(model):
    # A scalar concatenated into one global vector is the 13-dim aux null. It
    # has to move the branch it qualifies.
    name = "centroid_view"
    picked = BRANCH_SCALARS[name][0]
    probe = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(f"{name}_fc").output)

    batch = make_batch()
    before = probe(batch, training=False).numpy()
    batch["scalars"] = batch["scalars"].copy()
    batch["scalars"][:, SCALARS.index(picked)] += 25.0
    after = probe(batch, training=False).numpy()
    assert not np.allclose(before, after)


def test_an_unscoped_scalar_leaves_an_unrelated_branch_alone(model):
    # gap_view has no scoped scalars, so nothing in the scalar vector should
    # move it — that is what "scoped" means.
    probe = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer("gap_view_fc").output)
    batch = make_batch()
    before = probe(batch, training=False).numpy()
    batch["scalars"] = batch["scalars"] + 25.0
    assert np.allclose(before, probe(batch, training=False).numpy())


def test_masks_change_the_prediction(model):
    batch = make_batch()
    with_data = model(batch, training=False).numpy()
    batch["masks"] = np.zeros_like(batch["masks"])
    assert not np.allclose(with_data, model(batch, training=False).numpy())


def test_a_scalar_only_branch_contributes_nothing_when_its_report_is_absent():
    """`detection` and `ghost` are fed entirely from the DV report, which is
    absent on every Kepler and K2 row. Ungated they emit relu(bias) — a learned
    constant — into fusion for 56% of the training set."""
    batch = make_batch(seed=3)
    batch["masks"] = np.zeros((4, len(MASKS)), dtype=np.float32)
    gated = build_cnn_branches(SimpleNamespace(), scalar_columns=SCALARS, mask_columns=MASKS)

    layer_out = {
        name: gated.get_layer(f"{name}_gate")(
            [gated.get_layer(f"{name}_fc").output, gated.get_layer(f"{name}_mask").output]
        )
        for name in ("detection", "ghost")
    }
    probe = tf.keras.Model(gated.inputs, layer_out)
    for name, values in probe(batch, training=False).items():
        assert np.abs(values.numpy()).max() == 0.0, f"{name} leaked a bias on an absent report"


def test_a_declared_branch_scalar_missing_from_the_shard_set_raises():
    """Skipping it silently leaves the branch with no scalars and a plausible AUC."""
    with pytest.raises(ValueError, match="declared on a branch but absent"):
        build_cnn_branches(
            SimpleNamespace(),
            scalar_columns=[c for c in SCALARS if c != "odd_even_statistic"],
            mask_columns=MASKS,
        )


def test_a_missing_gate_mask_raises_rather_than_ungating_the_branch():
    with pytest.raises(ValueError, match="does not carry"):
        build_cnn_branches(SimpleNamespace(), scalar_columns=SCALARS, mask_columns=["has_ruwe"])


def test_every_declared_scalar_is_read_by_some_branch():
    """`secondary_phase` was written to every shard, normalised, and read by
    nothing — and it is the only DV-adjacent scalar present on all three
    missions. Separate literals with nothing tying them together."""
    consumed = {c for names in BRANCH_SCALARS.values() for c in names}
    consumed |= {c for names in SCALAR_BRANCHES.values() for c in names}
    consumed |= set(CONTRAST_SCALARS)
    assert set(FEATURE_COLUMNS) == consumed


def test_the_checkpoint_reloads_without_waiving_safe_mode(model, tmp_path):
    """Every custom layer must be a registered serialisable, not a `Lambda`.
    Gating once used a `Lambda` and the checkpoint could only be loaded with
    `safe_mode=False` — on the artefact that gets promoted and served."""
    batch = make_batch(seed=21)
    before = model(batch, training=False).numpy()

    path = tmp_path / "branches.keras"
    model.save(path)
    reloaded = tf.keras.models.load_model(path, compile=False)

    np.testing.assert_allclose(before, reloaded(batch, training=False).numpy(), rtol=1e-6)


# ------------------------------------------------- the shared local tower --


def test_the_flux_family_passes_through_one_set_of_weights(model):
    """Four independent towers made an odd-versus-even difference partly a
    difference between two sets of kernels — the one comparison this branch
    exists to make. Identical inputs must now give identical features."""
    batch = make_batch(seed=11)
    same = batch["local_view"].copy()
    for view in SHARED_LOCAL_VIEWS:
        batch[view] = same.copy()

    probe = tf.keras.Model(
        model.inputs, [model.get_layer(f"{v}_features").output for v in SHARED_LOCAL_VIEWS]
    )
    features = [f.numpy() for f in probe(batch, training=False)]
    for other in features[1:]:
        np.testing.assert_allclose(features[0], other, rtol=1e-6, atol=1e-6)


def test_identical_odd_and_even_transits_give_a_zero_contrast(model):
    """An eclipsing binary is the alternating-depth case; a planet is not. The
    difference can only be read as physical if the weights are tied."""
    batch = make_batch(seed=12)
    batch["even_view"] = batch["odd_view"].copy()
    difference = tf.keras.Model(model.inputs, model.get_layer("odd_even_difference").output)
    assert np.abs(difference(batch, training=False).numpy()).max() == pytest.approx(0.0, abs=1e-6)

    batch["even_view"] = batch["even_view"] * 0.5
    assert np.abs(difference(batch, training=False).numpy()).max() > 0


@pytest.mark.parametrize("absent", ["odd_view", "even_view"])
def test_the_contrast_needs_both_halves_measured(model, absent):
    batch = make_batch(seed=13)
    probe = tf.keras.Model(model.inputs, model.get_layer("odd_even_gate").output)
    assert np.abs(probe(batch, training=False).numpy()).sum() > 0

    batch[absent] = batch[absent].copy()
    batch[absent][..., -1] = 0.0
    assert np.abs(probe(batch, training=False).numpy()).sum() == pytest.approx(0.0)


# ------------------------------------------------ the unfolded branch -----

TRANSITS, BINS, CHANNELS = VIEW_SHAPES["unfolded_view"]


def unfolded_stack(measured: int, *, seed: int = 0, n: int = 1) -> np.ndarray:
    """An unfolded stack with `measured` slots filled and the rest padded.

    Padding is what the builder writes: `np.zeros`, presence channel included.
    """
    rng = np.random.default_rng(seed)
    stack = np.zeros((n, TRANSITS, BINS, CHANNELS), dtype=np.float32)
    stack[:, :measured, :, :-1] = rng.normal(size=(n, measured, BINS, CHANNELS - 1))
    stack[:, :measured, :, -1] = 1.0
    return stack


def probe_of(model, layer: str):
    return tf.keras.Model(model.inputs, model.get_layer(layer).output)


def embedding_width(model) -> int:
    """Per-transit embedding width, read off the tower rather than restated."""
    return int(model.get_layer("unfolded_view_td").output.shape[-1])


def test_each_transit_is_encoded_by_the_same_weights(model):
    """The branch's whole claim. Before 2026-08-08 it convolved along the
    transit axis with the 201 phase bins flattened into 603 unordered channels,
    so there was no weight sharing across phase at all."""
    batch = make_batch(seed=31)
    stack = unfolded_stack(4, seed=31)
    stack[:, 2] = stack[:, 0]  # the same transit in two different slots
    batch["unfolded_view"] = np.repeat(stack, 4, axis=0)

    encoded = probe_of(model, "unfolded_view_td")(batch, training=False).numpy()
    np.testing.assert_allclose(encoded[:, 0], encoded[:, 2], rtol=1e-6, atol=1e-6)


def test_the_tower_reads_phase_and_the_pool_ignores_transit_order(model):
    """The two halves of finding #23, stated as properties.

    Reordering *transits* must not change the pooled statistics — they are
    summaries of a set. Reordering the *phase bins inside* a transit must
    change the embedding, because that is the transit shape. The old
    implementation had neither property the right way round.
    """
    batch = make_batch(seed=32)
    batch["unfolded_view"] = unfolded_stack(TRANSITS, seed=32, n=4)
    pool = probe_of(model, "unfolded_view_pool")
    before = pool(batch, training=False).numpy()

    shuffled = dict(batch)
    shuffled["unfolded_view"] = batch["unfolded_view"][:, ::-1]
    np.testing.assert_allclose(before, pool(shuffled, training=False).numpy(), rtol=1e-5, atol=1e-5)

    rephased = dict(batch)
    rephased["unfolded_view"] = batch["unfolded_view"][:, :, ::-1]
    assert not np.allclose(before, pool(rephased, training=False).numpy(), atol=1e-5)


def test_padded_slots_do_not_dilute_the_pool(model):
    """`_unfolded` zero-fills unreached slots and 30.4% of the training set
    carries at least one. An unmasked mean divides by twenty regardless, so the
    branch output would scale with occupancy — and occupancy tracks the label
    (K2: 12.4 filled slots on planet hosts against 17.2 on false positives)."""
    one = unfolded_stack(1, seed=33)
    ten = np.repeat(one[:, :1], 10, axis=1)
    ten = np.concatenate([ten, np.zeros_like(one[:, 10:])], axis=1)

    batch = make_batch(n=1, seed=33)
    pool = probe_of(model, "unfolded_view_pool")
    batch["unfolded_view"] = one
    with_one = pool(batch, training=False).numpy()
    batch["unfolded_view"] = ten
    np.testing.assert_allclose(with_one, pool(batch, training=False).numpy(), rtol=1e-5, atol=1e-5)


def test_the_pooled_mean_is_the_mean_over_measured_slots_only(model):
    batch = make_batch(n=1, seed=34)
    batch["unfolded_view"] = unfolded_stack(3, seed=34)
    encoded = probe_of(model, "unfolded_view_td")(batch, training=False).numpy()
    pooled = probe_of(model, "unfolded_view_pool")(batch, training=False).numpy()

    width = encoded.shape[-1]
    np.testing.assert_allclose(pooled[:, :width], encoded[:, :3].mean(axis=1), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(pooled[:, width : 2 * width], encoded[:, :3].max(axis=1), atol=1e-5)


def test_the_spread_separates_identical_transits_from_varying_ones(model):
    """The statistic the branch exists for: an eclipsing binary and a
    background blend vary transit to transit, a planet does not. A mean alone
    returns the folded view again, which is what this branch is meant to add to."""
    width = embedding_width(model)
    pool = probe_of(model, "unfolded_view_pool")
    batch = make_batch(n=1, seed=35)

    same = unfolded_stack(1, seed=35)
    same[:, :6] = same[:, :1]
    batch["unfolded_view"] = same
    flat = pool(batch, training=False).numpy()[:, 2 * width :]

    batch["unfolded_view"] = unfolded_stack(6, seed=36)
    varying = pool(batch, training=False).numpy()[:, 2 * width :]

    assert flat.max() == pytest.approx(np.sqrt(SPREAD_EPSILON), abs=1e-5)
    assert varying.max() > flat.max()


@pytest.mark.parametrize(("measured", "expected"), [(0, 0.0), (1, 0.0), (2, 1.0), (9, 1.0)])
def test_a_spread_from_one_transit_is_flagged_unmeasured(model, measured, expected):
    """Zero spread from one transit and zero spread from twenty identical ones
    are the same float with opposite meanings, and `observed_transit_count` is
    uncapped so the head cannot recover the distinction from it."""
    batch = make_batch(n=1, seed=37)
    batch["unfolded_view"] = unfolded_stack(measured, seed=37)
    flag = probe_of(model, "unfolded_view_spread_measurable")(batch, training=False).numpy()
    assert flag.item() == expected


def test_an_empty_stack_pools_to_finite_numbers(model):
    """The branch is gated downstream and a gate multiplies, so a NaN here
    would not be zeroed — it would propagate into fusion for every row of a
    mission. Five training rows have no measured transit at all."""
    batch = make_batch(n=2, seed=38)
    batch["unfolded_view"] = unfolded_stack(0, seed=38, n=2)
    pooled = probe_of(model, "unfolded_view_pool")(batch, training=False).numpy()
    assert np.isfinite(pooled).all()
    assert np.isfinite(model(batch, training=False).numpy()).all()


def pool_gradients(epsilon: float, *, width: int = 8) -> list:
    """Gradients through the pool on a batch whose masked variance is exactly zero.

    A trainable layer sits *before* the pool deliberately: with only a head
    above it, every gradient would stop at the pooled values — which are finite
    either way — and the test could not fail.
    """
    rng = np.random.default_rng(39)
    encoded = rng.normal(size=(4, TRANSITS, width)).astype(np.float32)
    measured = np.zeros((4, TRANSITS), dtype=np.float32)
    measured[:, 0] = 1.0  # one measured slot, so every deviation is zero

    encoded_in = tf.keras.layers.Input(shape=(TRANSITS, width))
    measured_in = tf.keras.layers.Input(shape=(TRANSITS,))
    hidden = tf.keras.layers.Dense(width)(encoded_in)
    pooled = MaskedTransitPool(epsilon=epsilon)([hidden, measured_in])
    model = tf.keras.Model([encoded_in, measured_in], tf.keras.layers.Dense(1)(pooled))

    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(model([encoded, measured], training=True))
    return [g for g in tape.gradient(loss, model.trainable_variables) if g is not None]


def test_a_single_measured_transit_gives_finite_gradients():
    """`sqrt` has an infinite derivative at zero, and a masked variance is
    exactly zero whenever one slot is measured — 17 rows of the training set.
    Without the epsilon the fold dies on a NaN gradient rather than the batch.

    The second assertion is the point: it shows the guard is load-bearing. A
    constant that cannot be observed to matter is one nobody can safely remove.
    """
    assert all(np.isfinite(g.numpy()).all() for g in pool_gradients(SPREAD_EPSILON))
    assert not all(np.isfinite(g.numpy()).all() for g in pool_gradients(0.0))


def test_every_branch_scalar_name_exists_in_the_feature_vector():
    # A typo here silently drops a scalar from its branch rather than failing.
    for names in BRANCH_SCALARS.values():
        for name in names:
            assert name in SCALARS, f"{name} is not a shard feature column"


def test_model_trains_one_step(model):
    model.compile(optimizer="adam", loss="binary_crossentropy")
    batch = make_batch(n=8)
    labels = np.array([0, 1] * 4, dtype=np.float32)
    before = model.evaluate(batch, labels, verbose=0)
    model.fit(batch, labels, epochs=3, verbose=0)
    assert model.evaluate(batch, labels, verbose=0) < before


# ------------------------------------- stage 7: dropping a branch, declared --


def test_the_families_partition_every_branch():
    """`BRANCH_FAMILIES` is the unit stage 7's leave-one-out runs drop. A branch
    in no family would never be attributed; a branch in two would be dropped
    twice and its delta double-counted. Both are silent, so this is the check."""
    covered = [name for names in BRANCH_FAMILIES.values() for name in names]
    assert sorted(covered) == sorted(set(covered)), "a branch appears in two families"
    assert set(covered) == set(BRANCH_NAMES)


def test_every_named_branch_is_actually_built(model):
    """`BRANCH_NAMES` is derived from the constants, but "derived" is not
    "present" — a name that no layer answers to would make an ablation a no-op
    that still trains to a plausible AUC.

    Asserted on the branch's `_fc` head, not on a name prefix: a view branch's
    `Input` is named after the view and is retained even when the branch is
    dropped, so a prefix check passes on the input alone and proves nothing."""
    built = {layer.name for layer in model.layers}
    for branch in BRANCH_NAMES:
        assert f"{branch}_fc" in built, f"{branch} builds no head"


@pytest.mark.parametrize("family", sorted(BRANCH_FAMILIES))
def test_dropping_a_family_removes_its_branches_and_keeps_the_input_signature(family):
    """The shard stream always yields every view in `VIEW_SHAPES`, so an ablation that also
    changed the input contract would not be a controlled comparison — the
    dropped branch's `Input` stays, unused."""
    full = build_cnn_branches(SimpleNamespace(), scalar_columns=SCALARS, mask_columns=MASKS)
    ablated = build_cnn_branches(
        SimpleNamespace(drop_branches=[family]), scalar_columns=SCALARS, mask_columns=MASKS
    )
    assert {t.name.split(":")[0] for t in ablated.inputs} == {
        t.name.split(":")[0] for t in full.inputs
    }
    assert ablated.count_params() < full.count_params(), "dropping cost no parameters"

    dropped = resolve_dropped_branches([family])
    fused = next(layer for layer in ablated.layers if layer.name == "fusion")
    assert len(fused.input) == len(
        next(layer for layer in full.layers if layer.name == "fusion").input
    ) - len(dropped)

    out = ablated(make_batch())
    assert np.all(np.isfinite(out.numpy()))


def test_an_unrecognised_branch_raises_rather_than_dropping_nothing():
    """An ablation that quietly ablated nothing reports a delta of zero, which
    reads as "this branch does not matter" — the opposite of what happened."""
    with pytest.raises(ValueError, match="unknown branch or family"):
        resolve_dropped_branches(["periodogram_veiw"])
    with pytest.raises(ValueError, match="unknown branch or family"):
        build_cnn_branches(
            SimpleNamespace(drop_branches=["not_a_branch"]),
            scalar_columns=SCALARS,
            mask_columns=MASKS,
        )


def test_a_bare_scalar_spec_raises():
    """`drop_branches: 3` in the YAML is a typo, not an instruction. Without
    this it reaches the resolver as something that is neither a name nor a
    sequence of them."""
    with pytest.raises(ValueError, match="must be a name or a list of names"):
        resolve_dropped_branches(3)


def test_dropping_every_branch_raises():
    """With nothing left, the model scores every row from the presence masks
    alone and still returns a probability."""
    with pytest.raises(ValueError, match="no branch feeding fusion"):
        resolve_dropped_branches(sorted(BRANCH_FAMILIES))


def test_no_drop_is_the_main_line(model):
    """The default has to be untouched, or every run since stage 4 is ablated."""
    assert resolve_dropped_branches(()) == frozenset()
    assert resolve_dropped_branches(None) == frozenset()
    declared = build_cnn_branches(
        SimpleNamespace(drop_branches=[]), scalar_columns=SCALARS, mask_columns=MASKS
    )
    assert declared.count_params() == model.count_params()


def test_dropping_the_flux_family_removes_the_shared_tower_too():
    """The tower exists only to serve that family; leaving it built would make
    the ablation cost parameters it does not use and muddy a capacity reading."""
    ablated = build_cnn_branches(
        SimpleNamespace(drop_branches=["flux"]), scalar_columns=SCALARS, mask_columns=MASKS
    )
    assert not any(layer.name.startswith("local_shared") for layer in ablated.layers)
    assert np.all(np.isfinite(ablated(make_batch()).numpy()))


def test_a_single_branch_can_be_dropped_by_name():
    """Families are the experiment's unit, but the mechanism is not limited to
    them — a follow-up that suspects one half of a pair should not need a code
    edit either."""
    ablated = build_cnn_branches(
        SimpleNamespace(drop_branches=["gap_view"]), scalar_columns=SCALARS, mask_columns=MASKS
    )
    built = {layer.name for layer in ablated.layers}
    assert "gap_view_fc" not in built
    assert "periodogram_view_fc" in built
    # The input survives the branch: the shard stream still yields `gap_view`.
    assert "gap_view" in {t.name.split(":")[0] for t in ablated.inputs}


def test_the_dropped_set_is_recorded_in_the_resolved_config():
    """`run_config.model_config` is what makes the ablation recoverable from the
    artefact. Stage 4 produced four runs whose architecture was not."""
    from exoplanet_hunter.training.train_branches import _resolved_model_config

    cfg = SimpleNamespace(drop_branches=["periodogram"], init_filters=16)
    assert _resolved_model_config(cfg)["drop_branches"] == ["periodogram"]


def test_dropping_the_contrast_keeps_its_two_halves_measurable():
    """`odd_even` is the pair's *difference*; dropping it must not take
    `local_view` or `secondary_view` with it, since they share the tower."""
    ablated = build_cnn_branches(
        SimpleNamespace(drop_branches=[CONTRAST_BRANCH]),
        scalar_columns=SCALARS,
        mask_columns=MASKS,
    )
    assert not any(layer.name.startswith("odd_even") for layer in ablated.layers)
    assert any(layer.name.startswith("local_shared") for layer in ablated.layers)
    assert np.all(np.isfinite(ablated(make_batch()).numpy()))


# --------------------------------------------------------- difference images

DIFF_VIEW = "difference_view"
QUALITY_VIEW = "difference_quality_view"


def diff_batch(n: int = 4, *, sectors: int = 0, seed: int = 0, quality: float = 0.8) -> dict:
    """A batch whose difference branch has exactly `sectors` measured slots.

    Every other view stays present, so anything this moves is attributable to
    the difference branch rather than to a neighbour going absent.
    """
    rng = np.random.default_rng(seed)
    batch = make_batch(n, seed=seed)
    batch[DIFF_VIEW] = np.zeros_like(batch[DIFF_VIEW])
    batch[QUALITY_VIEW] = np.zeros_like(batch[QUALITY_VIEW])
    grid = batch[DIFF_VIEW].shape[2]
    edge = (grid - 11) // 2
    for slot in range(sectors):
        batch[DIFF_VIEW][:, slot, edge : edge + 11, edge : edge + 11, 0] = rng.normal(
            0, 0.05, (n, 11, 11)
        )
        # The target marker sits at the middle of the stamp; presence is the
        # LAST channel, which is what `SectorPresence` reads. Indexing either by
        # a hard-coded 2 would silently mark the slot absent.
        batch[DIFF_VIEW][:, slot, grid // 2, grid // 2, TARGET_CHANNEL] = 1.0
        batch[DIFF_VIEW][:, slot, edge : edge + 11, edge : edge + 11, -1] = 1.0
        batch[QUALITY_VIEW][:, slot] = (quality, 1.0)
    return batch


def test_a_target_with_no_difference_image_gives_finite_predictions(model):
    """58.9% of the set is in exactly this state — every Kepler and K2 row plus
    6.8% of TESS. A masked softmax written the textbook way returns NaN when
    every slot is masked, and a NaN reaching the gate multiplies to NaN rather
    than to nothing."""
    out = model(diff_batch(sectors=0), training=False).numpy()
    assert np.isfinite(out).all()
    probe = probe_of(model, f"{DIFF_VIEW}_pool")
    assert np.isfinite(probe(diff_batch(sectors=0), training=False).numpy()).all()


def test_the_pool_is_exactly_zero_when_no_sector_is_measured(model):
    pooled = probe_of(model, f"{DIFF_VIEW}_pool")(diff_batch(sectors=0), training=False).numpy()
    assert np.abs(pooled).sum() == pytest.approx(0.0)


def test_padded_sectors_do_not_dilute_the_pool(model):
    """The number of measured sectors is how many times TESS looked at the star,
    so a pool that averaged over the padding too would make this branch's output
    scale with observation baseline — the confound stage 8 exists to remove."""
    pool = probe_of(model, f"{DIFF_VIEW}_pool")
    one = diff_batch(sectors=1, seed=3)
    # Same single measured sector, but the slot count the pool could divide by
    # changes only in the padding.
    padded = {k: v.copy() for k, v in one.items()}
    assert np.array_equal(padded[DIFF_VIEW][:, 1:], np.zeros_like(padded[DIFF_VIEW][:, 1:]))
    assert np.allclose(pool(one, training=False).numpy(), pool(padded, training=False).numpy())


def test_the_pool_does_not_depend_on_the_order_sectors_arrive_in(model):
    """Slot order is the order `build_difference_views` happened to sort in.
    Attention is a weighted sum, so the pooled vector must be invariant to it —
    if it is not, the branch is reading slot index as a feature."""
    pool = probe_of(model, f"{DIFF_VIEW}_pool")
    batch = diff_batch(sectors=3, seed=5)
    shuffled = {k: v.copy() for k, v in batch.items()}
    order = [2, 0, 1, *range(3, batch[DIFF_VIEW].shape[1])]
    shuffled[DIFF_VIEW] = batch[DIFF_VIEW][:, order]
    shuffled[QUALITY_VIEW] = batch[QUALITY_VIEW][:, order]
    assert np.allclose(
        pool(batch, training=False).numpy(),
        pool(shuffled, training=False).numpy(),
        atol=1e-5,
    )


def test_the_quality_metric_changes_how_sectors_are_weighted(model):
    """The attention's reason for existing. Two identical stamps weighted
    differently only because DV rates them differently."""
    pool = probe_of(model, f"{DIFF_VIEW}_pool")
    low = diff_batch(sectors=2, seed=7, quality=0.1)
    high = {k: v.copy() for k, v in low.items()}
    # Same stamps, one sector's quality raised.
    high[QUALITY_VIEW] = low[QUALITY_VIEW].copy()
    high[QUALITY_VIEW][:, 0, 0] = 0.99
    assert not np.allclose(pool(low, training=False).numpy(), pool(high, training=False).numpy())


def test_a_declined_sector_and_a_flat_one_are_not_the_same_input(model):
    """DV writes a sector it declined to measure as zeros, and a stamp measured
    flat is also zeros. The presence channel is the only thing separating them,
    and the branch has to act on it."""
    pool = probe_of(model, f"{DIFF_VIEW}_pool")
    flat = diff_batch(sectors=1, seed=9)
    flat[DIFF_VIEW][..., 0] = 0.0  # measured, and genuinely featureless
    declined = {k: v.copy() for k, v in flat.items()}
    # DV produced nothing for this sector, so the slot is all zeros: presence is
    # the LAST channel, and the target marker goes with it — a declined sector
    # has no origin either, DV writing `ticReferenceCentroid` as 0.0 with
    # uncertainty -1.0. Zeroing a fixed channel index instead would have marked
    # the slot absent right up until a channel was inserted in front of presence.
    declined[DIFF_VIEW][..., TARGET_CHANNEL] = 0.0
    declined[DIFF_VIEW][..., -1] = 0.0
    declined[QUALITY_VIEW][:] = 0.0
    assert np.abs(pool(flat, training=False).numpy()).sum() > 0
    assert np.abs(pool(declined, training=False).numpy()).sum() == pytest.approx(0.0)


def test_the_quality_view_is_not_a_branch_of_its_own(model):
    """It weights the stamps; built as a branch it would put DV's opinion of the
    data into fusion as evidence about the target."""
    assert QUALITY_VIEW not in BRANCH_NAMES
    assert f"{QUALITY_VIEW}_fc" not in {layer.name for layer in model.layers}
    # ...but it must still reach the model, or the attention has nothing to read.
    assert QUALITY_VIEW in {t.name.split(":")[0] for t in model.inputs}


def test_every_sector_is_encoded_by_the_same_weights(model):
    """One tower under `TimeDistributed`, so a difference between two sectors is
    a difference in the data rather than between two sets of kernels."""
    encoded = probe_of(model, f"{DIFF_VIEW}_td")
    batch = diff_batch(sectors=2, seed=11)
    # Put the identical stamp in both slots; their embeddings must match.
    batch[DIFF_VIEW][:, 1] = batch[DIFF_VIEW][:, 0]
    out = encoded(batch, training=False).numpy()
    assert np.allclose(out[:, 0], out[:, 1], atol=1e-5)


def test_the_attention_pool_is_a_convex_combination():
    """Weights sum to one over measured slots, so the pooled vector stays on the
    scale of a single sector's embedding however many sectors there are."""
    from exoplanet_hunter.models.cnn_branches import MaskedAttentionPool

    encoded = tf.constant([[[1.0, 0.0], [3.0, 0.0], [9.0, 0.0]]])
    logits = tf.zeros((1, 3, 1))
    measured = tf.constant([[1.0, 1.0, 0.0]])
    pooled = MaskedAttentionPool()([encoded, logits, measured]).numpy()
    # Equal logits over the two measured slots: the mean of 1 and 3, not of all
    # three and not their sum.
    assert pooled[0, 0] == pytest.approx(2.0)


def test_the_attention_pool_survives_extreme_logits():
    """The logits are learned and unbounded; exp() of a large one overflows to
    inf in float32 well before training has visibly diverged."""
    from exoplanet_hunter.models.cnn_branches import MaskedAttentionPool

    encoded = tf.ones((1, 3, 2))
    logits = tf.constant([[[500.0], [-500.0], [0.0]]])
    measured = tf.constant([[1.0, 1.0, 1.0]])
    assert np.isfinite(MaskedAttentionPool()([encoded, logits, measured]).numpy()).all()
