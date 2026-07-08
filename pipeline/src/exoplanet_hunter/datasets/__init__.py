"""tf.data input pipeline — built in `feat/tfdata-pipeline`.

Planned contents:
  * TFRecord serialization of (global_view, local_view, aux_features, label)
    examples, sharded for streaming from R2.
  * A `make_dataset()` factory: map → cache → shuffle → batch →
    prefetch(AUTOTUNE), with cache() placed after the deterministic
    clean/flatten/fold steps and *before* stochastic augmentation so noise,
    phase shifts, depth scaling, and bin masking run fresh each epoch.
  * Augmentation ops ported from the V1 trainer as pure tf functions.
"""
