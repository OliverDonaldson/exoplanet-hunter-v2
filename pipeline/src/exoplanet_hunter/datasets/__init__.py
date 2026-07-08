"""tf.data input pipeline: TFRecord shards, split filtering, augmentation.

Built in `feat/tfdata-pipeline`. The deterministic work (parse, split filter,
aux normalisation) is cached; the stochastic augmentation runs fresh each
epoch after the cache, per the L6 prescription.
"""

from exoplanet_hunter.datasets.augment import AugmentConfig, augment_views
from exoplanet_hunter.datasets.aux_transform import (
    AuxConstants,
    aux_constants_from_pipeline,
    fit_aux_pipeline,
    tf_aux_transform,
)
from exoplanet_hunter.datasets.pipeline import Split, make_dataset, make_split_table
from exoplanet_hunter.datasets.tfrecords import (
    ShardMetadata,
    list_shards,
    load_index,
    make_parse_fn,
    write_tfrecord_shards,
)
from exoplanet_hunter.datasets.views_io import ViewArrays, load_views, slice_views

__all__ = [
    "AugmentConfig",
    "AuxConstants",
    "ShardMetadata",
    "Split",
    "ViewArrays",
    "augment_views",
    "aux_constants_from_pipeline",
    "fit_aux_pipeline",
    "list_shards",
    "load_index",
    "load_views",
    "make_dataset",
    "make_parse_fn",
    "make_split_table",
    "slice_views",
    "tf_aux_transform",
    "write_tfrecord_shards",
]
