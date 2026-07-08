"""Model training and post-hoc calibration.

V2 seeds this package with calibration only. The V1 trainer fed NumPy arrays
held in RAM to `model.fit` via a data_module — deliberately not ported. The
`feat/tfdata-pipeline` branch rewrites training around `tf.data` + TFRecord
shards with mixed precision, and `feat/validation-gates` adds the
beats-current-best promotion gate.
"""

from exoplanet_hunter.training.calibration import TemperatureScaler, fit_temperature

__all__ = ["TemperatureScaler", "fit_temperature"]
