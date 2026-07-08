"""Mixed-precision policy control for the GPU burst.

mixed_float16 gives 2-3x on Tensor Cores, which directly cuts per-run rental
cost; Keras' LossScaleOptimizer (applied automatically under the policy)
guards against underflow — relevant given the documented loss=NaN history
with unscaled inputs. The model's output layer is pinned to float32 in
`cnn_dualview.py`, keeping the calibration-critical sigmoid tails exact.

Off by default: on CPU (and Apple Metal) mixed_float16 is at best neutral.
Enable via `train.mixed_precision=true` on the burst, where the money is.
"""

from __future__ import annotations

from tensorflow.keras import mixed_precision

from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)


def apply_precision_policy(enable_mixed: bool) -> str:
    """Set the global Keras dtype policy; returns the policy name applied."""
    policy = "mixed_float16" if enable_mixed else "float32"
    mixed_precision.set_global_policy(policy)
    log.info("[precision] global policy = %s", policy)
    return policy
