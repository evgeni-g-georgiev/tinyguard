"""Identity separator — pass-through baseline.

Returns the L2 norm of the input as the anomaly score. Useful for testing
the lockstep pipeline end-to-end before plugging in real separators.
"""

import numpy as np

from simulation.registry import register_separator
from simulation.inference_models.base_on_device_separator import BaseOnDeviceSeparator


@register_separator("identity")
class IdentitySeparator(BaseOnDeviceSeparator):

    def calibrate(self, warmup_embeddings: list[np.ndarray]) -> None:
        pass

    def score(self, clip_embedding: np.ndarray) -> float:
        return float(np.linalg.norm(clip_embedding))

    def get_shareable_state(self) -> dict:
        return {}

    def merge_state(self, neighbour_states: list[dict]) -> None:
        pass