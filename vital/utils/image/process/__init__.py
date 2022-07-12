from abc import abstractmethod
from typing import Any, Dict, Union

import numpy as np


class PostProcessor:
    """Base class for handling the post-processing of batch of images."""

    post_tag: str  # If `__call__` returns a dict, the key under which the post-processed batch is stored.

    @abstractmethod
    def __call__(self, batch: np.ndarray) -> Union[np.ndarray, Dict[str, Any]]:
        """Processes a batch of images.

        Args:
            batch: (N, H, W), Batch of 2D images to post-process.

        Returns:
            (N, H, W), Processed batch of 2D images maps
            or
            Dict containing at least the processed batch of images under the `self.post_tag` key.
        """
