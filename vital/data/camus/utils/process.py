from pathlib import Path
from typing import Sequence, Union

from vital import get_vital_home
from vital.utils.image.process.autoencoder import ConstrainedAutoencoderSnake


class TEDTemporalRegularization(ConstrainedAutoencoderSnake):
    """Wrapper around `ConstrainedAutoencoderSnake` to set some default configs specific to the CAMUS/TED dataset."""

    def __init__(
        self,
        data_labels: Sequence[int] = (0, 1, 2),
        attr_thresholds: Union[str, Path] = get_vital_home() / "data/camus/statistics/attr_thresholds.yaml",
        **kwargs
    ):
        """Initializes class instance.

        Args:
            data_labels: Defaults to assuming the labels to process are the background (0), left ventricle (1) and
                myocardium (2).
            attr_thresholds: Defaults to the thresholds determined on the CAMUS dataset by Painchaud et al.
                (https://arxiv.org/abs/2112.02102).
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(data_labels=data_labels, attr_thresholds=attr_thresholds, **kwargs)
