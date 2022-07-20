from argparse import ArgumentParser
from typing import Sequence, Tuple

import numpy as np

from vital.data.camus.config import CamusTags, Label
from vital.data.config import ProtoLabel
from vital.metrics.camus.anatomical import config
from vital.metrics.camus.anatomical.utils import compute_anatomical_metrics_by_segmentation
from vital.results import anatomical_metrics
from vital.results.camus import CamusResultsProcessor
from vital.results.camus.utils.data_struct import InstantResult
from vital.results.camus.utils.itertools import PatientViewInstants
from vital.utils.image.transform import resize_image


class AnatomicalMetrics(anatomical_metrics.AnatomicalMetrics):
    """Class that computes anatomical metrics on the results and saves them to csv."""

    ResultsCollection = PatientViewInstants
    input_choices = [
        f"{CamusTags.pred}/{CamusTags.raw}",
        f"{CamusTags.pred}/{CamusTags.post}",
        f"{CamusTags.gt}/{CamusTags.raw}",
    ]
    thresholds = config.thresholds

    def __init__(self, labels: Sequence[ProtoLabel], shape: Tuple[int, int] = None, **kwargs):
        """Initializes class instance.

        Args:
            labels: Labels of the classes included in the segmentations.
            shape: Dimensions (height, width) in which to resize the segmentations before evaluating the anatomical
                metrics. If ``None``, perform no resize before evaluating the metrics.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)
        self.shape = shape

        # Make sure labels are defined using the enum
        self.labels = Label.from_proto_labels(labels)

    def process_result(self, result: InstantResult) -> Tuple[str, "AnatomicalMetrics.ProcessingOutput"]:
        """Computes anatomical metrics on data from an instant.

        Args:
            result: Data structure holding all the relevant information to compute the requested metrics for a single
                instant.

        Returns:
            - Identifier of the result for which the metrics where computed.
            - Mapping between the metrics and their value for the instant.
        """
        segmentation, voxelspacing = result[self.input_tag].data, result.voxelspacing
        if self.shape is not None:
            voxelspacing = tuple((np.array(voxelspacing) * np.array(segmentation.shape)) / np.array(self.shape))
            segmentation = resize_image(segmentation, self.shape[::-1])
        return result.id, compute_anatomical_metrics_by_segmentation(segmentation, voxelspacing, labels=self.labels)

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser for CAMUS anatomical metrics processor.

        Returns:
            Parser object for CAMUS anatomical metrics processor.
        """
        parser = super().build_parser()
        parser = CamusResultsProcessor.add_labels_args(parser)
        return parser


def main():
    """Run the script."""
    AnatomicalMetrics.main()


if __name__ == "__main__":
    main()
