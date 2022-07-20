from argparse import ArgumentParser
from typing import Sequence, Tuple

import medpy.metric as metric
import numpy as np

from vital.data.camus.config import CamusTags, Label
from vital.data.config import ProtoLabel
from vital.results.camus import CamusResultsProcessor
from vital.results.camus.utils.data_struct import InstantResult
from vital.results.camus.utils.itertools import PatientViewInstants
from vital.results.metrics import Metrics


class SegmentationMetrics(Metrics):
    """Class that computes segmentation metrics on the results and saves them to csv."""

    desc = "segmentation_metrics"
    input_choices = [f"{CamusTags.pred}/{CamusTags.raw}", f"{CamusTags.pred}/{CamusTags.post}"]
    target_choices = [f"{CamusTags.gt}/{CamusTags.raw}"]
    ResultsCollection = PatientViewInstants

    def __init__(self, labels: Sequence[ProtoLabel], **kwargs):
        """Initializes class instance.

        Args:
            labels: Labels of the classes included in the segmentations.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)

        # Make sure labels are defined using the enums
        self.labels = {str(label): label for label in Label.from_proto_labels(labels)}

        # Exclude background from the computation of the scores
        self.labels.pop(str(Label.BG), None)

        # In the case of the myocardium (EPI) we want to calculate metrics for the entire epicardium
        # Therefore we concatenate ENDO (lumen) and EPI (myocardium)
        if str(Label.LV) in self.labels and str(Label.MYO) in self.labels:
            self.labels.pop(str(Label.MYO))
            self.labels["epi"] = (Label.LV, Label.MYO)

        self.scores = {"dsc": metric.dc}
        self.distances = {"hd": metric.hd, "assd": metric.assd}

    def process_result(self, result: InstantResult) -> Tuple[str, "SegmentationMetrics.ProcessingOutput"]:
        """Computes metrics on data from an instant.

        Args:
            result: Data structure holding all the relevant information to compute the requested metrics for a single
                instant.

        Returns:
            - Identifier of the result for which the metrics where computed.
            - Mapping between the metrics and their value for the instant.
        """
        pred, gt, voxelspacing = result[self.input_tag].data, result[self.target_tag].data, result.voxelspacing

        metrics = {}
        for label_tag, label in self.labels.items():
            pred_mask, gt_mask = np.isin(pred, label), np.isin(gt, label)

            # Compute the reconstruction accuracy metrics
            metrics.update(
                {f"{label_tag}_{score}": score_fn(pred_mask, gt_mask) for score, score_fn in self.scores.items()}
            )

            # Compute the distance metrics (that require the images' voxelspacing)
            if np.any(pred_mask) and np.any(gt_mask):
                # Only compute the distance if the requested label is present in both result and reference
                metrics.update(
                    {
                        f"{label_tag}_{dist}": dist_fn(pred_mask, gt_mask, voxelspacing=voxelspacing)
                        for dist, dist_fn in self.distances.items()
                    }
                )
            else:
                # Otherwise, mark distances as NaN for this item
                metrics.update({f"{label_tag}_{distance}": np.NaN for distance in self.distances})

        return result.id, metrics

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser for CAMUS metrics processor.

        Returns:
            Parser object for CAMUS metrics processor.
        """
        parser = super().build_parser()
        parser = CamusResultsProcessor.add_labels_args(parser)
        return parser


def main():
    """Run the script."""
    SegmentationMetrics.main()


if __name__ == "__main__":
    main()
