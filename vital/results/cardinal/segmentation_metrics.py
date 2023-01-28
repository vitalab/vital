import itertools
from argparse import ArgumentParser
from pathlib import Path
from typing import Mapping, Tuple

import medpy.metric as metric
import numpy as np
import pandas as pd

from vital.data.cardinal.config import CardinalTag, Label
from vital.data.cardinal.utils.data_struct import View
from vital.data.cardinal.utils.itertools import Views
from vital.results.metrics import Metrics
from vital.utils.image.us.measure import EchoMeasure


class SegmentationMetrics(Metrics):
    """Class that computes segmentation metrics on the results and saves them to csv."""

    desc = "segmentation_metrics"
    ResultsCollection = Views
    ProcessingOutput = pd.DataFrame
    input_choices = []
    target_choices = []  # Enable target w/o explicitly specifying target tags

    def __init__(self, reduce_over_ed_es: bool = False, **kwargs):
        """Initializes class instance.

        Args:
            reduce_over_ed_es: Whether to compute reduction of the metrics over ED and ES frames specifically.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)

        self.labels = {"endo": Label.LV, "epi": [Label.LV, Label.MYO]}
        self.scores = {"dsc": metric.dc}
        self.distances = {"hd": metric.hd, "assd": metric.assd}

        self._reduce_over_ed_es = reduce_over_ed_es

    def process_result(self, result: View) -> Tuple[View.Id, "SegmentationMetrics.ProcessingOutput"]:
        """Computes metrics on data from a sequence.

        Args:
            result: Data structure holding all the relevant information to compute the requested metrics for a single
                sequence.

        Returns:
            - Identifier of the result for which the metrics where computed.
            - Mapping between the metrics and their values for the sequence.
        """
        input, target = result.data[self.input_tag], result.data[self.target_tag]

        metrics = {}
        for label_tag, label in self.labels.items():
            input_mask, target_mask = np.isin(input, label), np.isin(target, label)

            for metric_tag, metric_fn in itertools.chain(self.scores.items(), self.distances.items()):
                metric_kwargs = {}
                if metric_tag in self.distances:
                    # For distances, add the voxelspacing as an argument forwarded to the metric's function
                    metric_kwargs["voxelspacing"] = result.attrs[self.target_tag][CardinalTag.voxelspacing]

                metrics[f"{label_tag}_{metric_tag}"] = [
                    metric_fn(input_frame, target_frame, **metric_kwargs)
                    for input_frame, target_frame in zip(input_mask, target_mask)
                ]

        metrics = pd.DataFrame.from_dict(metrics)
        return result.id, metrics

    def aggregate_outputs(self, outputs: Mapping[View.Id, ProcessingOutput], output_path: Path) -> None:
        """Override of the parent method to concatenate existing dataframes rather than build a new one from dicts."""
        metrics = pd.concat(outputs).rename_axis(index=["patient", "view", "frame"])
        agg_metrics = self._aggregate_metrics(metrics)

        # Save the individual and aggregated scores in two steps
        agg_metrics.to_csv(output_path, na_rep="Nan")
        with output_path.open("a") as f:
            f.write("\n")
        metrics.to_csv(output_path, mode="a", na_rep="Nan")

    def _aggregate_metrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        agg_metrics = [super()._aggregate_metrics(metrics)]
        metrics_to_compute = agg_metrics[0].index.tolist()

        if self._reduce_over_ed_es:
            views = self.ResultsCollection(**self.results_collection_kwargs)

            # The ED frame is always the first
            ed_frames_mask = metrics.index.get_level_values("frame") == 0
            # Identify the ES frame in a view as the frame where the LV is the smallest (in 2D)
            es_frames_indices = [
                (*view_id, EchoMeasure.structure_area(view.data[self.target_tag], labels=Label.LV).argmin())
                for view_id, view in views.items()
            ]
            es_frames_mask = [idx_val in es_frames_indices for idx_val in metrics.index]

            # Reduce over each frame specifically, and add them to the overall aggregated metrics
            for frame, frame_mask in [["ed", ed_frames_mask], ["es", es_frames_mask]]:
                frame_agg_metrics = (
                    metrics[frame_mask]
                    .agg(metrics_to_compute)
                    .rename(index={metric_name: f"{metric_name}_{frame}" for metric_name in metrics_to_compute})
                )

                agg_metrics.append(frame_agg_metrics)

        return pd.concat(agg_metrics)

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser for Cardinal segmentation metrics processor.

        Returns:
            Parser object for Cardinal segmentation metrics processor.
        """
        parser = super().build_parser()
        parser.add_argument(
            "--reduce_over_ed_es",
            action="store_true",
            help="Whether to compute reduction of the metrics over ED and ES frames specifically",
        )
        return parser


def main():
    """Run the script."""
    SegmentationMetrics.main()


if __name__ == "__main__":
    main()
