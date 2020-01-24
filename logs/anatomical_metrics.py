from typing import Union, Dict

import pandas as pd

from vital.logs.metrics import MetricsLogger
from vital.metrics.evaluate.segmentation import check_metric_validity
from vital.utils.config import ResultTags


class AnatomicalMetricsLogger(MetricsLogger):
    """Class that computes anatomical metrics on the results and saves them to csv."""
    name = 'anatomical_metrics'
    data_choices = [ResultTags.post_pred, ResultTags.pred, ResultTags.gt]
    thresholds: Dict[str, Dict[str, Union[int, float]]]

    @classmethod
    def _aggregate_metrics(cls, metrics: pd.DataFrame) -> pd.DataFrame:
        """

        Args:
            metrics:

        Returns:

        """
        # Aggregate the metrics
        def count_metric_errors(metric_name):
            return lambda series: sum(not check_metric_validity(metric_value, cls.thresholds.get(metric_name),
                                                                optional_structure=False)
                                      for metric_value in series)

        aggregation_dict = {metric_name: count_metric_errors(metric_name) for metric_name in metrics.keys()}
        aggregated_metrics = pd.DataFrame(metrics.agg(aggregation_dict)).T
        aggregated_metrics.name = 'anatomical_errors_count'  # The series' name will be its index in the dataframe
        return aggregated_metrics
