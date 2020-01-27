from typing import Union, Dict

import pandas as pd

from vital.logs.metrics import MetricsLogger
from vital.metrics.evaluate.segmentation import check_metric_validity
from vital.utils.config import ResultTags


class AnatomicalMetricsLogger(MetricsLogger):
    """Abstract class that computes anatomical metrics on the results and saves them to csv."""
    name = 'anatomical_metrics'
    data_choices = [ResultTags.post_pred, ResultTags.pred, ResultTags.gt]
    thresholds: Dict[str, Dict[str, Union[int, float]]]  # anatomical metrics' threshold values

    @classmethod
    def _aggregate_metrics(cls, metrics: pd.DataFrame) -> pd.DataFrame:
        """ Computes the number of results that were anatomically invalid for each anatomical metric.

        Args:
            metrics: metrics computed over each result.

        Returns:
            the number of results that were anatomically invalid for each anatomical metric.
        """

        def count_metric_errors(metric_name):
            return lambda series: sum(not check_metric_validity(metric_value, cls.thresholds.get(metric_name),
                                                                optional_structure=False)
                                      for metric_value in series)

        aggregation_dict = {metric_name: count_metric_errors(metric_name) for metric_name in metrics.keys()}
        aggregated_metrics = pd.DataFrame(metrics.agg(aggregation_dict)).T
        aggregated_metrics.name = 'anatomical_errors_count'  # The series' name will be its index in the dataframe
        return aggregated_metrics
