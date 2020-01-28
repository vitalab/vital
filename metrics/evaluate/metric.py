"""
Schematic view of the class heritage:

Metric
    |  Distance
    |     | Hausdorff
    |     | Assd
    |  Score
    |     | Precision
    |     | Recall
    |     | Dice
    |     | Jaccard
    |     | Accuracy
"""
from numbers import Real
from statistics import mean
from typing import Callable, Sequence, Union

import numpy as np
from medpy.metric import hd, assd, precision, recall, dc, jc
from sklearn.metrics import accuracy_score

from vital.utils.delegate import delegate_inheritance


class Metric:
    """Abstract class for metrics to compute on the results of a model."""

    def __init__(self, metric_fn: Callable, desc: str, struct_labels: Sequence[Union[int, Sequence[int]]]):
        """
        Args:
            desc: name of the metric.
        """
        self.metric_fn = metric_fn
        self.desc = desc
        self.struct_labels = struct_labels

    def __call__(self, result: np.ndarray, reference: np.ndarray, **metric_params) -> Sequence[Real]:
        """ Computes a metric between a result and a reference.

        Args:
            result: predicted result for which to compute the metric.
            reference: reference against which to compare the result.
            metric_params: additional parameters (apart from result/reference) required to compute the metric.

        Returns:
            value of the metric for the result/reference pair.
        """
        # Compute results for all the structures
        metrics = [self.call_metric_wrapper(np.isin(result, struct_label),
                                            np.isin(reference, struct_label),
                                            **metric_params)
                   for struct_label in self.struct_labels]
        return metrics

    def call_metric_wrapper(self, result, reference, **metric_params) -> Real:
        """ Computes a metric between result and reference binary masks.

        This wrapper makes a unified interface for metrics possible, by propagating additional parameters necessary for
        some types of metrics (e.g. voxelspacing for distance metrics) through ``metric_params``, while also allowing
        to capture and discard unnecessary parameters.

        Args:
            result: predicted result binary mask for which to compute the metric.
            reference: reference binary mask against which to compare the result.
            metric_params: additional parameters (apart from result/reference) required to compute the metric.

        Returns:
            value of the metric for the result/reference pair.
        """
        raise NotImplementedError


@delegate_inheritance()
class Distance(Metric):
    """Abstract class for distance metrics."""

    def __init__(self, distance_fn: Callable[[np.ndarray, np.ndarray, Sequence[float]], Real], **kwargs):
        """
        Args:
            distance_fn: function that measures a distance between a result/reference pair, based on the voxelspacing.
        """
        super().__init__(metric_fn=distance_fn, **kwargs)

    def call_metric_wrapper(self, result, reference, voxelspacing=None, **kwargs):
        distance = np.NaN
        if np.any(result) and np.any(reference):
            distance = self.metric_fn(result, reference, voxelspacing=voxelspacing)
        return distance


@delegate_inheritance()
class Hausdorff(Distance):
    """Hausdorff class."""

    def __init__(self, desc='hausdorff', **kwargs):
        super().__init__(hd, desc=desc, **kwargs)


@delegate_inheritance()
class Assd(Distance):
    """ Average Symmetric Surface Distance (ASSD) class.

    This metric is the same as the Median Absolute Deviation (MAD) score.
    """

    def __init__(self, desc='assd', **kwargs):
        super().__init__(assd, desc=desc, **kwargs)


@delegate_inheritance()
class Score(Metric):
    """Abstract class for score metrics."""

    def __init__(self, score_fn: Callable[[np.ndarray, np.ndarray], Real], **kwargs):
        """
        Args:
            score_fn: function that evaluates a score on the result/reference pair.
        """
        super().__init__(metric_fn=score_fn, **kwargs)
        # self._metric_fn = score_fn

    def call_metric_wrapper(self, result, reference, **kwargs):
        return self.metric_fn(result, reference)


@delegate_inheritance()
class Precision(Score):
    """Precision class."""

    def __init__(self, desc='precision', **kwargs):
        super().__init__(precision, desc=desc, **kwargs)


@delegate_inheritance()
class Recall(Score):
    """Recall class."""

    def __init__(self, desc='recall', **kwargs):
        super().__init__(recall, desc=desc, **kwargs)


@delegate_inheritance()
class Dice(Score):
    """Dice class."""

    def __init__(self, desc='dice', **kwargs):
        super().__init__(dc, desc=desc, **kwargs)


@delegate_inheritance()
class Jaccard(Score):
    """Jaccard class."""

    def __init__(self, desc='jaccard', **kwargs):
        super().__init__(jc, desc=desc, **kwargs)


@delegate_inheritance()
class Accuracy(Score):
    """Accuracy class."""

    def __init__(self, desc='accuracy', **kwargs):
        super().__init__(self._accuracy, desc=desc, **kwargs)

    @staticmethod
    def _accuracy(result: np.ndarray, reference: np.ndarray) -> float:
        """ Computes the accuracy score of the segmentation.

        Args:
            result: predicted result binary mask for which to compute the accuracy.
            reference: reference binary mask against which to compare the result.

        Returns:
            accuracy of the result.
        """
        acc = 0.0
        if reference.ndim == 3:
            acc = mean(accuracy_score(reference_i, result_i) for reference_i, result_i in zip(reference, result))
        if reference.ndim == 2:
            acc = accuracy_score(reference, result)
        return acc
