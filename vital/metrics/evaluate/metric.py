from numbers import Real
from statistics import mean
from typing import Callable, Sequence

import numpy as np
from medpy.metric import assd, dc, hd, jc, precision, recall
from sklearn.metrics import accuracy_score

from vital.data.config import SemanticStructureId
from vital.utils.delegate import delegate_inheritance


class Metric:
    """Abstract class for metrics that are to be computed on the results of a model."""

    def __init__(
        self, metric_fn: Callable, desc: str, struct_labels: Sequence[SemanticStructureId]
    ):  # noqa: D205,D212,D415
        """
        Args:
            metric_fn: Function that computes the metric for a pair of binary result/reference images.
            desc: Name of the metric.
            struct_labels: Labels of the structures on which to compute the metric.
        """
        self.metric_fn = metric_fn
        self.desc = desc
        self.struct_labels = struct_labels

    def __call__(self, result: np.ndarray, reference: np.ndarray, **metric_kwargs) -> Sequence[Real]:
        """Computes a metric between a result and a reference.

        Args:
            result: Predicted result for which to compute the metric.
            reference: Reference against which to compare the result.
            **metric_kwargs: Additional parameters (apart from result/reference) required to compute the metric.

        Returns:
            Value of the metric for the result/reference pair.
        """
        # Compute results for all the structures
        metrics = [
            self.call_metric_wrapper(np.isin(result, struct_label), np.isin(reference, struct_label), **metric_kwargs)
            for struct_label in self.struct_labels
        ]
        return metrics

    def call_metric_wrapper(self, result, reference, **metric_kwargs) -> Real:
        """Computes a metric between result and reference binary masks.

        This wrapper makes a unified interface for metrics possible, by propagating additional parameters necessary for
        some types of metrics (e.g. voxelspacing for distance metrics) through ``metric_params``, while also allowing
        to capture and discard unnecessary parameters.

        Args:
            result: Predicted binary mask for which to compute the metric.
            reference: Reference binary mask against which to compare the result.
            **metric_kwargs: Additional parameters (apart from result/reference) required to compute the metric.

        Returns:
            Value of the metric for the result/reference pair.
        """
        raise NotImplementedError


@delegate_inheritance()
class Distance(Metric):
    """Abstract class for distance metrics."""

    def __init__(
        self, distance_fn: Callable[[np.ndarray, np.ndarray, Sequence[float]], Real], **kwargs
    ):  # noqa: D205,D212,D415
        """
        Args:
            distance_fn: Function that measures a distance between a result/reference pair, based on the voxelspacing.
        """
        super().__init__(metric_fn=distance_fn, **kwargs)

    def call_metric_wrapper(self, result, reference, voxelspacing=None, **metric_kwargs):  # noqa: D102
        distance = np.NaN
        if np.any(result) and np.any(reference):
            distance = self.metric_fn(result, reference, voxelspacing=voxelspacing)
        return distance


@delegate_inheritance()
class Hausdorff(Distance):
    """Hausdorff distance class."""

    def __init__(self, desc="hausdorff", **kwargs):
        super().__init__(hd, desc=desc, **kwargs)


@delegate_inheritance()
class Assd(Distance):
    """Average Symmetric Surface Distance (ASSD) class.

    Notes:
        - This metric is also known as the Median Absolute Deviation (MAD) score.
    """

    def __init__(self, desc="assd", **kwargs):
        super().__init__(assd, desc=desc, **kwargs)


@delegate_inheritance()
class Score(Metric):
    """Abstract class for score metrics."""

    def __init__(self, score_fn: Callable[[np.ndarray, np.ndarray], Real], **kwargs):  # noqa: D205,D212,D415
        """
        Args:
            score_fn: Function that evaluates a score on the result/reference pair.
        """
        super().__init__(metric_fn=score_fn, **kwargs)

    def call_metric_wrapper(self, result, reference, **kwargs):  # noqa: D102
        return self.metric_fn(result, reference)


@delegate_inheritance()
class Precision(Score):
    """Precision score class."""

    def __init__(self, desc="precision", **kwargs):
        super().__init__(precision, desc=desc, **kwargs)


@delegate_inheritance()
class Recall(Score):
    """Recall score class."""

    def __init__(self, desc="recall", **kwargs):
        super().__init__(recall, desc=desc, **kwargs)


@delegate_inheritance()
class Dice(Score):
    """Dice score class."""

    def __init__(self, desc="dice", **kwargs):
        super().__init__(dc, desc=desc, **kwargs)


@delegate_inheritance()
class Jaccard(Score):
    """Jaccard score class."""

    def __init__(self, desc="jaccard", **kwargs):
        super().__init__(jc, desc=desc, **kwargs)


@delegate_inheritance()
class Accuracy(Score):
    """Accuracy score class."""

    def __init__(self, desc="accuracy", **kwargs):
        super().__init__(self._accuracy, desc=desc, **kwargs)

    @staticmethod
    def _accuracy(result: np.ndarray, reference: np.ndarray) -> float:
        """Computes the accuracy score of the segmentation.

        Args:
            result: Predicted binary mask for which to compute the accuracy.
            reference: Reference binary mask against which to compare the result.

        Returns:
            Accuracy of the predicted binary mask.
        """
        acc = 0.0
        if reference.ndim == 3:
            acc = mean(accuracy_score(reference_i, result_i) for reference_i, result_i in zip(reference, result))
        if reference.ndim == 2:
            acc = accuracy_score(reference, result)
        return acc
