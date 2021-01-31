import math
from numbers import Real
from typing import Dict

import numpy as np


def score_metric(
    metric_value: Real,
    thresholds: Dict[str, Real] = None,
    domains: Dict[str, Real] = None,
    ideal_value: Real = 0,
    optional_structure: bool = False,
) -> Real:
    """Give a score of the metric within zero and one where one means the value respect well the threshold.

    0.5 means the value is on the threshold and zero means the value really not respect the threshold. See more docs
    in score_segmentation_docs.pdf

    Args:
        metric_value: value of the metric for a segmentation.
        thresholds: set the 0.5 score.
        domains: Value of the min and max domain of the metric.
        ideal_value: That would be the target value for that metric. Score of 1 for that.
        optional_structure: indicate whether the segmentation can be considered valid even if the metric' value
                            indicates the structure for which it was measured is absent.

    Returns:
        score the value between zero to one where one is the ideal value, 0.5 the threshold and 0 is bad.
    """

    def _gaussian_model(ideal_value, threshold, domain, metric_value):
        """Give a score between 0 and 1 base on a gaussian model.

        Args:
            ideal_value: expect the ideal value
            threshold: expect one threshold between ideal_value and the domain
            domain: expect the domain
            metric_value: metric value the method gonna calculate the score

        Returns:
            return the score
        """

        def _gaussian_equation(u, v):
            """Returns gaussian equation.

            Returns:
                return the score between 0 and 0.5
            """
            if v == u:
                return 0.5
            if u == 0:
                u = u + 0.5
                v = v + 0.5
            return (
                1
                / (0.5105 * u * math.sqrt(2 * math.pi))
                * (1 / 3 * v * v * v - u * v * v + u * u * v)
                / (2 * (0.5105 * u) * (0.5105 * u))
                * math.exp(-(v - u) * (v - u) / (2 * (0.5105 * u) * (0.5105 * u)))
            )

        score = 0
        if ideal_value == metric_value:
            return 1
        if ideal_value <= threshold <= domain:  # We have a max threshold
            if (
                ideal_value <= metric_value <= threshold
            ):  # we get 0.5 score for being better than threshold and we have the point of integral between metric_value and threshold
                if ideal_value == -np.inf:
                    # we want the integral of distance metric with the threshold in direction of an infinite terminal.
                    score = 0.5 + _gaussian_equation(threshold, threshold + (threshold - metric_value))
                else:
                    # Since gaussian equation will return the integral of ideal_value to metric value, we will do
                    # 0.5-gaussian equation to get the final score
                    score = 1 - _gaussian_equation(threshold - ideal_value, metric_value - ideal_value)
            elif threshold <= metric_value <= domain:  # We need to check if the domains is np.inf
                if domain == np.inf:
                    score = _gaussian_equation(threshold, metric_value)
                else:
                    score = _gaussian_equation(domain - threshold, domain - metric_value)

        elif domain <= threshold <= ideal_value:  # We have a max threshold
            if (
                threshold <= metric_value <= ideal_value
            ):  # we get 0.5 score for being better than threshold and we have the point of integral between metric_value and threshold
                if ideal_value == np.inf:
                    # we want the integral of distance metric with the threshold in direction of an infinite terminal.
                    score = 0.5 + _gaussian_equation(threshold, metric_value)
                else:
                    # Since gaussian equation will return the integral of ideal_value to metric value, we will do
                    # 0.5-gaussian equation to get the final score
                    score = 1 - _gaussian_equation(ideal_value - threshold, ideal_value - metric_value)
            elif domain <= metric_value <= threshold:  # We need to check if the domains is np.inf
                if domain == -np.inf:
                    score = _gaussian_equation(threshold, 2 * threshold - metric_value)
                else:
                    score = _gaussian_equation(threshold - domain, metric_value - domain)
        else:
            score = 0
        return score

    if np.isnan(metric_value):  # If the structure on which to compute the metric was not present in the segmentation
        if optional_structure:
            result = 1
        else:
            result = 0
    elif thresholds:
        if len(thresholds.keys()) == 1:
            if "min" in thresholds.keys():
                if ideal_value <= metric_value <= domains.get("max", np.inf):
                    return 1
                result = _gaussian_model(
                    ideal_value, thresholds.get("min", -0.5), domains.get("min", -np.inf), metric_value
                )
            elif "max" in thresholds.keys():
                if domains.get("min", -np.inf) <= metric_value <= ideal_value:
                    return 1
                result = _gaussian_model(
                    ideal_value, thresholds.get("max", 0.5), domains.get("max", np.inf), metric_value
                )
        elif len(thresholds.keys()) == 2:
            if metric_value <= ideal_value:
                result = _gaussian_model(
                    ideal_value, thresholds.get("min", -0.5), domains.get("min", -np.inf), metric_value
                )
            elif metric_value >= ideal_value:
                result = _gaussian_model(
                    ideal_value, thresholds.get("max", 0.5), domains.get("max", np.inf), metric_value
                )

    else:
        if metric_value <= ideal_value:
            result = _gaussian_model(ideal_value, -0.5, domains.get("min", -np.inf), metric_value)
        elif metric_value >= ideal_value:
            result = _gaussian_model(ideal_value, 0.5, domains.get("max", np.inf), metric_value)
    return result


if __name__ == "__main__":
    from argparse import ArgumentParser

    import matplotlib.pyplot as plt

    # Show the scoring normalization of the score_metric function
    args = ArgumentParser(add_help=False)
    args.add_argument("--ideal_value", default=0, type=Real)
    args.add_argument("--max_domain", default=np.inf, type=Real)
    args.add_argument("--min_domain", default=-np.inf, type=Real)
    args.add_argument("--min_threshold", default=-5, type=Real)
    args.add_argument("--max_threshold", default=5, type=Real)
    args.add_argument("--graph_step", default=0.001, type=float)
    args.add_argument("--min_graph", default=0, type=float)
    args.add_argument("--max_graph", default=20, type=float)
    params = args.parse_args()
    ideal_value = params.ideal_value
    threshold = {"min": params.min_threshold, "max": params.max_threshold}
    domain = {"min": params.min_domain, "max": params.max_domain}
    x = []
    y = []
    for i in np.arange(params.min_graph, params.max_graph, params.graph_step):
        x.append(i)
        y.append(score_metric(i, threshold, domain, ideal_value))
    plt.plot(x, y)
    plt.title("Normalization map")
    plt.show()
