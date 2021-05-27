from abc import ABC
from typing import Callable, Dict, Sequence, Any

import torch
import torch.nn as nn
from torch import Tensor

from vital.data.config import Tags
from vital.systems.computation import TrainValComputationMixin


class SupervisedComputationMixin(TrainValComputationMixin, ABC):
    """Abstract mixin for generic supervised train/val step.

    Implements useful generic utilities and boilerplate Lighting code:
        - Handling of identical train/val step results (metrics logging and printing)
    """

    def __init__(self, network: nn.Module, loss: Callable, metrics: Sequence[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        self.network = network

    def forward(self, x) -> Any:
        self.network(x)

    # def trainval_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
    #     x, y = batch[Tags.img], batch[Tags.gt]
    #
    #     # Forward
    #     y_hat = self.network(x)
    #
    #     loss = self.loss(y, y_hat)
    #     logs = {"loss": loss}
    #
    #     if self.metrics:
    #         metrics = self.compute_metrics(y_hat, y, self.metrics, **self.get_batch_dict(batch))
    #         logs.update(metrics)
    #     print(logs)
    #     return logs
    #
    # @staticmethod
    # def compute_metrics(y_hat: torch.Tensor, y: torch.Tensor, metrics: Sequence[Callable], **kwargs) -> Dict:
    #     """Compute system metrics.
    #
    #     Args:
    #         y_hat: model prediction
    #         y : target
    #         metrics: list of metrics to evaluate. Metrics can return single float or dict.
    #         kwargs: extra parameters to compute metrics.
    #
    #     Returns:
    #         computed metrics
    #     """
    #     metric_results = {}
    #     for metric in metrics:
    #         m = metric(y_hat, y, **kwargs)
    #         if isinstance(m, dict):
    #             metric_results.update(m)
    #         else:
    #             metric_results[metric.__class__.__name__] = torch.tensor(m)
    #
    #     return metric_results
