from abc import ABC
from typing import Dict, List

import torch
from torch import Tensor

from vital.systems.vital_system import SystemComputationMixin
from vital.utils.decorators import prefix


class SupervisedComputationMixin(SystemComputationMixin, ABC):
    """Abstract mixin for generic supervised train/val step.

    Implements useful generic utilities and boilerplate Lighting code:
        - Handling of identical train/val step results (metrics logging and printing)
    """

    def trainval_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Handles steps for both training and validation loops, assuming the behavior should be the same.

        For models where the behavior in training and validation is different, then override ``training_step`` and
        ``validation_step`` directly (in which case ``trainval_step`` doesn't need to be implemented).
        """
        raise NotImplementedError

    @prefix("train", exclude="loss")
    def training_step(self, *args, **kwargs) -> Dict[str, Tensor]:  # noqa: D102
        return self.trainval_step(*args, **kwargs)

    @prefix("val")
    def validation_step(self, *args, **kwargs) -> Dict[str, Tensor]:  # noqa: D102
        return self.trainval_step(*args, **kwargs)

    def trainval_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Dict[str, Tensor]]:
        """Reduces all outputs by averaging them, and forwards them to be both logged and displayed in the progress bar.

        In cases where metrics from either training or validation need to handled in a special case (e.g. `train loss`),
        you can simply override ``training_epoch_end`` or ``validation_epoch_end`` to avoid this method being called.
        """
        metric_names = outputs[0].keys()
        reduced_metrics = {
            metric_name: torch.stack([output[metric_name] for output in outputs]).mean() for metric_name in metric_names
        }
        return {"progress_bar": reduced_metrics, "log": reduced_metrics.copy()}

    def training_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Dict[str, Tensor]]:  # noqa: D102
        out = self.trainval_epoch_end(outputs)
        # No need to add 'train_loss' to the progress bar, as it's displayed automatically by Lightning as 'loss'
        del out["progress_bar"]["loss"]
        # Log 'loss' rather as 'train_loss', to be consistent with other metrics (and validation)
        out["log"]["train_loss"] = out["log"].pop("loss")
        return out

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Dict[str, Tensor]]:  # noqa: D102
        return self.trainval_epoch_end(outputs)
