from abc import ABC
from typing import Dict

from torch import Tensor

from vital.system import VitalSystem
from vital.utils.format.native import prefix


class SharedTrainEvalTask(VitalSystem, ABC):
    """Abstract task that shares a train/val step.

    Implements useful generic utilities and boilerplate Lighting code:
        - Handling of identical train/val step results (metrics logging and printing)
    """

    def _shared_train_val_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Handles steps for both training and validation loops, assuming the behavior should be the same.

        For models where the behavior in training and validation is different, then override ``training_step`` and
        ``validation_step`` directly (in which case ``_shared_train_val_step`` doesn't need to be implemented).

        Returns:
            Mapping between metric names and their values. It must contain at least a ``'loss'``, as that is the value
            optimized in training and monitored by callbacks during validation.
        """
        raise NotImplementedError

    def training_step(self, *args, **kwargs) -> Dict[str, Tensor]:  # noqa: D102
        result = prefix(self._shared_train_val_step(*args, **kwargs), "train/")
        self.log_dict(result, **self.hparams.train_log_kwargs)
        # Add reference to 'train_loss' under 'loss' keyword, requested by PL to know which metric to optimize
        result["loss"] = result["train/loss"]
        return result

    def validation_step(self, *args, **kwargs) -> Dict[str, Tensor]:  # noqa: D102
        result = prefix(self._shared_train_val_step(*args, **kwargs), "val/")
        self.log_dict(result, **self.hparams.val_log_kwargs)
        return result
