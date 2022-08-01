from abc import ABC
from typing import Dict

from torch import Tensor

from vital.system import VitalSystem
from vital.utils.format.native import prefix


class SharedStepsTask(VitalSystem, ABC):
    """Abstract task that shares a train/val/test step.

    Implements useful generic utilities and boilerplate Lighting code:
        - Handling of identical train/val step results (metrics logging and printing)
    """

    def _shared_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Handles steps for the train/val/test loops, assuming the behavior should be the same.

        Returns:
            Mapping between metric names and their values. It must contain at least a ``'loss'``, as that is the value
            optimized in training and monitored by callbacks during validation.
        """
        raise NotImplementedError

    def training_step(self, *args, **kwargs) -> Dict[str, Tensor]:  # noqa: D102
        result = prefix(self._shared_step(*args, **kwargs), "train/")
        self.log_dict(result, **self.hparams.train_log_kwargs)
        # Add reference to 'train_loss' under 'loss' keyword, requested by PL to know which metric to optimize
        result["loss"] = result["train/loss"]
        return result

    def validation_step(self, *args, **kwargs) -> Dict[str, Tensor]:  # noqa: D102
        result = prefix(self._shared_step(*args, **kwargs), "val/")
        self.log_dict(result, **self.hparams.val_log_kwargs)
        return result

    def test_step(self, *args, **kwargs) -> Dict[str, Tensor]:  # noqa: D102
        result = prefix(self._shared_step(*args, **kwargs), "test/")
        self.log_dict(result, **self.hparams.val_log_kwargs)
        return result
