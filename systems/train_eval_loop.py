from abc import ABC
from typing import Literal, Dict, List

import torch
from torch import Tensor

from vital.systems.vital_system import SystemTrainEvalLoopMixin


class SupervisedTrainEvalLoopMixin(SystemTrainEvalLoopMixin, ABC):
    """Abstract mixin for generic supervised train/eval loop.

    Implements useful generic utilities and boilerplate Lighting code:
        - Handling of identical train/val step results (metrics logging and printing)
    """

    def trainval_step(self, *args, metric_prefix: Literal['', 'val_'] = '', **kwargs) -> Dict[str, Tensor]:
        """Handles steps for both training and validation loops, assuming the behavior should be the same.

        For models where the behavior in training and validation is different, then override ``training_step`` and
        ``validation_step`` directly (in which case ``trainval_step`` doesn't need to be implemented).
        """
        raise NotImplementedError

    def training_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        return self.trainval_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        return self.trainval_step(*args, metric_prefix='val_', **kwargs)

    def trainval_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Dict[str, Tensor]]:
        """Handles logging and displaying metrics in the progress bar for both training and validation loops, assuming
        the behavior should be the same.

        By default, marks all outputs (accumulated over the steps) for ``progress_bar`` and ``log``.

        For models where the outputs in training and validation are different, then override ``training_epoch_end`` and
        ``validation_epoch_end`` directly (in which case the default implementation of ``trainval_epoch_end`` won't be
        called).
        """
        metric_names = outputs[0].keys()
        reduced_metrics = {metric_name: torch.stack([output[metric_name]
                                                     for output in outputs]).mean()
                           for metric_name in metric_names}
        return {'progress_bar': reduced_metrics,
                'log': reduced_metrics}

    def training_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Dict[str, Tensor]]:
        # Remove loss from outputs passed to generic epoch end so that it doesn't conflict with Lightning's automatic
        # logging of the loss
        for output in outputs:
            del output['loss']

        return self.trainval_epoch_end(outputs)

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Dict[str, Tensor]]:
        return self.trainval_epoch_end(outputs)
