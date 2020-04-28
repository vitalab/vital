from abc import ABC
from typing import Literal

import torch

from vital.systems.vital_system import SystemTrainEvalLoopMixin


class SupervisedTrainEvalLoopMixin(SystemTrainEvalLoopMixin, ABC):
    """Abstract mixin for generic supervised train/eval loop.

    Implements useful generic utilities and boilerplate Lighting code:
        - Handling of identical train/val step results (metrics logging and printing)
    """

    def trainval_step(self, metric_prefix: Literal['', 'val_'] = '', **kwargs):
        """Must be implemented if either ``training_step`` or ``validation_step`` are not overridden.

        As the name indicates, handles steps for both training and validation loops, assuming the behavior should be
        the same. For models where the behavior in training and validation is different, than override
        ``training_step`` and ``validation_step`` directly."""
        raise NotImplementedError

    def training_step(self, *args, **kwargs):
        """Returns loss and metrics for callbacks, assuming ``training_epoch_end`` is not overridden."""
        metrics = self.trainval_step(*args, metric_prefix='', **kwargs)
        loss = metrics.pop('loss')
        return {'loss': loss,
                'progress_bar': metrics,
                'log': metrics}

    def validation_step(self, *args, **kwargs):
        """Returns metrics to be reduced ànd made accessible for display and callbacks in ``validation_epoch_end`."""
        return self.trainval_step(*args, metric_prefix='val_', **kwargs)

    def validation_epoch_end(self, outputs):
        """By default, mark all metrics for ``progress_bar`` and ``log``."""
        metric_names = outputs[0].keys()
        reduced_metrics = {metric_name: torch.stack([output[metric_name]
                                                     for output in outputs]).mean()
                           for metric_name in metric_names}
        return {'progress_bar': reduced_metrics,
                'log': reduced_metrics}
