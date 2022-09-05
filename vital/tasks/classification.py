from typing import Dict

from torch import Tensor
from torch.nn import functional as F
from torchmetrics.functional import accuracy

from vital.tasks.generic import SharedStepsTask


class ClassificationTask(SharedStepsTask):
    """Generic classification training and inference steps.

    Implements generic classification train/val step and inference, assuming the following conditions:
        - the model from ``self.configure_model()`` returns one output: the raw, unnormalized scores for each class.
    """

    def __init__(self, image_tag: str, label_tag: str, *args, **kwargs):
        """Initializes the metric objects used repeatedly in the train/eval loop.

        Args:
            image_tag: Key to locate the input image from all the data returned in a batch.
            label_tag: Key to locate the target classification label from all the data returned in a batch.
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(*args, **kwargs)
        self.model = self.configure_model()

    def forward(self, *args, **kwargs):  # noqa: D102
        return self.model(*args, **kwargs)

    def _shared_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        x, y = batch[self.hparams.image_tag], batch[self.hparams.label_tag]

        # Forward
        y_hat = self.model(x)

        # Loss and metrics
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)

        # Format output
        return {"loss": loss, "accuracy": acc}
