from typing import Dict, Literal, Mapping, Sequence, Tuple, Union

import hydra
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics.utilities.data import to_onehot

from vital.data.config import Tags
from vital.metrics.train.functional import kl_div_zmuv
from vital.metrics.train.metric import DifferentiableDiceCoefficient
from vital.tasks.generic import SharedTrainEvalTask
from vital.utils.decorators import auto_move_data


class SegmentationAutoencoderTask(SharedTrainEvalTask):
    """Generic segmentation autoencoder training and inference steps.

    Implements generic segmentation train/val step and inference, assuming the following conditions:
        - the model from ``self.configure_model()`` returns as output a mapping between identifiers (specified by
          class attributes of the model) and predictions necessary to compute the loss:
            - the raw, unnormalized scores for each class in the reconstructed segmentation;
            - the latent space encoding.
        - The loss used is a weighted combination of Dice and cross-entropy.
    """

    def __init__(
        self, segmentation_data_tag: str = Tags.gt, ce_weight: float = 0.1, dice_weight: float = 1, *args, **kwargs
    ):
        """Initializes class instance.

        Args:
            segmentation_data_tag: Key to locate the data to reconstruct from all the data returned in a batch.
            ce_weight: Weight to give to the cross-entropy term of the autoencoder's reconstruction loss.
            dice_weight: Weight to give to the dice term of the autoencoder's reconstruction loss.
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(*args, **kwargs)
        self.model = self.configure_model()
        self.example_input_array = torch.randn((2, *self.hparams.data_params.out_shape))

        # Configure metric objects used repeatedly in the train/eval loop
        self._dice = DifferentiableDiceCoefficient(include_background=False, reduction="none")

    def configure_model(self) -> nn.Module:
        """Configure the network architecture used by the system."""
        return hydra.utils.instantiate(
            self.hparams.model,
            image_size=self.hparams.data_params.out_shape[1:],
            channels=self.hparams.data_params.out_shape[0],
        )

    @auto_move_data
    def forward(self, x: Tensor, task: Literal["encode", "decode", "reconstruct"] = "reconstruct") -> Tensor:
        """Performs test-time inference on the input.

        Args:
            x: - if ``task`` == 'decode': (N, ?), Encoding in the latent space.
               - else: (N, [C,] H, W), Input, in either categorical or onehot format.
            task: Flag indicating which type of inference task to perform.

        Returns:
            if ``task`` == 'encode':
                (N, ``Z``), Encoding of the input in the latent space.
            else:
                (N, C, H, W), Raw, unnormalized scores for each class in the input's reconstruction.
        """
        if task in ["encode", "reconstruct"]:
            if len(x.shape) == 3:
                x = self._categorical_to_input(x)
            x = self.model.encoder(x)
        if task in ["decode", "reconstruct"]:
            x = self.model.decoder(x)
        return x

    def _categorical_to_input(self, x: Tensor) -> Tensor:
        """Transform categorical tensor into onehot tensor that can be used as input by torch modules.

        Args:
            x: (N, H, W), Categorical tensor to transform into onehot tensor that can be used as input by torch modules.

        Returns:
            (N, C, H, W), Onehot tensor that can be used as input by torch modules.
        """
        return to_onehot(x, num_classes=len(self.hparams.data_params.labels)).float()

    def _shared_train_val_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:  # noqa: D102
        # Forward
        out = self.model(self._categorical_to_input(batch[self.hparams.segmentation_data_tag]))

        # Compute categorical input reconstruction metrics
        x_hat, x = out[self.model.reconstruction_tag], batch[self.hparams.segmentation_data_tag]
        ce = F.cross_entropy(x_hat, x)
        dice_values = self._dice(x_hat, x)
        dices = {f"dice/{label}": dice for label, dice in zip(self.hparams.data_params.labels[1:], dice_values)}
        mean_dice = dice_values.mean()
        metrics = {"ce": ce, "dice": mean_dice, **dices}

        # Compute loss and metrics
        metrics.update(self._compute_latent_space_metrics(out, batch))
        metrics["loss"] = self._compute_loss(metrics)

        return metrics

    def _compute_latent_space_metrics(self, out: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Computes metrics on the input's encoding in the latent space.

        Args:
             out: Output of a forward pass with the autoencoder network.
             batch: Content of the batch of data returned by the dataloader.

        Returns:
            Metrics useful for computing the loss and tracking the system's training progress.
        """
        return {}

    def _compute_loss(self, metrics: Mapping[str, Tensor]) -> Tensor:
        """Computes loss for a train/val step based on various metrics computed on the system's predictions.

        Args:
            metrics: Metrics useful for computing the loss (usually a combination of metrics from
                ``self._shared_train_val_step`` and ``self._compute_latent_space_metrics``).

        Returns:
            Loss for a train/val step.
        """
        # Weighted reconstruction loss
        loss = (self.hparams.ce_weight * metrics["ce"]) + (self.hparams.dice_weight * (1 - metrics["dice"]))
        return loss

    def predict_step(  # noqa: D102
        self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, Tensor]:
        x = batch[self.hparams.segmentation_data_tag]

        # Split the sequences in batches, in case the sequences are bigger than the batch size that fits in memory
        x_hat, z = [], []
        batch_size = self.trainer.datamodule.batch_size
        for batch_idx in range(int(np.ceil(len(x) / batch_size))):
            x_batch = x[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            z.append(self(x_batch, task="encode"))
            x_hat.append(self(z[-1], task="decode"))
        # Assemble the predictions on the whole batch from those of the sub-batches
        x_hat, z = torch.cat(x_hat), torch.cat(z)

        return {Tags.pred: x_hat, Tags.encoding: z}


class SegmentationBetaVaeTask(SegmentationAutoencoderTask):
    """Generic segmentation beta-VAE (VAE with weighted KL divergence) training and inference steps.

    Builds on top of the generic autoencoder train/eval loop with 2 minor adjustments:
        - adding the sampling step
        - computing the KL divergence metric and adding it to the global loss

    References:
        - VAE paper: https://arxiv.org/pdf/1312.6114.pdf
        - beta-VAE paper: https://openreview.net/pdf?id=Sy2fzU9gl
        - beta-VAE with modified training regime: https://arxiv.org/pdf/1804.03599.pdf
    """

    def __init__(self, beta: float = 1e-4, capacity: float = 0, capacity_schedule: int = 0, *args, **kwargs):
        """Initializes class instance.

        Args:
            beta: Weight to give to the Kullback-Leibler divergence term when computing the VAE loss.
            capacity: Target Kullback-Leibler divergence from which any divergence, positive or negative, will be
                penalized.
            capacity_schedule: Number of steps over which to linearly increase the capacity of the latent space encoding
                from 0 to `capacity`.
            *args: Positional arguments to pass to the parent's constructor.
            **kwargs: Keyword arguments to pass to the parent's constructor.
        """
        super().__init__(*args, **kwargs)

    @auto_move_data
    def forward(
        self, x: Tensor, task: Literal["encode2distr", "encode", "decode", "reconstruct"] = "reconstruct"
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Performs test-time inference on the input.

        Args:
            x: - if ``task`` == 'decode': (N, ?), Encoding in the latent space.
               - else: (N, [C,] H, W), Input, in either categorical or onehot format.
            task: Flag indicating which type of inference task to perform.

        Returns:
            if ``task`` == 'encode2distr':
                - (N, ``Z``), Mean of the predicted distribution of the input in the latent space, used to sample ``z``,
                  the encoding in the latent space.
                - (N, ``Z``), Log variance of the predicted distribution of the input in the latent space, used to
                  sample ``z``, the encoding in the latent space.
            if ``task`` == 'encode':
                (N, ``Z``), Encoding of the input in the latent space.
            if ``task`` in ['decode', 'reconstruct']:
                (N, C, H, W), Raw, unnormalized scores for each class in the input's reconstruction.
        """
        if task in ["encode2distr", "encode", "reconstruct"]:
            if len(x.shape) == 3:
                x = self._categorical_to_input(x)
            x = self.model.encoder(x)
            if task != "encode2distr":
                x = x[0]  # Continue with the latent space posterior's mean as the encoding
        if task in ["decode", "reconstruct"]:
            x = self.model.decoder(x)
        return x

    def _compute_latent_space_metrics(self, out: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Computes metrics on the input's encoding in the latent space.

        Adds the VAE's KL divergence term to the metrics already computed by the parent's implementation.

        Args:
            out: Output of a forward pass with the autoencoder network.
            batch: Content of the batch of data returned by the dataloader.

        Returns:
            Metrics useful for computing the loss and tracking the system's training progress:
                - metrics computed by ``super()._compute_latent_space_metrics``
                - VAE's KL divergence term
        """
        metrics = super()._compute_latent_space_metrics(out, batch)
        metrics["kl_div"] = kl_div_zmuv(out[self.model.distr_mean_tag], out[self.model.distr_logvar_tag])
        return metrics

    def _compute_loss(self, metrics: Mapping[str, Tensor]) -> Tensor:
        """Computes loss for a train/val step based on various metrics computed on the system's predictions.

        Adds the weighted KL divergence term to the loss already computed by the parent's implementation.

        The weight used on the KL divergence term comes from a follow-up paper to the original beta-VAE paper, by the
        original authors, where they alter the training regime to be able to target a specific KL divergence, with the
        goal of allowing to control the capacity of the reconstructions.

        Args:
            metrics: Metrics useful for computing the loss (usually a combination of metrics from
                ``self._compute_reconstruction_metrics`` and ``self._compute_latent_space_metrics``).

        Returns:
            Loss for a train/val step.
        """
        loss = super()._compute_loss(metrics)
        capacity = self.hparams.capacity * (
            min(self.global_step, self.hparams.capacity_schedule) / self.hparams.capacity_schedule
            if self.hparams.capacity_schedule
            else 1
        )
        loss += (metrics["kl_div"] - capacity) * self.hparams.beta
        return loss


class SegmentationArVaeTask(SegmentationBetaVaeTask):
    """Generic segmentation AR-VAE (Attribute Regularization VAE) training and inference steps.

    Builds on top of the generic variational autoencoder train/eval loop with the AR-VAE adjustments:
        - adds an attribute regularization term computed on the latent space encodings

    References:
        - AR-VAE paper: https://arxiv.org/pdf/2004.05485.pdf
    """

    def __init__(self, attrs: Sequence[str], gamma: float = 10, delta: float = 1, **kwargs):
        """Initializes class instance.

        Args:
            attrs: Labels identifying each attribute the AR-VAE should regularize for.
                This label should correspond to the key identifying the attribute's data in a batch dict, i.e.
                `batch[attrs[i]]`, where `batch` is a batch from the dataloader should return a (N, 1) tensor of the
                attribute's target values.
            gamma: Weight to give to the attribute regularization loss when computing the VAE's loss.
            delta: Attribute regularization hyperparameter that decides the spread of the posterior distribution.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)

    def _compute_latent_space_metrics(self, out: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Computes metrics on the input's encoding in the latent space.

        Adds the attribute regularization term to the loss already computed by the parent's implementation.

        Args:
            out: Output of a forward pass with the autoencoder network.
            batch: Content of the batch of data returned by the dataloader.

        References:
            - Computation of the attribute regularization term inspired by the original paper's implementation:
              https://github.com/ashispati/ar-vae/blob/master/utils/trainer.py#L378-L403

        Returns:
            Metrics useful for computing the loss and tracking the system's training progress:
                - metrics computed by ``super()._compute_latent_space_metrics``
                - attribute regularization term for each attribute (under the "{attr}_attr_reg" label format)
        """
        metrics = super()._compute_latent_space_metrics(out, batch)

        for attr_idx, attr in enumerate(self.hparams.attrs):
            # Extract dimension to regularize and target for the current attribute
            latent_code = out[self.model.encoding_tag][:, attr_idx].unsqueeze(1)
            attribute = batch[attr]

            # Compute latent distance matrix
            latent_code = latent_code.repeat(1, latent_code.shape[0])
            lc_dist_mat = latent_code - latent_code.transpose(1, 0)

            # Compute attribute distance matrix
            attribute = attribute.repeat(1, attribute.shape[0])
            attribute_dist_mat = attribute - attribute.transpose(1, 0)

            # Compute regularization loss
            lc_tanh = torch.tanh(lc_dist_mat * self.hparams.delta)
            attribute_sign = torch.sign(attribute_dist_mat)
            metrics[f"attr_reg/{attr}"] = F.l1_loss(lc_tanh, attribute_sign)

        return metrics

    def _compute_loss(self, metrics: Mapping[str, Tensor]) -> Tensor:
        """Computes loss for a train/val step based on various metrics computed on the system's predictions.

        Adds the attribute regularization term to the loss already computed by the parent's implementation.

        Args:
            metrics: Metrics useful for computing the loss (usually a combination of metrics from
                ``self._compute_reconstruction_metrics`` and ``self._compute_latent_space_metrics``).

        Returns:
            Loss for a train/val step.
        """
        loss = super()._compute_loss(metrics)
        attr_reg_sum = sum(metrics[f"attr_reg/{attr}"] for attr in self.hparams.attrs)
        loss += attr_reg_sum * self.hparams.gamma
        return loss
