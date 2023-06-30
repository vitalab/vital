import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Sequence, Tuple, Union

import numpy as np
import torch
import yaml
from torchmetrics.utilities.data import to_categorical

from vital.metrics.evaluate.attribute import check_temporal_consistency_errors
from vital.tasks.autoencoder import SegmentationArVaeTask, SegmentationAutoencoderTask
from vital.utils.image.process import PostProcessor
from vital.utils.image.register.affine import AffineRegisteringTransformer
from vital.utils.image.transform import remove_labels, resize_image
from vital.utils.saving import load_from_checkpoint
from vital.utils.signal.regression import kernel_ridge_regression
from vital.utils.signal.snake import DualLagrangianRelaxationSnake, PenaltySnake, Snake

logger = logging.getLogger(__name__)


class AutoencoderPostprocessing(PostProcessor):
    """Post-processing that reconstructs a batch of 2D segmentation maps using an autoencoder."""

    def __init__(
        self,
        autoencoder: Union[str, Path, SegmentationAutoencoderTask],
        data_labels: Sequence[int],
        registering_transformer: AffineRegisteringTransformer = None,
    ):
        """Initializes class instance.

        Args:
            autoencoder: Autoencoder model that can reconstruct segmentation maps, or location of an autoencoder model's
                checkpoint.
            data_labels: Labels provided in the data to process. Any labels not supported by the autoencoder model will
                simply be discarded prior to feeding the data to the autoencoder.
            registering_transformer: ``None`` if the segmentations don't need to be registered. Otherwise, this
                transformer will register segmentations to post-process before they are fed to the autoencoder.
                - If `None`, the autoencoder is given the segmentations as is, and the reconstructed segmentations are
                  not transformed in any way.
                - If present, the segmentations to post-process will be registered before being fed to the autoencoder
                  network. The reconstructed segmentations are then transformed back to fit with the original data.
        """
        self.autoencoder = autoencoder
        if isinstance(self.autoencoder, (str, Path)):
            self.autoencoder = load_from_checkpoint(
                self.autoencoder, expected_checkpoint_type=SegmentationAutoencoderTask
            )
        self.registering_transformer = registering_transformer
        self.post_tag = "post_mask"

        self._labels_to_remove = [
            label for label in data_labels if label not in self.autoencoder.hparams.data_params.labels
        ]
        if self._labels_to_remove:
            logger.warning(
                f"Labels available in the data and labels used by the autoencoder model do not match. Therefore, "
                f"autoencoder will discard the following labels: {self._labels_to_remove}. "
                f"Labels available in the data: {data_labels} "
                f"Labels used by the autoencoder: {self.autoencoder.hparams.data_params.labels}"
            )

    def __call__(self, batch, **kwargs):
        """Processes a batch of 2D segmentation maps.

        Args:
            batch: (N, H, W), Segmentation maps to post-process.

        Returns:
            post_tag: (N, H, W), Segmentation maps reconstructed by an autoencoder model.
            encoding_tag: (N, ?), Encoding of the items in the autoencoder's latent space.
        """
        # Remove labels from the segmented sequence that the autoencoder model as not been trained to reconstruct
        batch = remove_labels(batch, self._labels_to_remove, fill_label=0)

        shape = batch.shape[1:]  # Extract original segmentation shape info

        if self.registering_transformer:
            # Register the segmentation if the autoencoder requires input to be registered
            reg_params, batch = self.registering_transformer.register_batch(batch)
        else:
            # Resize the image to fit as input for the AE (reverse the shape order to fit resize fn convention)
            ae_input_shape = self.autoencoder.hparams.data_params.out_shape[1:]
            batch = resize_image(batch, ae_input_shape[::-1])

        proc_encoding, reconstruction = self._process(batch)

        if self.registering_transformer:
            # Undo the registering on the segmentation if it was applied at first
            proc_batch = self.registering_transformer.undo_batch_registering(reconstruction, reg_params)
        else:
            # Revert image to its original shape (reverse the shape order to fit resize fn convention)
            proc_batch = resize_image(reconstruction.astype(np.uint8), shape[::-1])

        return {
            self.post_tag: proc_batch,
            self.autoencoder.hparams.encoding_tag: proc_encoding,
        }

    def _process(self, batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Processes a batch of 2D segmentation maps.

        Args:
            batch: (N, H, W), Segmentation maps to post-process, with a shape matching the autoencoder's expected
                input shape.

        Returns:
            post_tag: (N, H, W), Segmentation maps reconstructed by an autoencoder model, with a shape matching the
              autoencoder's output.
            encoding_tag: (N, ?), Encoding of the segmentation maps in the autoencoder's latent space.
        """
        # Process the segmented sequence w/ the autoencoder
        encoding = self.autoencoder(torch.from_numpy(batch), task="encode")
        reconstruction = self.autoencoder(encoding, task="decode")

        # Convert output tensors to numpy arrays (w/ standard channel order)
        encoding = encoding.detach().cpu().numpy()
        reconstruction = to_categorical(reconstruction).detach().cpu().numpy()

        return encoding, reconstruction


class _AutoencoderFiltering(AutoencoderPostprocessing):
    """Autoencoder reconstruction post-processing that filters latent dimensions as 1D signals before decoding."""

    def __init__(self, filter_attrs_only: bool = False, **kwargs):
        """Initializes class instance.

        Args:
            filter_attrs_only: Whether to only filter the dimensions in the latent space directly correlated to image
                attributes.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)
        if filter_attrs_only and not isinstance(self.autoencoder, SegmentationArVaeTask):
            raise ValueError(
                "You requested functionalities requiring autoencoder models that directly correlate some latent "
                "dimensions to image attributes (by setting `filter_attrs_only=True`). However, the provided "
                "autoencoder model does not possess these correlations, i.e. it's not an AR-VAE. Either only request "
                "generic autoencoder functionalities, or provide an AR-VAE model."
            )

        # Determine dimensions to smooth, and the degree of the polynomials to fit for each of these dimensions
        if filter_attrs_only:
            self._latent_dims_to_filter = list(range(len(self.autoencoder.hparams.attrs)))
        else:
            self._latent_dims_to_filter = list(range(self.autoencoder.hparams.model.latent_dim))

    def _process(self, batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Encode the segmentation maps w/ the autoencoder
        encoding = self.autoencoder(torch.from_numpy(batch), task="encode")
        encoding = encoding.detach().cpu().numpy()

        # Replace each attribute across the batch with its smoothed values
        for dim in self._latent_dims_to_filter:
            attr_vals = encoding[:, dim]
            encoding[:, dim] = self._process_dim(dim, attr_vals)

        # Reconstruct the segmentation maps from the smoothed encodings w/ the autoencoder
        reconstruction = self.autoencoder(torch.from_numpy(encoding), task="decode")

        # Convert output tensors to numpy arrays (w/ standard channel order)
        reconstruction = to_categorical(reconstruction).detach().cpu().numpy()

        return encoding, reconstruction

    @abstractmethod
    def _process_dim(self, dim_idx: int, dim: np.ndarray) -> np.ndarray:
        """Processes a batch's specific latent dimension as a 1D signal.

        Args:
            dim_idx: Index in the latent space of the dimension to process.
            dim: (N,), Specific latent dimension of the encoding of the segmentation maps in the autoencoder's latent
                space.

        Returns:
            (N,), Filtered values of the input latent dimension.
        """


class AutoencoderRegression(_AutoencoderFiltering):
    """Autoencoder reconstruction post-processing that regresses latent dimensions before decoding."""

    def __init__(
        self,
        kernel: str = "rbf",
        alpha: float = 0.1,
        kernel_kwargs: Mapping[str, Any] = None,
        pad_mode: str = None,
        pad_width: Union[int, float] = None,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            kernel: Name of the kernel to use for the kernel ridge regression.
            alpha: Regularization strength for the kernel ridge regression.
            kernel_kwargs: Additional parameters for the kernel function.
            pad_mode: Mode used to determine how to pad points at the beginning/end of the array. The options available
                are those of the ``mode`` parameter of ``numpy.pad``.
            pad_width: If it is an integer, the number of entries to repeat before/after the array. If it is a float,
                the fraction of the data's length to repeat before/after the array.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)
        if kernel_kwargs is None:
            kernel_kwargs = {}
        self._krr_kwargs = {
            "kernel": kernel,
            "alpha": alpha,
            **kernel_kwargs,
            "pad_mode": pad_mode,
            "pad_width": pad_width,
        }

    def _process_dim(self, dim_idx: int, dim: np.ndarray) -> np.ndarray:
        return kernel_ridge_regression(dim, **self._krr_kwargs)


class AutoencoderSnake(_AutoencoderFiltering):
    """Autoencoder reconstruction post-processing that minimizes energies on latent dimensions before decoding."""

    def __init__(
        self,
        grad_step: float = 1e-2,
        smoothness_weight: float = 50,
        num_neighbors: int = 1,
        neighbors_pad_mode: str = "edge",
        max_iterations: int = int(1e5),
        convergence_eps: float = np.finfo(np.float32).eps,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            grad_step: Step size to take at each optimization iteration.
            smoothness_weight: Weight of the loss' smoothness term.
            num_neighbors: Number of neighbors to take into account when computing the smoothness term. This also
                dictates the width by which to pad the signal at the edges.
            neighbors_pad_mode: Mode used to determine how to pad points at the beginning/end of the array. The options
                available are those of the ``mode`` parameter of ``numpy.pad``.
            max_iterations: If the model doesn't converge to a stable configuration, number of steps after which to
                stop.
            convergence_eps: Threshold on the L2 norm between consecutive optimization steps below which the algorithm
                is considered to have converged.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)
        self._snake_kwargs = {
            "grad_step": grad_step,
            "smoothness_weight": smoothness_weight,
            "num_neighbors": num_neighbors,
            "neighbors_pad_mode": neighbors_pad_mode,
            "max_iterations": max_iterations,
            "convergence_eps": convergence_eps,
        }
        self._snakes = [
            self._build_snake_by_latent_dimension(dim) for dim in range(self.autoencoder.hparams.model.latent_dim)
        ]

    def _build_snake_by_latent_dimension(self, dim: int) -> Snake:
        """Creates a `Snake` instance customized to optimize a specific latent dimension.

        Args:
            dim: Index in the latent space of the dimension for which to configure the snake.

        Returns:
            `Snake` instance configured to optimize the requested latent dimension.
        """
        return Snake(**self._snake_kwargs)

    def _process_dim(self, dim_idx: int, dim: np.ndarray) -> np.ndarray:
        return self._snakes[dim_idx](dim)


class ConstrainedAutoencoderSnake(AutoencoderSnake):
    """Autoencoder reconstruction post-processing that enforces constraints on latent dimensions before decoding."""

    def __init__(
        self,
        attr_thresholds: Union[str, Path],
        attr_smoothness_constraint: Literal["dlr", "penalty"] = "dlr",
        constraint_mode_kwargs: Mapping[str, Any] = None,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            attr_thresholds: File containing pre-computed thresholds on the acceptable temporal consistency metrics'
                values for each attribute.
            attr_smoothness_constraint: How to handle the hard constraint on the attributes' smoothness.
                - ``'dlr'``: optimizes the global smoothness' weight in a dual lagrangian relaxation process to enforce
                  the smoothness constraint on the attributes' temporal signal;
                - ``'penalty'``: adds the constraint as a penalty function that enforces the smoothness constraint on
                  the attributes' temporal signal.
            constraint_mode_kwargs: Additional parameters for the specific snake configuration dictated by
                ``attr_smoothness_constraint``, passed along to the ``Snake``'s ``init``.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        # Assign `ConstrainedAutoencoderSnake` fields before calling super().__init__() so that the snakes configured by
        # the parents can know which class to instantiate and which parameters to pass to its constructor.
        match attr_smoothness_constraint:
            case "dlr":
                self._constrained_snake_cls = DualLagrangianRelaxationSnake
            case "penalty":
                self._constrained_snake_cls = PenaltySnake
            case _:
                raise ValueError(
                    f"Unexpected value for `attr_smoothness_constraint`: '{attr_smoothness_constraint}'. Pick "
                    f"one of: ['dlr', 'penalty']."
                )
        self._constraint_mode_kwargs = constraint_mode_kwargs if constraint_mode_kwargs else {}

        # Load statistics on the attribute thresholds from the config file
        with open(attr_thresholds) as f:
            self._attr_thresholds = yaml.safe_load(f)

        super().__init__(**kwargs)
        if not isinstance(self.autoencoder, SegmentationArVaeTask):
            raise ValueError(
                f"{self.__class__.__name__} requires an autoencoder models that directly correlate some latent "
                f"dimensions to image attributes, to enforce the requested constraints. However, the provided "
                f"'{self.autoencoder.__class__.__name__}' autoencoder model does not display these correlations. "
                f"Either change the requested autoencoder postprocessing, or provided a "
                f"'{SegmentationArVaeTask.__name__}' autoencoder model."
            )

    @property
    def _latent_space_attr_stats(self) -> Dict[str, Tuple[float, float]]:
        """Extract the statistics to normalize the latent space attributes from the model checkpoint.

        We do this in a property rather than in the constructor to avoid a cycle between which class in the hierarchy
        to initialize first. This lazy query on the latent space attributes' statistics allows us to avoid the cycle.
        """
        return {
            attr: (
                self.autoencoder.latent_min[attr_idx].cpu().numpy(),
                self.autoencoder.latent_max[attr_idx].cpu().numpy(),
            )
            for attr_idx, attr in enumerate(self.autoencoder.hparams.attrs)
        }

    def _build_snake_by_latent_dimension(self, dim: int) -> Snake:
        if dim < len(self.autoencoder.hparams.attrs):
            attr = self.autoencoder.hparams.attrs[dim]
            snake = self._constrained_snake_cls(
                **self._snake_kwargs,
                smoothness_constraint_func=lambda attr_vals: check_temporal_consistency_errors(
                    self._attr_thresholds[attr], attr_vals, bounds=self._latent_space_attr_stats[attr]
                ),
                **self._constraint_mode_kwargs,
            )
        else:
            snake = super()._build_snake_by_latent_dimension(dim)
        return snake
