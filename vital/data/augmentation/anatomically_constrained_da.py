import functools
import itertools
import logging
from abc import ABC
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Dict, Tuple, Type

import h5py
import numpy as np
from tqdm import tqdm

from vital.data.camus.data_module import CamusDataModule
from vital.data.data_module import VitalDataModule
from vital.tasks.autoencoder import SegmentationAutoencoderTask
from vital.tasks.utils.autoencoder import decode, encode_dataset
from vital.utils.image.io import sitk_save
from vital.utils.logging import configure_logging
from vital.utils.parsing import StoreDictKeyPair
from vital.utils.sampling.rejection_sampling import RejectionSampler
from vital.utils.saving import load_from_checkpoint

logger = logging.getLogger(__name__)


def _robust_validity_check(*args, checker: Callable[..., bool], **kwargs) -> bool:
    """Wraps ``checker`` function in a try/catch to avoid a crash in the function crashing the whole program.

    Args:
        *args: Additional parameters to pass along to ``checker``.
        checker: Function to wrap in try/catch, whose crash should not interrupt the continuation of the program.
        **kwargs: Additional parameters to pass along to ``checker``.

    Returns:
        Value returned by ``checker`` if call returned, ``False`` if the call to ``checker`` raised an exception.
    """
    try:
        is_valid = checker(*args, **kwargs)
    except Exception:
        is_valid = False
    return is_valid


class AnatomicallyConstrainedDataAugmenter(ABC):
    """Framework to generate a massive dataset of artificially generated segmentations.

    The artificial segmentations (sampled using rejection sampling in a latent space encoding the domain data) are also
    classified as anatomically correct or incorrect.
    """

    datamodule_cls: Type[VitalDataModule]  # Implementation of the data module for the dataset we want to augment
    img_format: str  # File extension of the image format to save the reconstructed samples as

    def __init__(self, ae_system: SegmentationAutoencoderTask, **data_module_kwargs):
        """Initializes class instance.

        Args:
            ae_system: Autoencoder system with generative capabilities used to encode the starting samples and decode
                the generated samples.
        """
        self.autoencoder = ae_system.eval()
        self.datamodule = self.datamodule_cls(**data_module_kwargs)

        # Function that takes as input a single parameter (the decoded sample to check for anatomical validity) and
        # indicates whether the segmentation is anatomically plausible or not.
        # This wrapper is specified as a field (rather than a method) to allow for instance-specific configuration
        # (e.g. through currying) while not requiring the whole class to picklable when using the multiprocessing API.
        self._check_segmentation_validity: Callable[[np.ndarray], bool]

    def sample(
        self, num_samples: int, rs_batch_size: int = None, **rejection_sampler_kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Samples artificial data from the latent space, based on the distribution of the training data.

        Args:
            num_samples: Number of samples to gather from rejection sampling before filtering out anatomical errors
            rs_batch_size: Number of samples to generate by each parallel rejections sampling task.
                If ``rs_batch_size`` is too small, the sampling might take longer because of process creation overhead.
                If ``rs_batch_size`` is too big, the user feedback on how sampling is progressing might not be reliable
                (the progress bar only updates every few batches).
                If not specified, this uses the rejection sampling's default value of the N / 100, where N is the total
                number of samples to generate.
            **rejection_sampler_kwargs: Keyword arguments to pass to the ``RejectionSampler``'s init.

        Returns:
            Datasets of samples that are without or with anatomical metrics, respectively.
        """
        # Encode training groundtruths to latent space
        starting_samples = encode_dataset(self.autoencoder, self.datamodule, progress_bar=True)

        # Sampling
        # The value of the bandwidth was determined empirically so as to accept a non-negligible proportion of
        # generated samples
        rejection_sampler = RejectionSampler(starting_samples, **rejection_sampler_kwargs)
        latent_space_samples = rejection_sampler.sample(num_samples, batch_size=rs_batch_size)

        # Postprocessing on samples to generate the dataset and the decoded samples
        return self._classify_latent_space_anatomical_errors(latent_space_samples)

    def _classify_latent_space_anatomical_errors(
        self, latent_space_samples: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Classifies latent space samples based on whether they project to segmentations with anatomical errors or not.

        Args:
            latent_space_samples: N x D array where N is the number of latent space samples and D is the dimensionality
                of the latent space.

        Returns:
            Tuple of two ? x D arrays, where D is the dimensionality of the latent space.
            The first is an M x D array, where M is the number of latent space samples projecting to segmentation maps
            without anatomical errors.
            The second is an K x D array, where K is the number of latent space samples projecting to segmentation maps
            with anatomical errors.
        """

        # A generator that decodes latent space samples by batch (to efficiently process batches at each call)
        # but returns the decoded samples one at a time
        def _decoded_latent_samples_generator():
            for batch_idx in range(int(np.ceil(len(latent_space_samples) / self.datamodule.batch_size))):
                # Gather batch of sampled encodings
                latent_space_samples_batch = latent_space_samples[
                    batch_idx * self.datamodule.batch_size : (batch_idx + 1) * self.datamodule.batch_size
                ]

                # Decode batch of sampled encodings
                decoded_latent_space_samples = decode(self.autoencoder, latent_space_samples_batch)

                # Yield each decoded sample individually
                for decoded_latent_space_sample in decoded_latent_space_samples:
                    yield decoded_latent_space_sample

        # Wrap anatomical metrics computation in try/catch to avoid crashing the whole process if we fail to compute the
        # anatomical metrics on a particularly degenerated sample that would have been accepted by rejection sampling
        _robust_segmentation_validity_check = functools.partial(
            _robust_validity_check, checker=self._check_segmentation_validity
        )

        # Process each decoded sample in parallel
        # The result is a nested list of booleans indicating whether the segmentations are anatomically plausible or not
        with Pool() as pool:
            valid_segmentations = list(
                tqdm(
                    pool.imap(_robust_segmentation_validity_check, _decoded_latent_samples_generator()),
                    total=len(latent_space_samples),
                    unit="sample",
                    desc="Classifying latent space samples by the presence of anatomical errors in the projected "
                    "segmentation maps",
                )
            )

        # Convert list to array to be able to invert the values easily
        valid_segmentations = np.array(valid_segmentations)

        logger.info(
            "Percentage of latent samples that do not produce anatomical errors: "
            f"{sum(valid_segmentations) * 100 / len(latent_space_samples):.2f}"
        )

        return latent_space_samples[valid_segmentations], latent_space_samples[~valid_segmentations]

    @classmethod
    def save_samples(cls, samples: Dict[str, np.ndarray], output: Path) -> None:
        """Creates an HDF5 dataset containing the samples, split in multiple groups.

        Args:
            samples: Set of ? x D arrays, where D is the dimensionality of the latent space and the unknown length is
                the number of samples of the group. The key identifying each group of samples will be used to name the
                dataset inside the HDF5 where the decoded samples will be saved.
            output: Path of the HDF5 dataset to create to save the samples.
        """
        logger.info(f"Saving the dataset of samples from the latent space to: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output, "w") as dataset:
            for samples_group_name, samples_group in samples.items():
                dataset.create_dataset(samples_group_name, data=samples_group)

    def save_random_decoded_samples(
        self, samples: Dict[str, np.ndarray], num_sample_images: int, output_folder: Path
    ) -> None:
        """Decodes groups of random samples from the latent space and saves them as images.

        Args:
            samples: Set of ? x D arrays, where D is the dimensionality of the latent space and the unknown length is
                the number of samples of the group. The key identifying each group of samples will be used to name the
                sub-folder where the decoded samples will be saved.
            num_sample_images: Number of random samples to decode and save as images, to provide visual examples of
                rejection sampling results.
            output_folder: Root directory where to save the decoded images.
        """
        samples_names = []

        random_samples = {}
        for samples_group_name, samples_group in samples.items():
            # Choose random samples from latent space with and without anatomical errors
            num_random_samples = min(num_sample_images, len(samples_group))
            if num_random_samples != num_sample_images:
                logger.warning(
                    f"Couldn't decode {num_sample_images} '{samples_group_name}' sample images, because only "
                    f"{len(samples_group)} are available for that group. "
                    f"Defaulting to decoding {num_random_samples} '{samples_group_name}' sample images."
                )
            random_samples[samples_group_name] = (
                samples_group[np.random.choice(len(samples_group), num_sample_images, replace=False), :]
                if num_random_samples
                else np.empty((0, 0))
            )

            # Setup directories for decoded samples
            samples_dir = output_folder / samples_group_name
            samples_dir.mkdir(parents=True, exist_ok=True)
            samples_names += [samples_dir / f"sample{i + 1}" for i in range(num_random_samples)]

        # Decode each latent sample individually and save it to disk
        pbar = tqdm(
            zip(itertools.chain(np.vstack(list(random_samples.values()))), samples_names),
            total=len(samples_names),
            desc="Saving decoded images of random samples",
            unit="sample",
        )
        for sample, name in pbar:
            sitk_save(decode(self.autoencoder, sample), name.with_suffix(f".{self.img_format}"), dtype=np.uint8)

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds generic anatomically-constrained data augmentation arguments to a parser object.

        Args:
            parent_parser: Parser object to which generic anatomically-constrained data augmentation arguments will be
                added.

        Returns:
            Parser object with added generic anatomically-constrained data augmentation arguments.
        """
        parent_parser.add_argument(
            "--rejection_sampler_kwargs",
            action=StoreDictKeyPair,
            default=dict(),
            help="Parameters that will be passed along to the `RejectionSampler`'s init",
        )
        return parent_parser


# # TODO Uncomment once ACDC dataset and anatomical metrics are implemented
# class AcdcAnatomicallyConstrainedDataAugmenter(AnatomicallyConstrainedDataAugmenter):
#     """Implementation of the artificial data augmenter framework for the CAMUS dataset."""
#
#     datamodule_cls = AcdcDataModule
#     img_format = "nii.gz"
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#         # The `voxelspacing` parameter was determined empirically by using mean voxel spacing across the dataset
#         from vital.metrics.acdc.anatomical.utils import check_segmentation_validity
#
#         self._check_segmentation_validity = functools.partial(check_segmentation_validity, voxelspacing=(1.4, 1.4))


class CamusAnatomicallyConstrainedDataAugmenter(AnatomicallyConstrainedDataAugmenter):
    """Implementation of the artificial data augmenter framework for the CAMUS dataset."""

    datamodule_cls = CamusDataModule
    img_format = "mhd"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # The `voxelspacing` parameter was determined empirically by scaling the original voxel spacing, identical
        # across all CAMUS images, from the average cropped bbox shape (always square so as to remain isotropic) to the
        # CNNs shape
        # Original voxel spacing (H,W): (0.154,0.308)
        # CAMUS average shape (H,W): (840,840)
        # CNNs shape (H,W): (256,256)
        # Target voxel spacing:
        #   Height: 840 * 0.154 / 256 = 0.505
        #   Width: 840 * 0.308 / 256 = 1.011
        from vital.metrics.camus.anatomical.utils import check_segmentation_validity

        self._check_segmentation_validity = functools.partial(
            check_segmentation_validity, voxelspacing=(0.505, 1.011), labels=self.autoencoder.hparams.data_params.labels
        )


def main():
    """Run the script."""
    configure_logging(console_level=logging.INFO)

    dataset_augmenters = {
        # 'acdc': AcdcAnatomicallyConstrainedDataAugmenter, TODO Uncomment once ACDC AEs are implemented
        "camus": CamusAnatomicallyConstrainedDataAugmenter,
    }

    # Initialize the parser with generic data augmentation arguments
    parser = ArgumentParser()
    parser = AnatomicallyConstrainedDataAugmenter.add_argparse_args(parser)
    parser.add_argument(
        "pretrained_ae",
        type=Path,
        help="Path to a model checkpoint, or name of a model from a Comet model registry, of the autoencoder to use "
        "to sample its latent space",
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        default=Path.cwd(),
        help="Root directory where to save the augmented dataset of generated samples and the decoded samples",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=int(5e6),
        help="Number of samples to gather from rejection sampling before filtering out anatomical errors",
    )
    parser.add_argument(
        "--num_sample_images",
        type=int,
        default=100,
        help="Number of random samples to decode and save as images, to provide visual examples of rejection "
        "sampling results",
    )
    parser.add_argument(
        "--rs_batch_size", type=int, help="Number of samples to generate by each parallel rejections sampling task"
    )

    # Add subparsers for all data augmenters available
    datasets_subparsers = parser.add_subparsers(
        title="dataset", dest="dataset", required=True, description="Dataset on which to perform data augmentation"
    )

    for dataset, data_augmenter in dataset_augmenters.items():
        ds_parser = datasets_subparsers.add_parser(dataset, help=f"{dataset.upper()} dataset")
        data_augmenter.datamodule_cls.add_argparse_args(ds_parser)

    # Parse args and setup the target data augmenter
    args = parser.parse_args()
    data_augmenter_cls = dataset_augmenters[args.dataset]
    data_augmenter = data_augmenter_cls(
        ae_system=load_from_checkpoint(args.pretrained_ae, expected_checkpoint_type=SegmentationAutoencoderTask),
        **vars(args),
    )

    # Perform the data augmentation
    samples_wo_errors, samples_w_errors = data_augmenter.sample(
        args.num_samples, rs_batch_size=args.rs_batch_size, **args.rejection_sampler_kwargs
    )
    samples = {"no_anatomical_errors": samples_wo_errors, "anatomical_errors": samples_w_errors}

    # Save the results of the data augmentation
    data_augmenter.save_samples(samples, args.output_folder / "samples.h5")
    data_augmenter.save_random_decoded_samples(samples, args.num_sample_images, args.output_folder)


if __name__ == "__main__":
    main()
