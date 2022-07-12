import itertools

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.states import TrainerFn
from tqdm.auto import tqdm

from vital.tasks.autoencoder import SegmentationAutoencoderTask


def encode_dataset(
    autoencoder: SegmentationAutoencoderTask,
    datamodule: LightningDataModule,
    segmentation_data_tag: str = None,
    progress_bar: bool = False,
) -> np.ndarray:
    """Encodes masks from the train and val sets of a dataset in the latent space learned by an autoencoder model.

    Args:
        autoencoder: Autoencoder model used to encode masks in a latent space.
        datamodule: Abstraction of the dataset to encode, allowing to access both the training and validation sets.
        segmentation_data_tag: Key to locate the data to encode from all the data returned in a batch. If not provided,
            defaults to the tag of the dataset the autoencoder was trained on.
        progress_bar: Whether to display a progress bar for the encoding of the samples.

    Returns:
        Array of training and validation samples encoded in the latent space.
    """
    # Setup the datamodule used to get the data points to encode in the latent space
    datamodule.setup(stage=TrainerFn.FITTING)
    train_dataloader, val_dataloader = datamodule.train_dataloader(), datamodule.val_dataloader()
    data = itertools.chain(train_dataloader, val_dataloader)
    num_batches = len(train_dataloader) + len(val_dataloader)

    if progress_bar:
        data = tqdm(data, desc="Encoding groundtruths", total=num_batches, unit="batch")

    # Encode training and validation groundtruths in the latent space
    seg_data_tag = autoencoder.hparams.segmentation_data_tag if not segmentation_data_tag else segmentation_data_tag
    with torch.no_grad():
        dataset_samples = [autoencoder(batch[seg_data_tag], task="encode").cpu() for batch in data]
    dataset_samples = torch.cat(dataset_samples).numpy()

    return dataset_samples


def decode(autoencoder: SegmentationAutoencoderTask, encoding: np.ndarray) -> np.ndarray:
    """Decodes a sample, or batch of samples, from the latent space to the output space.

    Args:
        autoencoder: Autoencoder model with generative capabilities used to decode the encoded samples.
        encoding: Sample, or batch of samples, from the latent space to decode.

    Returns:
        Decoded sample, or batch of samples.
    """
    encoding = encoding.astype(np.float32)  # Ensure the encoding is in single-precision float dtype
    if len(encoding.shape) == 1:
        # If the input isn't a batch of data, add the batch dimension
        encoding = encoding[None, :]
    encoding_tensor = torch.from_numpy(encoding)
    decoded_sample = autoencoder(encoding_tensor, task="decode").argmax(dim=1).squeeze()
    return decoded_sample.cpu().detach().numpy()
