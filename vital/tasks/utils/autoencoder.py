import itertools
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.states import TrainerFn
from tqdm.auto import tqdm

from vital.tasks.autoencoder import SegmentationAutoencoderTask


def rename_significant_dims(encodings: pd.DataFrame, autoencoder: SegmentationAutoencoderTask) -> pd.DataFrame:
    """Renames columns in the `encodings` dataframe to explain their significance, if the AE model provides any.

    Args:
        encodings: Latent space encodings, each column matching to a latent space dimension.
        autoencoder: Autoencoder model that produced the latent space encodings.

    Returns:
        Latent space encodings dataframe, where column names match the explainable names of the latent dimensions.
    """
    # For AE models w/ explicitly named dimensions, used these names instead of generic index label
    if attrs := autoencoder.hparams.get("attrs"):
        cur_columns = encodings.columns
        encodings = encodings.rename({cur_columns[i]: attr for i, attr in enumerate(attrs)}, axis="columns")
    return encodings


def encode_dataset(
    autoencoder: SegmentationAutoencoderTask,
    datamodule: LightningDataModule,
    mask_tag: str = None,
    progress_bar: bool = False,
) -> np.ndarray:
    """Encodes masks from the train and val sets of a dataset in the latent space learned by an autoencoder model.

    Args:
        autoencoder: Autoencoder model used to encode masks in a latent space.
        datamodule: Abstraction of the dataset to encode, allowing to access both the training and validation sets.
        mask_tag: Key to locate the data to encode from all the data returned in a batch. If not provided, defaults to
            the tag of the dataset the autoencoder was trained on.
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
    mask_tag = autoencoder.hparams.mask_tag if not mask_tag else mask_tag
    with torch.no_grad():
        dataset_samples = [autoencoder(batch[mask_tag], task="encode").cpu() for batch in data]
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


def load_encodings(dataset: Literal["camus"], results: Path, progress_bar: bool = False, **kwargs) -> pd.DataFrame:
    """Loads the latent space encodings predicted by a model as a pandas `DataFrame`.

    Args:
        dataset: Name of the dataset for which to load the save latent space data.
            Acts as a switch to know how to read the provided file, assuming predictions are saved in a consistent
            way across a dataset.
        results: Path to the file of saved predictions from which to load the latent space encodings.
        progress_bar: Whether to display a progress bar for the loading of the encodings.
        **kwargs: Dataset specific keyword arguments to pass along to the loading function.

    Returns:
        Latent space encodings predicted by a model.
    """
    if dataset == "camus":
        encodings = _load_camus_encodings(results, progress_bar=progress_bar, **kwargs)
    else:
        raise ValueError(
            f"Loading latent space data from dataset '{dataset}' is not supported. \n"
            f"Please provide saved results from one of the following datasets: ['camus']."
        )
    return encodings


def _load_camus_encodings(
    results: Path, progress_bar: bool = False, include_voxelspacing: bool = False
) -> pd.DataFrame:
    """Loads the latent space encodings predicted by a model as a pandas `DataFrame`.

    Args:
        results: Path to the HDF5 of saved predictions from which to load the latent space encodings.
        progress_bar: Whether to display a progress bar for the loading of the encodings.
        include_voxelspacing: Whether to add voxelspacing associated with each sample as the following additional
            columns in the dataframe:
            - `'vsize_time'`
            - `'vsize_height'`
            - `'vsize_width'`

    Returns:
        Latent space encodings predicted by a model, along with any additional requested metadata.
    """
    from vital.data.camus.config import CamusTags
    from vital.results.camus.utils.itertools import PatientViews

    view_results = PatientViews(results_path=results, use_sequence=True)
    if progress_bar:
        view_results = tqdm(view_results, unit=view_results.desc, desc=f"Loading encodings from '{results}'")
    encodings = {}
    for view_result in view_results:
        view_encodings = view_result[f"{CamusTags.pred}/{CamusTags.encoding}"].data
        view_dict = {f"z_{dim}": view_encodings[:, dim] for dim in range(view_encodings.shape[1])}
        if include_voxelspacing:
            view_dict.update(zip(("vsize_time", "vsize_height", "vsize_width"), view_result.voxelspacing))
        encodings[view_result.id] = pd.DataFrame.from_dict(view_dict)

    return pd.concat(encodings).rename_axis(["group", "sample"])
