from pathlib import Path
from typing import Any, Sequence

import numpy as np
from lightning_utilities import apply_to_collection
from numpy.random import RandomState
from sklearn import model_selection
from tqdm.auto import tqdm

from vital.data.cardinal.config import TabularAttribute
from vital.data.cardinal.utils.itertools import Patients
from vital.utils.parsing import int_or_float

TRAIN_SET = "train"
VAL_SET = "val"
TEST_SET = "test"


def k_fold(
    data: Sequence[Any],
    stratify: Sequence[Any] | None = None,
    n_splits: int = 5,
    val_size: float | int = 0.1,
    shuffle: bool = True,
    random_state: int | RandomState | None = 12345,
) -> list[dict[str, list[int]]]:
    """Splits a sequence of sample indices into k folds for cross-validation.

    Args:
        data: Data of length `n_samples` to split.
        stratify: If provided, the data is split in a stratified fashion, using this as the class labels.
        n_splits: The number of folds/splits to create.
        val_size: The size of the validation set to split from each remaining training set after creating the K folds.
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the
            validation set. If int, represents the absolute number of validation samples.
        shuffle: Whether to shuffle the data before splitting.
        random_state: The random state to use for reproducibility.

    Returns:
        A list of splits, where each split contains the indices of its train, val and (optional) test sets.
    """
    # Create an array of indices relative to the input data +
    # ensure that labels are a numpy array to easily index them using arrays
    indices = np.arange(len(data))
    if stratify is not None:
        stratify = np.array(stratify)

    # If `val_size` represents a proportion of the dataset, compute the proportion relative to the training sets
    # (to increase the proportion to account for the test fold)
    if isinstance(val_size, float):
        val_size *= (n_splits - 1) / n_splits

    if stratify is None:
        k_fold_cls = model_selection.KFold
        train_val_split_cls = model_selection.ShuffleSplit
    else:
        k_fold_cls = model_selection.StratifiedKFold
        train_val_split_cls = model_selection.StratifiedShuffleSplit

    k_fold = k_fold_cls(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    train_val_split = train_val_split_cls(n_splits=1, test_size=val_size, random_state=random_state)
    splits = []
    for train_val_idx, test_idx in k_fold.split(indices, y=stratify):
        # Exclude the test set from the remaining data to split
        train_val_indices = indices[train_val_idx]
        train_val_stratify = stratify[train_val_idx] if stratify is not None else None

        train_idx, val_idx = next(train_val_split.split(indices[train_val_idx], train_val_stratify))

        splits.append(
            {TRAIN_SET: train_val_indices[train_idx], VAL_SET: train_val_indices[val_idx], TEST_SET: indices[test_idx]}
        )

    # Convert arrays of int64 (e.g. returned by `KFold.split`) to a sorted native int list
    # to avoid serialization issues if the caller tries to save the splits to disk
    splits = apply_to_collection(splits, np.ndarray, lambda x: np.sort(x).tolist())

    return splits


def main():
    """Run the script."""
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = Patients.add_args(parser)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where to save the files listing the patients making up each splits",
    )
    parser.add_argument(
        "--stratify_attr",
        required=True,
        type=TabularAttribute,
        choices=list(TabularAttribute),
        help="Name of the tabular attribute whose distribution in each of the subset should be similar",
    )
    parser.add_argument("--n_splits", type=int, default=5, help="Number of cross-validation folds to generate")
    parser.add_argument(
        "--val_size",
        type=int_or_float,
        default=20,
        help="If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the "
        "validation set. If int, represents the absolute number of validation samples.",
    )
    parser.add_argument("--seed", type=int, help="Seed to control the reproducibility of the split")
    args = parser.parse_args()
    kwargs = vars(args)

    output_dir, stratify_attr, n_splits, val_size, seed = (
        kwargs.pop("output_dir"),
        kwargs.pop("stratify_attr"),
        kwargs.pop("n_splits"),
        kwargs.pop("val_size"),
        kwargs.pop("seed"),
    )

    patients = Patients(**kwargs)
    patients_pbar = tqdm(patients.values(), desc="Collecting patients' data", unit="patient")
    # Collect the data of the attribute by which to stratify the split from the patient
    patients_stratify = [patient.attrs[stratify_attr] for patient in patients_pbar]

    splits = k_fold(list(patients), stratify=patients_stratify, n_splits=n_splits, val_size=val_size, random_state=seed)

    # Save the generated splits
    patient_ids = np.array(list(patients))
    for split_idx, split in enumerate(splits):
        (output_dir / str(split_idx)).mkdir(parents=True, exist_ok=True)
        for subset, subset_indices in split.items():
            (output_dir / str(split_idx) / f"{subset}.txt").write_text("\n".join(patient_ids[subset_indices]))


if __name__ == "__main__":
    main()
