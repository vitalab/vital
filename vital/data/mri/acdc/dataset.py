from pathlib import Path

from vital.data.config import Subset
from vital.data.mri.dataset import ShortAxisMRI, visualize_dataset


class Acdc(ShortAxisMRI):
    """Implementation of torchvision's ``VisionDataset`` for the ACDC dataset."""

    pass


"""
This script can be run to test and visualize the data from the dataset.
"""
if __name__ == "__main__":
    from argparse import ArgumentParser

    args = ArgumentParser(add_help=False)
    args.add_argument("path", type=str)
    args.add_argument("--use_da", action="store_true")
    args.add_argument("--predict", action="store_true")
    params = args.parse_args()

    ds = ShortAxisMRI(Path(params.path), image_set=Subset.TRAIN, predict=params.predict, use_da=params.use_da)

    visualize_dataset(ds, params.predict)
