import itertools
import logging
from pathlib import Path
from typing import Literal

from PIL import Image
from tqdm.auto import tqdm

from vital.utils.image.io import sitk_load
from vital.utils.logging import configure_logging
from vital.utils.path import remove_suffixes


def seq2pairs(
    sequence_file: Path,
    output_dir: Path,
    img_format: str = "png",
    pairs_order: Literal["forward", "backward"] = "forward",
    progress_bar: bool = False,
) -> None:
    """Unrolls a sequence as a series of pairs of images, since it`s the input format used by PWC-Net.

    Args:
        sequence_file: Path to the sequence, saved under a format readable by SimpleITK.
        output_dir: Root directory where to save the sequence as series of paired images
        img_format: Image format to use to save the images. Must be one of the format supported by Pillow.
        pairs_order: Ordering to generate between the images in each pair.
            - 'forward' means the first image is the first one chronologically in the sequence,
            - 'backward' the first image is the last one chronologically in the sequence
        progress_bar: If ``True``, enables progress bars detailing the progress of the generation of each pair of images
            from the sequence.
    """
    output_dir /= pairs_order
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_array, _ = sitk_load(sequence_file)

    if pairs_order == "forward":
        pairs_indices = itertools.pairwise(range(len(seq_array)))
    elif pairs_order == "backward":
        pairs_indices = itertools.pairwise(range(len(seq_array) - 1, -1, -1))
    else:
        raise ValueError(f"Unexpected value for 'pairs_order': {pairs_order}. Use either 'forward' or 'backward'.")

    if progress_bar:
        pairs_indices = tqdm(pairs_indices, desc=f"Saving pairs of images to '{output_dir}'", unit="pair", leave=False)
    for pair_idx, (idx_1, idx_2) in enumerate(pairs_indices):
        Image.fromarray(seq_array[idx_1]).save(output_dir / f"{pair_idx:02d}_img1.{img_format}")
        Image.fromarray(seq_array[idx_2]).save(output_dir / f"{pair_idx:02d}_img2.{img_format}")


def main():
    """Run the script."""
    from argparse import ArgumentParser

    configure_logging(log_to_console=True, console_level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument(
        "seq_file", nargs="+", type=Path, help="Path to the sequence, saved under a format readable by SimpleITK"
    )
    parser.add_argument(
        "output_dir", type=Path, help="Root directory where to save the sequence as series of paired images"
    )
    parser.add_argument(
        "--img_format",
        type=str,
        default="png",
        help="Image format to use to save the images. Must be one of the format supported by Pillow.",
    )
    parser.add_argument(
        "--pairs_order",
        choices=["forward", "backward"],
        nargs="+",
        default=["forward", "backward"],
        help="Ordering to generate between the images in each pair. "
        "'forward' means the first image is the first one chronologically in the sequence, "
        "'backward' the first image is the last one chronologically in the sequence",
    )
    args = parser.parse_args()

    for seq_file in tqdm(args.seq_file, desc="Unrolling sequences as pairs of images", unit="sequence"):
        for pairs_order in args.pairs_order:
            output_dir = args.output_dir / remove_suffixes(seq_file).name  # Handle multiple extensions, e.g. '.nii.gz'
            output_dir.mkdir(parents=True, exist_ok=True)
            seq2pairs(seq_file, output_dir, img_format=args.img_format, pairs_order=pairs_order, progress_bar=True)


if __name__ == "__main__":
    main()
