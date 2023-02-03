import logging
from pathlib import Path
from typing import Sequence, Tuple, Union

import h5py
from tqdm.auto import tqdm

from vital.data.cardinal.config import HDF5_FILENAME_PATTERN, CardinalTag
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.utils.data_struct import View
from vital.utils.image.us.bmode import CartesianBMode, PolarBMode
from vital.utils.logging import configure_logging


def _extract_metadata_from_name(filepath_or_name: Union[str, Path]) -> Tuple[str, ViewEnum]:
    """Extracts metadata about the content of a data file from its name.

    Args:
        filepath_or_name: Path of a data file, or directly the name of the file (w/o the extension).

    Returns:
        Metadata associated with the data file, namely i) the patient ID and ii) the view.
    """
    if isinstance(filepath_or_name, Path):
        name = filepath_or_name.stem
    else:
        name = filepath_or_name
    try:
        patient_id, view = name.split("_")
        view = ViewEnum(view)
    except ValueError:
        raise RuntimeError(f"Filename {filepath_or_name} does not match the expected format '{HDF5_FILENAME_PATTERN}'.")
    return patient_id, view


def convert_ge_hdf5_to_nifti(
    ge_hdf5_files: Sequence[Path], output_dir: Path, target_img_size: Tuple[int, int] = None
) -> None:
    """Processes HDF5 files of polar image data into cartesian images, and saves them as NIFTI files.

    Args:
        ge_hdf5_files: Path to the GE-formatted HDF5 files, extracted from EchoPAC, to convert to a cartesian NIFTI
            format.
        output_dir: Root of where to save the processed image and clinical data of the patients.
        target_img_size: Target (height, width) at which to resize the images.
    """
    if output_dir.is_file():
        logging.warning(f"Output '{output_dir}' already exists as a file, and is going to be overwritten.")
        output_dir.unlink()
    elif output_dir.is_dir():
        logging.warning(
            f"Output directory '{output_dir}' already exists. Already existing files in that directory that do not "
            f"conflict with files to be created will be left untouched, but any conflicting files will be overwritten."
        )

    for ge_hdf5_filepath in tqdm(
        ge_hdf5_files, desc=f"Saving patient data to {output_dir}", total=len(ge_hdf5_files), unit="view"
    ):
        patient_id, view_tag = _extract_metadata_from_name(ge_hdf5_filepath)

        # Extract the image data from the file into a format we can manipulate, and save it to a `View` object
        view = View(id=(patient_id, view_tag))
        with h5py.File(ge_hdf5_filepath) as ge_hdf5_file:
            bmode = CartesianBMode.from_polar(PolarBMode.from_ge_hdf5(ge_hdf5_file), progress_bar=True)
            view.add_image(CardinalTag.bmode, bmode.data, voxelspacing=bmode.voxelspacing)

        if target_img_size:
            # Process the B-mode sequence to resize all sequences to a constant size
            # and add the resized sequence to the view
            resized_bmode, attrs, resized_tag = view.resize_image(
                CardinalTag.bmode, target_img_size=target_img_size, return_tag=True
            )
            view.data[resized_tag] = resized_bmode
            view.attrs[resized_tag] = attrs

        view.save(output_dir / patient_id)


def main():
    """Run the script."""
    from argparse import ArgumentParser

    configure_logging(log_to_console=True, console_level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "ge_hdf5_file",
        nargs="+",
        type=Path,
        help="Path to the GE-formatted HDF5 file, extracted from EchoPAC, to convert to a cartesian (NIFTI) format",
    )
    parser.add_argument("output_dir", type=Path, help="Root of where to save the processed image data of the patients")
    parser.add_argument("--image_size", type=int, nargs=2, help="Target height and width at which to resize the images")
    args = parser.parse_args()

    convert_ge_hdf5_to_nifti(args.ge_hdf5_file, args.output_dir, target_img_size=args.image_size)


if __name__ == "__main__":
    main()
