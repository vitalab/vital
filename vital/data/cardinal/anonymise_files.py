import logging
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Mapping, Tuple

import h5py
import pandas as pd
from pathos.multiprocessing import Pool
from tqdm.auto import tqdm

from vital.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def get_id_mapping_from_file(id_mapping_file: Path) -> Dict[Tuple[str, str], str]:
    """Read mapping between sensitive patient IDs and anonymized IDs from a csv file.

    Args:
        id_mapping_file: Path to the csv file containing the mapping.

    Returns:
        Mapping where the keys are sensitive patient IDs and image date pairs and the values are the anonymized IDs.
    """
    id_mapping_df = pd.read_csv(
        id_mapping_file, converters={"PatientID": str, "ImageDate": str, "AnonID": str}, delimiter=";"
    )
    id_mapping = dict(zip(zip(id_mapping_df.PatientID, id_mapping_df.ImageDate), id_mapping_df.AnonID))
    return id_mapping


def _extract_and_check_metadata(sensitive_file: Path) -> Tuple[str, str, str]:
    """Extracts metadata about a sensitive file from its filename.

    Notes:
        - In the case of an HDF5 file, the file is assumed to be following GE's standard, proprietary format. The
          metadata extracted from the filename is then compared to the metadata in the file's attributes. This is done<
          to catch possible data manipulation errors.

    Args:
        sensitive_file: Path to a file traceable back to a patient. The filename itself should follow the convention:
            '{PatientID}_{ImageDate}_{View}.h5'.

    Returns:
        Metadata associated with the file, namely i) the patient ID, ii) the date of the acquisition and iii) the view.
    """
    if is_h5_file := sensitive_file.suffix == ".h5":
        with h5py.File(sensitive_file, "r") as h5_file:
            attr_image_date = h5_file.attrs["ImageDate"].decode()
    filename_patient_id, filename_image_date, filename_view = sensitive_file.stem.rsplit("_", maxsplit=2)
    if is_h5_file and attr_image_date != filename_image_date:
        raise AssertionError(
            f"In file '{sensitive_file}', ImageDate in the attributes does not match the ImageDate in the filename. \n"
            f"Attribute ImageDate: {attr_image_date} \n"
            f"Filename ImageDate: {filename_image_date}"
        )
    return filename_patient_id, filename_image_date, filename_view


def create_anonymized_copy(
    sensitive_filepath: Path, id_mapping: Mapping[Tuple[str, str], str], output_dir: Path
) -> None:
    """Copies the content of `sensitive_file` to `output_dir`, but renames files and forgets sensitive attributes.

    Args:
        sensitive_filepath: The file with sensitive patient information to copy.
        id_mapping: Mapping where the keys are sensitive patient IDs and image date pairs and the values are the
            anonymized IDs.
        output_dir: The directory in which to save the anonymized copy of `sensitive_file`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the ID in the filename matches the ID in the metadata
    # This step could change depending on the naming convention of the files to anonymize, but with the current
    # convention it can detect files that were not named appropriately because of human error
    patient_id, image_date, view = _extract_and_check_metadata(sensitive_filepath)

    # Get the anonymized ID associated to the sensitive patient ID
    anon_id = id_mapping[(patient_id, image_date)]

    anon_filepath = output_dir / f"{anon_id}_{view}{sensitive_filepath.suffix}"
    if anon_filepath.is_file():
        logger.warning(f"Anonymized file '{anon_filepath}' will overwrite existing file")

    if sensitive_filepath.suffix == ".h5":
        # If it is an HDF5 file following GE's standard, proprietary format, copy the data without copying the sensitive
        # attributes at the root of the file
        with h5py.File(sensitive_filepath, "r") as sensitive_file:
            with h5py.File(anon_filepath, "w") as anon_file:
                # Copy datasets containing the sensitive file
                for dataset_name in sensitive_file:
                    sensitive_file.copy(dataset_name, anon_file)
    else:
        # Assume the only sensitive information is in the filename, and simply rename the file while copying it
        shutil.copy(sensitive_filepath, anon_filepath)


def main():
    """Run the script."""
    configure_logging(log_to_console=True, console_level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("sensitive_file", nargs="+", type=Path, help="Path to the sensitive file to anonymize")
    parser.add_argument(
        "id_mapping_file", type=Path, help="CSV file mapping sensitive patient IDs and image dates to anonymized IDs"
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory in which to save the anonmyised versions of the sensitive files"
    )
    parser.add_argument(
        "--disable_multiprocessing",
        dest="multiprocessing",
        action="store_false",
        help="Disables parallel processing of each sensitive file, useful to debug the script",
    )
    args = parser.parse_args()

    id_mapping = get_id_mapping_from_file(args.id_mapping_file)

    # Define single-parameter function
    def create_anonymized_copy_wrapper(sensitive_file: Path) -> None:
        create_anonymized_copy(sensitive_file, id_mapping, args.output_dir)

    # Create iterator over sensitive files
    if args.multiprocessing:
        pool = Pool()
        anonymisation_jobs = pool.imap(create_anonymized_copy_wrapper, args.sensitive_file)
    else:
        anonymisation_jobs = (create_anonymized_copy_wrapper(result) for result in args.sensitive_file)

    # Loop to execute jobs. Empty since the jobs return nothing
    for _ in tqdm(
        anonymisation_jobs,
        desc=f"Creating anonymized copies of files in {args.output_dir}",
        total=len(args.sensitive_file),
        unit="file",
    ):
        pass

    # In case of parallel processing, ensure the pool's resources are freed at the end
    if args.multiprocessing:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
