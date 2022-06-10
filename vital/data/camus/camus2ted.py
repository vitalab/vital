import logging
import shutil
from pathlib import Path

from tqdm import tqdm

from vital.utils.logging import configure_logging


def convert_camus_data_to_ted_format(camus_root: Path, ted_output: Path) -> None:
    """Converts data following the CAMUS format to follow the TED format.

    This format convert means renaming the patient IDs, and only keeping one set of annotations (the original CAMUS
    'cut' version) when two are available.

    Args:
        camus_root: Root directory of the CAMUS data to convert.
        ted_output: Output directory where to save the data converted to TED format.
    """
    camus_patient_dirs = sorted(list(camus_root.iterdir()))
    for idx, camus_patient_dir in tqdm(
        enumerate(camus_patient_dirs, 1),
        desc="Converting patient data from CAMUS format to TED format",
        total=len(camus_patient_dirs),
        unit="patient",
    ):
        camus_patient_id = camus_patient_dir.name
        ted_patient_id = f"patient{idx:03d}"  # Pad indices with less than 3 digits to 3 digits
        ted_patient_dir = ted_output / ted_patient_id

        # Create directory for patient in TED
        ted_patient_dir.mkdir(parents=True, exist_ok=True)

        # Detect if the CAMUS has two sets of annotations, depending on how the myocardium was cut at the base
        has_cut_version = any("_cut" in camus_file.name for camus_file in camus_patient_dir.iterdir())

        # Rename patient ID in filenames
        for camus_file in camus_patient_dir.iterdir():
            if has_cut_version and "_gt." in camus_file.name:
                # If a 'cut' annotation is available, discard the other annotation
                continue
            ted_name = camus_file.name.replace(camus_patient_id, ted_patient_id).replace("_cut", "")
            shutil.copy2(camus_file, ted_patient_dir / ted_name)

        # Make sure that RAW filenames in MHD files are also updated
        for ted_mhd_file in ted_patient_dir.glob("*.mhd"):
            # Read in the file
            with open(ted_mhd_file, "r") as file:
                mhd_data = file.read()

            # Replace the CAMUS patient ID and annotation identifier
            mhd_data = mhd_data.replace(camus_patient_id, ted_patient_id).replace("_cut", "")

            # Write the file out again
            with open(ted_mhd_file, "w") as file:
                file.write(mhd_data)


def main():
    """Run the script."""
    from argparse import ArgumentParser

    configure_logging(log_to_console=True, console_level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("camus_root", type=Path, help="Root directory of the CAMUS data to convert")
    parser.add_argument("ted_output", type=Path, help="Output directory where to save the data converted to TED format")
    args = parser.parse_args()

    convert_camus_data_to_ted_format(args.camus_root, args.ted_output)


if __name__ == "__main__":
    main()
