import logging
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd
from tqdm.auto import tqdm

from vital.data.cardinal.config import TabularAttribute
from vital.data.cardinal.utils.attributes import TABULAR_ATTR_UNITS
from vital.data.cardinal.utils.itertools import Patients

logger = logging.getLogger(__name__)


def merge_records(
    patients: Patients,
    records: pd.DataFrame,
    output_dir: Path,
    subdir_levels: Sequence[Literal["patient"]] = None,
    progress_bar: bool = False,
) -> None:
    """Merges tabular attributes from patient records with existing tabular attributes provided by the dataset.

    Args:
        patients: Collection of patients with which to merge records.
        records: Records of new tabular data to merge with existing tabular attributes, indexed by patient.
        output_dir: Root of where to save the merging of records with existing tabular attributes, in the format of a
            YAML file for each patient.
        subdir_levels: Levels of subdirectories to create under the root output folder.
        progress_bar: If ``True``, enables progress bars detailing the progress of the merging of records with existing
            tabular attributes for each patient.
    """
    # Warn user if the list of patients in the collection and in the records does not match exactly
    records_patient_ids = records.index.tolist()
    patient_ids = list(patients)
    if patients_not_in_dataset := sorted(set(records_patient_ids) - set(patient_ids)):
        logger.warning(
            f"Patients with the following IDs: {patients_not_in_dataset} are present in the records but not in the "
            f"collection of patients. Their records will be ignored for the merging with the dataset."
        )
    if patients_not_in_records := sorted(set(patient_ids) - set(records_patient_ids)):
        logger.warning(
            f"Patients with the following IDs: {patients_not_in_records} are present in the dataset but not in the "
            f"provided records. The tabular data available in the dataset for these patients will remain unchanged."
        )

    patients = patients.values()
    msg = "Merging patient records with existing tabular data"
    if progress_bar:
        patients = tqdm(patients, desc=msg, unit="patient")
    else:
        logger.info(msg)

    for patient in patients:
        # Extract patient's tabular attributes from records, if they are available
        patient_record = {}
        if patient.id in records.index:
            patient_record = records.loc[patient.id]
            patient_record = patient_record[patient_record.notna()]  # Discard missing attributes
            patient_record = patient_record.to_dict()

        # Update the tabular attributes available for the patient based on the content of their record
        patient.attrs.update(patient_record)

        # Save the merged tabular attributes to a new yaml file on disk
        patient.save(output_dir, subdir_levels=subdir_levels, save_tabular_attrs=True, include_tags=[])


def read_records(records_csv: Path, col_names: Sequence[str] = None, drop_missing_data: bool = False) -> pd.DataFrame:
    """Loads and processes patient records to clean the data and keep only the attributes of interest.

    Args:
        records_csv: Path of the CSV file containing the patient records to read.
        col_names: Names of a subset of columns to keep from the records.
        drop_missing_data: Drop patient that are missing data for any of the requested `col_names`.

    Returns:
        Records of tabular data to process, indexed by patient.
    """
    # Eagerly determine what types to cast the data to, to avoid unsafe casts once the CSV has already been loaded
    cat_dtypes = {
        cat_attr: "boolean" if cat_attr in TabularAttribute.boolean_attrs() else "category"
        for cat_attr in TabularAttribute.categorical_attrs()
    }

    # When reading the file, cast boolean/categorical columns, but wait to manually cast numerical types later
    records = pd.read_csv(records_csv, index_col="AnonID", dtype=cat_dtypes)
    # Make sure to format the patient IDs in case the leading zeros were lost
    records = records.set_index(records.index.astype(str).str.zfill(4))

    # Manually convert integer columns to use pandas' `Int64` type, which supports missing values unlike the native int
    # We cannot rely on `read_csv`'s `dtype` param to do the casting because it results in an unsafe cast exception,
    # which pandas seems unwilling to fix (see this issue: https://github.com/pandas-dev/pandas/issues/37429)
    for int_attr in (
        num_attr for num_attr in TabularAttribute.numerical_attrs() if TABULAR_ATTR_UNITS[num_attr][1] == int
    ):
        records[int_attr] = records[int_attr].astype("Int64")

    if col_names:
        # Discard the columns not part of the requested subset
        records = records[col_names]

        if drop_missing_data:
            records = records[~(records.isna().any(axis="columns"))]

    return records


def main():
    """Run the script."""
    from argparse import ArgumentParser

    from vital.utils.logging import configure_logging

    configure_logging(log_to_console=True, console_level=logging.INFO)
    parser = ArgumentParser()
    parser = Patients.add_args(parser)
    parser.add_argument(
        "records_csv",
        type=Path,
        help="Path of the CSV file containing the patient records to merge with the existing tabular attributes",
    )
    parser.add_argument(
        "output_dir", type=Path, help="Root directory where to save the YAML files containing the results of the merge"
    )
    parser.add_argument(
        "--col_names", type=str, nargs="+", help="Names of a subset of columns to keep from the records"
    )
    parser.add_argument(
        "--drop_missing_data",
        action="store_true",
        help="Drop patient that are missing data for any of the requested `col_names`",
    )
    parser.add_argument(
        "--subdir_levels",
        type=str,
        nargs="+",
        choices=["patient"],
        help="Levels of subdirectories to create under the root output folder",
    )
    args = parser.parse_args()
    kwargs = vars(args)
    records_csv, col_names, drop_missing_data, output_dir, subdir_levels = (
        kwargs.pop("records_csv"),
        kwargs.pop("col_names"),
        kwargs.pop("drop_missing_data"),
        kwargs.pop("output_dir"),
        kwargs.pop("subdir_levels"),
    )

    # Load and pre-process patient records, for example selecting a subset of attributes or imputing missing values
    logger.info(f"Reading patient records from '{records_csv}'")
    records = read_records(records_csv, col_names=col_names, drop_missing_data=drop_missing_data)

    # Merge patient records' data with tabular attributes already provided for the patients
    merge_records(Patients(**kwargs), records, output_dir, subdir_levels=subdir_levels, progress_bar=True)


if __name__ == "__main__":
    main()
