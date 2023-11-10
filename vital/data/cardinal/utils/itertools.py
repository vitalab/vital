import logging
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Mapping, Optional, Sequence, Union

import pandas as pd
from pandas.api.types import CategoricalDtype

from vital.data.cardinal.config import IMG_FILENAME_PATTERN, IMG_FORMAT, PATIENT_ID_REGEX, TabularAttribute
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.utils.attributes import TABULAR_ATTR_UNITS, TABULAR_CAT_ATTR_LABELS
from vital.data.cardinal.utils.data_struct import Patient, View, load_attributes
from vital.utils.itertools import Collection

logger = logging.getLogger(__name__)


def views_avail_by_patient(data_roots: Sequence[Path], patient_id: str) -> List[ViewEnum]:
    """Searches for files related to a patient, to list all the views for which data is available for the patient.

    Args:
        data_roots: Root directories inside which to search recursively for files related to the patient.
        patient_id: ID of the patient for whom to search.

    Returns:
        Views for which data exists for the patient.
    """

    def extract_view_from_filename(filename: Path) -> ViewEnum:
        return ViewEnum(filename.name.split("_")[1])

    # Collect all files related to the patient from the multiple root directories
    patient_files_pattern = IMG_FILENAME_PATTERN.format(patient_id=patient_id, view="*", tag="*", ext=IMG_FORMAT)
    patient_files = []
    for data_root in data_roots:
        # Search recursively inside the provided directory
        patient_files.extend(data_root.rglob(patient_files_pattern))

    # Identify all the unique views across all the files related to the patient
    view_by_files = [extract_view_from_filename(patient_file) for patient_file in patient_files]
    views = sorted(set(view_by_files), key=lambda view: list(ViewEnum).index(view))
    return views


class Patients(Collection, Mapping[Patient.Id, Patient]):
    """Mapping between patient IDs and their data."""

    item = "patient"
    desc = item  # For compatibility with `ResultsProcessor` progress bar

    def __init__(
        self,
        data_roots: Sequence[Union[str, Path]],
        include_patients: Sequence[str] = None,
        exclude_patients: Sequence[str] = None,
        patient_attrs_filter: Sequence[Union[str, TabularAttribute]] = None,
        views: Sequence[Union[str, ViewEnum]] = None,
        handle_attrs_errors: Optional[Literal["warning", "error"]] = None,
        eager_loading: bool = False,
        overwrite_attrs_cache: bool = False,
        fast_dev_patients: int = None,
    ):
        """Initializes class instance.

        Args:
            data_roots: Folders in which to look for files related to the patient. We look for files recursively inside
                the directories, so as long as the files respect the `IMG_FILENAME_PATTERN` pattern, they will be
                collected.
            include_patients: Specific patients for which to collect data, in case not all patients available should be
                collected. If not specified, then all available patients will be collected (unless `exclude_patients` is
                specified).
            exclude_patients: Specific patients to exclude from the data to collect, in case not all patients available
                should be collected. If not specified, then all available patients will be collected (unless
                `include_patients` is specified).
            patient_attrs_filter: Attributes for which the patients have to have a value to be collected. This filter is
                applied after the filters on the specific patients to include/exclude.
            views: Specific views for which to collect data, in case not all views available should be collected. If not
                specified, then all available views will be collected by default.
            handle_attrs_errors: How to handle missing or duplicate attributes' file/entries in the file.
            eager_loading: By default, the `Patient`s' children `View` objects use lazy loading and only load images/
                compute attributes upon the first time they are accessed. Enabling eager loading will by-pass this
                behavior and force the children `View`s to load all the images and compute their attributes directly
                upon instantiation.
            overwrite_attrs_cache: Whether to discard the current cache of attributes and compute them again. Has no
                effect when no attributes cache exists.
            fast_dev_patients: In development mode, specify a limited number of patients on which to iterate. Only the
                first `n` will then be returned.
        """
        super().__init__()
        if include_patients and exclude_patients:
            raise ValueError(
                "`include_patients` and `exclude_patients` are mutually exclusive. Only one of them should be "
                "specified at a time."
            )

        data_roots = [Path(data_root) for data_root in data_roots]
        if views is not None:
            views = [ViewEnum[str(view)] for view in views]

        # Identify unique patient IDs across all the files in the different data roots
        # First we identify unique beginnings in file/folder names (where patient IDs should appear)
        # Then we filter these "candidates" through a regex to make sure they match the format expected of a patient ID
        # This allows to simply ignore non-compliant files in the folder tree, which can be useful to store additional
        # info, e.g. lists of patients by subsets, etc. along with the data
        patient_ids = sorted(
            patient_id_candidate
            for patient_id_candidate in set(
                file.stem.split("_")[0] for data_root in data_roots for file in data_root.rglob("*")
            )
            if re.match(PATIENT_ID_REGEX, patient_id_candidate)
        )

        if include_patients:
            # Filter patients to only include those that have been explicitly requested
            patient_ids = [patient_id for patient_id in patient_ids if patient_id in include_patients]
            if missing_patients := set(include_patients) - set(patient_ids):
                logger.warning(
                    f"No data found in provided data roots: {data_roots} for requested patients: {missing_patients}. "
                    f"The missing requested patients are simply ignored, but you might want to check your data folders "
                    f"if you expect all requested patients have available data."
                )
        if exclude_patients:
            # Filter patients to ignore those that have been explicitly excluded
            patient_ids = [patient_id for patient_id in patient_ids if patient_id not in exclude_patients]

        if patient_attrs_filter:
            # Make sure the requested patient attributes are recognized attributes
            patient_attrs_filter = [TabularAttribute[str(patient_attr)] for patient_attr in patient_attrs_filter]
            # Filter patients to only keep those who provide the requested attributes
            patient_ids = [
                patient_id
                for patient_id in patient_ids
                if set(patient_attrs_filter)
                <= load_attributes(patient_id, data_roots, handle_errors=handle_attrs_errors).keys()
            ]

        # Determine number of patients to iterate over
        num_patients = len(patient_ids)
        if fast_dev_patients is not None:
            if fast_dev_patients > num_patients:
                logger.warning(
                    f"'{self.__class__.__name__}' `fast_dev_patients` parameter set to iterate over "
                    f"{fast_dev_patients} patients, but data is only available for {num_patients} patients. "
                    f"'{self.__class__.__name__}' limited to the {num_patients} available patients."
                )
            else:
                num_patients = fast_dev_patients
        patient_ids = patient_ids[:num_patients]

        # Create the internal dictionary of patients
        self._patients = {
            patient_id: Patient.from_dir(
                patient_id,
                data_roots=data_roots,
                views=views,
                handle_attrs_errors=handle_attrs_errors,
                img_format=IMG_FORMAT,
                eager_loading=eager_loading,
                overwrite_attrs_cache=overwrite_attrs_cache,
            )
            for patient_id in patient_ids
        }

    @classmethod
    def from_dict(cls, patients: Dict[Patient.Id, Patient]) -> "Patients":
        """Builds an instance of the custom `Patients` collection from a builtin dictionary of patient IDs/data.

        Args:
            patients: Mapping between patient IDs and their data.

        Returns:
            `Patients` instance wrapped around the provided dict of `Patient`s.
        """
        # Use `__new__` to avoid having to provide an alternative path using a pre-built dict in `__init__`
        self = cls.__new__(cls)
        self._patients = patients
        return self

    def __getitem__(self, key: Patient.Id) -> Patient:  # noqa: D105
        return self._patients[key]

    def __iter__(self) -> Iterator[Patient.Id]:  # noqa: D105
        return iter(self._patients)

    def __len__(self) -> int:  # noqa: D105
        return len(self._patients)

    def to_dataframe(
        self, tabular_attrs: Sequence[TabularAttribute] = None, cast_to_pandas_dtypes: bool = True
    ) -> pd.DataFrame:
        """Converts the patients' tabular attrs to a dataframe, optionally handling missing values using pandas' dtypes.

        Notes:
            - In general, casting to pandas' dtype is recommended (hence it is done by default) since it can help handle
              missing values in a more standardized way (because pandas dtypes support missing values, while numpy's nan
              is only available for the float dtype). However, in some special cases pandas' `NAType` can cause problems
              (e.g. not being serializable when using the dataframe in holoviews/bokeh plots). These special cases are
              why the option to skip the pandas dtype casting exists.

        Args:
            tabular_attrs: Tabular attributes to include in the dataframe that is returned.
            cast_to_pandas_dtypes: Whether to cast the attributes to the most appropriate pandas dtype (e.g. Int64,
                boolean, category, etc.).

        Returns:
            A dataframe of the patients' tabular attributes.
        """
        if tabular_attrs is None:
            tabular_attrs = list(TabularAttribute)

        patients_df = pd.DataFrame.from_dict(
            {patient.id: {attr: patient.attrs.get(attr) for attr in tabular_attrs} for patient in self.values()},
            orient="index",
        ).rename_axis("patient")

        if cast_to_pandas_dtypes:
            cat_attrs = [attr for attr in tabular_attrs if attr in TabularAttribute.categorical_attrs()]
            num_attrs = [attr for attr in tabular_attrs if attr in TabularAttribute.numerical_attrs()]

            # Cast boolean/categorical columns
            cat_dtypes = {
                cat_attr: "boolean"
                if cat_attr in TabularAttribute.boolean_attrs()
                else CategoricalDtype(categories=TABULAR_CAT_ATTR_LABELS[cat_attr], ordered=True)
                for cat_attr in cat_attrs
            }
            patients_df = patients_df.astype(cat_dtypes)

            # Manually convert integer columns to use pandas' `Int64` type, which supports missing values unlike the
            # native int
            for int_attr in (num_attr for num_attr in num_attrs if TABULAR_ATTR_UNITS[num_attr][1] == int):
                patients_df[int_attr] = patients_df[int_attr].astype("Int64")

        return patients_df

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:  # noqa: D102
        parser = super().add_args(parser)
        parser.add_argument(
            "--data_roots",
            type=Path,
            nargs="+",
            help="Folders in which to look for files related to the patient. We look for files recursively inside the "
            "directories, so as long as the files respect the '{patient_id}_{view}_{tag}.{ext}' pattern, they will be "
            "collected.",
        )
        patients_parser = parser.add_mutually_exclusive_group(required=False)
        patients_parser.add_argument(
            "--include_patients",
            type=str,
            nargs="+",
            help="Specific patients for which to collect data, in case not all patients available should be collected. "
            "If not specified, then all available patients will be collected (unless `exclude_patients` is specified).",
        )
        patients_parser.add_argument(
            "--exclude_patients",
            type=str,
            nargs="+",
            help="Specific patients to exclude from the data to collect, in case not all patients available should be "
            "collected. If not specified, then all available patients will be collected (unless `include_patients` is "
            "specified).",
        )
        parser.add_argument(
            "--patient_attrs_filter",
            type=TabularAttribute,
            choices=list(TabularAttribute),
            nargs="+",
            help="Attributes for which the patients have to have a value to be collected. This filter is applied after "
            "the filters on the specific patients to include/exclude.",
        )
        parser.add_argument(
            "--views",
            type=ViewEnum,
            choices=list(ViewEnum),
            nargs="+",
            help="Specific views for which to collect data, in case not all views available should be collected. If "
            "not specified, then all available views will be collected by default.",
        )
        parser.add_argument(
            "--handle_attrs_errors",
            type=str,
            choices=["warning", "error"],
            help="How to handle missing or duplicate attributes' file/entries in the file",
        )
        parser.add_argument(
            "--eager_loading",
            action="store_true",
            help="By default, the `Patient`s' children `View` objects use lazy loading and only load images/compute "
            "attributes upon the first time they are accessed. Enabling eager loading will by-pass this behavior and "
            "force the children `View`s to load all the images and compute their attributes directly upon "
            "instantiation.",
        )
        parser.add_argument(
            "--overwrite_attrs_cache",
            action="store_true",
            help="Whether to discard the current cache of attributes and compute them again. Has no effect when no "
            "attributes cache exists.",
        )
        parser.add_argument(
            "--fast_dev_patients",
            type=int,
            help="In development mode, specify a limited number of patients on which to iterate. Only the first `n` "
            "will then be returned.",
        )
        return parser


class Views(Collection, Mapping[View.Id, View]):
    """Mapping between view IDs and their data."""

    item = "view"
    desc = item  # For compatibility with `ResultsProcessor` progress bar

    def __init__(self, **kwargs):
        super().__init__()

        self._patients = Patients(**kwargs)
        # Filter views to only include those that have been explicitly requested
        self._view_ids = [
            View.Id(patient.id, view_id) for patient in self._patients.values() for view_id in patient.views
        ]

    @classmethod
    def from_patients(cls, patients: Patients) -> "Views":
        """Builds an instance of the custom `Views` collection from a custom `Patients` collection.

        Args:
            patients: Mapping between patient IDs and their data.

        Returns:
            `Views` instance wrapped around the provided `Patients` collection.
        """
        # Use `__new__` to avoid having to provide an alternative path using a pre-built `Patients` in `__init__`
        self = cls.__new__(cls)
        self._patients = patients
        # Filter views to only include those that have been explicitly requested
        self._view_ids = [
            View.Id(patient.id, view_id) for patient in self._patients.values() for view_id in patient.views
        ]
        return self

    def __getitem__(self, key: View.Id) -> View:  # noqa: D105
        return self._patients[key.patient].views[key.view]

    def __iter__(self) -> Iterator[View.Id]:  # noqa: D105
        return iter(self._view_ids)

    def __len__(self) -> int:  # noqa: D105
        return len(self._view_ids)

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:  # noqa: D102
        return Patients.add_args(parser)
