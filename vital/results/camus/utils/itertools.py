import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterator, List, Literal

import h5py

from vital.data.camus.config import CamusTags, FullCycleInstant
from vital.results.camus.utils.data_struct import InstantResult, PatientResult, ViewResult
from vital.utils.itertools import Collection

logger = logging.getLogger(__name__)


class Patients(Collection[PatientResult]):
    """Collection of patients from an HDF5 results dataset."""

    desc = "patient"

    def __init__(
        self,
        results_path: Path,
        sequence: Literal["half_cycle", "full_cycle"] = "full_cycle",
        use_sequence: bool = False,
        fast_dev_patients: int = None,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            results_path: Path of the HDF5 file of results.
            sequence: Flag to indicate which subset of the full cardiac cycle to include frames from.
            use_sequence: If ``True``, include all available frames in the sequence, and not only the clinically
                important instants in the cycle (ED, ES).
            fast_dev_patients: In development mode, specify a limited number of patients on which to iterate. Only the
                first `n` will then be returned.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)
        self.results_path = results_path
        self.sequence = sequence
        self.use_sequence = use_sequence

        with h5py.File(self.results_path, "r") as results_file:
            self.num_patients = len(results_file)
        if fast_dev_patients is not None:
            if fast_dev_patients > self.num_patients:
                logger.warning(
                    f"'{self.__class__.__name__}' `fast_dev_patients` parameter set to iterate over "
                    f"{fast_dev_patients} patients, but data is only available for {self.num_patients} patients. "
                    f"'{self.__class__.__name__}' limited to the {self.num_patients} available patients."
                )
            else:
                self.num_patients = fast_dev_patients

    def __iter__(self) -> Iterator[PatientResult]:  # noqa: D105
        with h5py.File(self.results_path, "r") as results_file:
            for patient_id in list(results_file)[: self.num_patients]:
                yield PatientResult.from_hdf5(
                    results_file[patient_id], sequence=self.sequence, use_sequence=self.use_sequence
                )

    def __len__(self) -> int:  # noqa: D105
        return self.num_patients

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:  # noqa: D102
        parser = super().add_args(parser)
        parser.add_argument("--results_path", type=Path, required=True, help="Path of the HDF5 file of results")
        parser.add_argument(
            "--sequence",
            type=str,
            default="full_cycle",
            choices=["half_cycle", "full_cycle"],
            help="Flag to indicate which (sub)set of the whole sequence to log results for. `'half-cycle'` will ignore "
            "all frames after the ES frame. If the data only contains half a cycle, both modes are equivalent.",
        )
        parser.add_argument(
            "--use_sequence",
            action="store_true",
            help="Log results for all available frames in the sequence. If this flag is not set, logs are only "
            "performed on clinically important instants in the cycle (typically ED and ES).",
        )
        parser.add_argument(
            "--fast_dev_patients",
            type=int,
            help="In development mode, specify a limited number of patients on which to iterate. Only the first `n` "
            "will then be returned.",
        )
        return parser


class PatientViews(Collection[ViewResult]):
    """Collection of patient/views from an HDF5 results dataset."""

    desc = f"{Patients.desc}/view"

    def __init__(self, **kwargs):
        super().__init__()
        self.patients = Patients(**kwargs)

    def __iter__(self) -> Iterator[ViewResult]:  # noqa: D105
        for patient in self.patients:
            for view_result in patient.views.values():
                yield view_result

    def __len__(self) -> int:  # noqa: D105
        return sum(len(patient.views) for patient in self.patients)

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:  # noqa: D102
        return Patients.add_args(parser)


class PatientViewInstants(Collection[InstantResult]):
    """Collection of patient/view/instants from an HDF5 results dataset."""

    desc = f"{PatientViews.desc}/instant"

    def __init__(self, **kwargs):
        super().__init__()
        self.patient_views = PatientViews(**kwargs)

    def __iter__(self) -> Iterator[InstantResult]:  # noqa: D105
        for view_result in self.patient_views:
            for instant in range(view_result.num_frames):
                yield InstantResult.from_view(view_result, instant)

    def __len__(self) -> int:  # noqa: D105
        return sum(view.num_frames for view in self.patient_views)

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:  # noqa: D102
        return PatientViews.add_args(parser)


def get_instants_by_view(
    view: h5py.Group, sequence: Literal["half_cycle", "full_cycle"] = "full_cycle", use_sequence: bool = False
) -> List[int]:
    """Determines relevant instants in the sequence, based on user-configurable criteria.

    Args:
        view: Root of the sequence's HDF5 hierarchical structure.
        sequence: Flag to indicate which subset of the full cardiac cycle to include frames from.
        use_sequence: If ``True``, include all available frames in the sequence, and not only the clinically important
            instants in the cycle (ED, ES).

    Returns:
        Relevant instants in the sequence, based on the users' criteria.
    """
    key_instants = {instant: view.attrs[instant] for instant in view.attrs[CamusTags.instants]}
    if sequence == "half_cycle":
        key_instants.pop(FullCycleInstant.ED_E, None)
    key_instants = sorted(key_instants.values())

    return list(range(key_instants[0], key_instants[-1] + 1)) if use_sequence else key_instants
