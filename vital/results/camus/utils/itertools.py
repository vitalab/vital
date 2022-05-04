from argparse import ArgumentParser
from pathlib import Path
from typing import Iterator, List, Literal

import h5py

from vital.data.camus.config import CamusTags, FullCycleInstant
from vital.results.camus.utils.data_struct import InstantResult, PatientResult, ViewResult
from vital.results.utils.itertools import IterableResult


class _ResultOptionsMixin(IterableResult):
    """Mixin for configuring common options across result iterators."""

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
        self._fast_dev_patients = fast_dev_patients

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


class Patients(_ResultOptionsMixin, IterableResult[PatientResult]):
    """Iterable over each patient in an HDF5 results dataset."""

    desc = "patient"

    def __iter__(self) -> Iterator[PatientResult]:  # noqa: D105
        with h5py.File(self.results_path, "r") as results_file:
            for patient_count, patient_id in enumerate(results_file):
                if self._fast_dev_patients and patient_count > self._fast_dev_patients:
                    break
                yield PatientResult.from_hdf5(
                    results_file[patient_id], sequence=self.sequence, use_sequence=self.use_sequence
                )

    def __len__(self) -> int:  # noqa: D105
        with h5py.File(self.results_path, "r") as results_file:
            length = len(results_file)
        return length


class PatientViews(_ResultOptionsMixin, IterableResult[ViewResult]):
    """Iterable over each patient/view in an HDF5 results dataset."""

    desc = f"{Patients.desc}/view"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patients = Patients(**kwargs)

    def __iter__(self) -> Iterator[ViewResult]:  # noqa: D105
        for patient in self.patients:
            for view_result in patient.views.values():
                yield view_result

    def __len__(self) -> int:  # noqa: D105
        return sum(len(patient.views) for patient in self.patients)


class PatientViewInstants(_ResultOptionsMixin, IterableResult[InstantResult]):
    """Iterable over each patient/view/instant in an HDF5 results dataset."""

    desc = f"{PatientViews.desc}/instant"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patient_views = PatientViews(**kwargs)

    def __iter__(self) -> Iterator[InstantResult]:  # noqa: D105
        for view_result in self.patient_views:
            for instant in range(view_result.num_frames):
                yield InstantResult.from_view(view_result, instant)

    def __len__(self) -> int:  # noqa: D105
        return sum(view.num_frames for view in self.patient_views)


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
