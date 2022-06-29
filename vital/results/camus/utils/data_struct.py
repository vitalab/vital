from dataclasses import dataclass
from numbers import Real
from typing import Mapping, Tuple

import h5py

from vital.data.camus.config import CamusTags
from vital.utils.data_struct import Group, filter_leaf_group, load_leaf_group_from_hdf5


@dataclass
class PatientResult:
    """Data structure that bundles references, results and metadata for one patient.

    Args:
        - id: Patient's identifier (in format "patient0123").
        - views: Mapping between each view available for the patient and its data.
    """

    id: str
    views: Mapping[str, "ViewResult"]

    @classmethod
    def from_hdf5(cls, patient: h5py.Group, **sequence_filter_kwargs) -> "PatientResult":
        """Constructs an instance of the data structure from its corresponding HDF5 hierarchical structure.

        Args:
            patient: Root of the HDF5 hierarchical structure to translate into a data structure's instance.
            sequence_filter_kwargs: Arguments to forward to the underlying 'ViewResult's to decide which frames to keep
                from each sequence.

        Returns:
            Instance of the data structure with values corresponding to the input HDF5 hierarchical structure.
        """
        return cls(
            id=patient.name.strip("/"),
            views={view: ViewResult.from_hdf5(patient[view], **sequence_filter_kwargs) for view in patient},
        )


@dataclass
class ViewResult(Group):
    """Data structure that bundles references, results and metadata for one sequence from a patient.

    Args:
        - id: Patient/view's identifier (in format "patient0123/{2CH|4CH}").
        - num_frames: Number of frames in the sequence, i.e. temporal resolution of the sequence.
        - voxelspacing: Size of the segmentations' voxels ang each (time, height, width) dimension (in mm).
    """

    id: str
    num_frames: int
    voxelspacing: Tuple[Real, Real, Real]

    @classmethod
    def from_hdf5(cls, view: h5py.Group, **sequence_filter_kwargs) -> "ViewResult":
        """Constructs an instance of the data structure from its corresponding HDF5 hierarchical structure.

        Args:
            view: Root of the HDF5 hierarchical structure to translate into a data structure's instance.
            sequence_filter_kwargs: Arguments to forward to `get_instants_by_view` to decide which frames to keep from
                the sequence.

        Returns:
            Instance of the data structure with values corresponding to the input HDF5 hierarchical structure.
        """
        from vital.results.camus.utils.itertools import get_instants_by_view

        instants = get_instants_by_view(view, **sequence_filter_kwargs)
        return cls(
            id=view.name.strip("/"),
            data={
                group_name: load_leaf_group_from_hdf5(group, keep_indices=instants)
                for group_name, group in view.items()
            },
            attrs=dict(view.attrs),
            num_frames=len(instants),
            voxelspacing=view.attrs[CamusTags.voxelspacing],
        )


@dataclass
class InstantResult(Group):
    """Data structure that bundles reference, result and metadata for an instant from one sequence of a patient.

    Args:
        - id: Patient/view/instant's identifier (in format "patient0123/{2CH|4CH}/i").
        - voxelspacing: Size of the segmentation's voxels along each (height, width) dimension (in mm).
    """

    id: str
    voxelspacing: Tuple[Real, Real]

    @classmethod
    def from_view(cls, view_result: ViewResult, instant: int) -> "InstantResult":
        """Constructs an instance of the instant's data structure from a data structure of a full sequence.

        Args:
            view_result: Data structure, containing results for a full sequence, from which to extract the instant's
                results.
            instant: Index of the instant in the sequence for which to extract the results.

        Returns:
            Instance of the data structure with values corresponding to the input HDF5 hierarchical structure.
        """
        return InstantResult(
            id=f"{view_result.id}/{instant}",
            data={
                group_name: filter_leaf_group(group, keep_indices=instant)
                for group_name, group in view_result.data.items()
            },
            attrs=view_result.attrs,
            voxelspacing=view_result.voxelspacing[1:],
        )
