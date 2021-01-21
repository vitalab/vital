from dataclasses import dataclass, field
from numbers import Real
from typing import Mapping, Tuple

import h5py
import numpy as np
# from vital.data.acdc.config import ResultTags


@dataclass
class PatientData:
    """ Data structure that bundles data from the ACDC dataset for one patient.

    Args:
        id: patient's identifier (in format "patient0123").
    """
    id: str
    img: np.ndarray
    gt: np.ndarray
    # info: np.ndarray
    voxelspacing: np.ndarray
    registering: Mapping[str, np.ndarray] = field(default_factory=dict)


# @dataclass
# class PatientResult:
#     """ Data structure that bundles reference, result and metadata for one patient.
#
#     Args:
#         - id: patient's identifier (in format "patient123_{ES|ED}_0").
#         - gt: (N, H, W), unprocessed groundtruths, used as reference when evaluating methods' scores.
#         - pred: (N, H, W), labelled raw predictions (usually from argmax on output probabilities), that can be used as
#                 result when evaluating methods' scores.
#         - post_pred: (N, H, W), labelled post-processed predictions, that can be used as result when evaluating
#                      methods' scores.
#         - voxelspacing: the size of the segmentations' voxels along each (time, height, width) dimension (in mm).
#         - encoding: (N, ?), optional latent representation of the data, if the method relies on one.
#     """
#     id: str
#     gt: np.ndarray
#     pred: np.ndarray
#     post_pred: np.ndarray
#     voxelspacing: Tuple[Real, Real, Real]
#     encoding: np.ndarray = None
#
#     @classmethod
#     def from_hdf5(cls, patient: h5py.Group,
#                   filter_processed: bool = False) -> "PatientResult":
#         """ Constructs an instance of the data structure from its corresponding HDF5 hierarchical structure.
#
#         Args:
#             patient: root of the HDF5 hierarchical structure to translate into a data structure's instance.
#             filter_processed: whether to only include results that were affected by post-processed (useful to study
#                               the impact of the post-processing).
#
#         Returns:
#             an instance of the data structure with values corresponding to the input HDF5 hierarchical structure.
#         """
#         # instants = get_instants_by_patient(patient, filter_processed=filter_processed)
#         patient_result = cls(patient.name.strip('/'),
#                              patient[ResultTags.gt].value,
#                              patient[ResultTags.pred].value,
#                              patient[ResultTags.post_pred].value,
#                              tuple(patient.attrs[ResultTags.voxel_size]))
#
#         if ResultTags.encoding in patient:
#             patient_result.encoding = patient[ResultTags.encoding]
#         return patient_result
#
#
# @dataclass
# class InstantResult:
#     """ Data structure that bundles reference, result and metadata for one patient.
#
#     Args:
#         - id: patient/instant's identifier (in format "patient123_{ES|ED}_0").
#         - gt: (H, W), unprocessed ground truth, used as reference when evaluating methods' scores.
#         - pred: (H, W), labelled raw prediction (usually from argmax on output probabilities), that can be used as
#                 result when evaluating methods' scores.
#         - post_pred: (H, W), labelled post-processed prediction, that can be used as result when evaluating methods'
#                      scores.
#         - voxelspacing: the size of the segmentation's voxels along each (height, width) dimension (in mm).
#         - encoding: (?), optional latent representation of the data, if the method relies on one.
#     """
#     id: str
#     gt: np.ndarray
#     pred: np.ndarray
#     post_pred: np.ndarray
#     voxelspacing: Tuple[Real, Real]
#     encoding: np.ndarray = None
#
#     @classmethod
#     def from_patient(cls, patient_result: PatientResult, instant: int) -> "InstantResult":
#         """ Constructs an instance of the instant's data structure from a data structure of a full sequence.
#
#         Args:
#             patient_result: data structure, containing results for a full sequence, from which to extract the instant's
#                          results.
#             instant: index of the instant in the sequence for which to extract the results.
#
#         Returns:
#             an instance of the data structure with values corresponding to the input HDF5 hierarchical structure.
#         """
#         instant_result = InstantResult(id=f'{patient_result.id}/{instant}',
#                                        gt=patient_result.gt[instant],
#                                        pred=patient_result.pred[instant],
#                                        post_pred=patient_result.post_pred[instant],
#                                        voxelspacing=patient_result.voxelspacing)
#
#         if patient_result.encoding is not None:
#             instant_result.encoding = patient_result.encoding[instant]
#         return instant_result
