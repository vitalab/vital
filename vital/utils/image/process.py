from abc import abstractmethod
from typing import Sequence

import numpy as np
from scipy import ndimage

from vital.data.config import SemanticStructureId
from vital.utils.decorators import batch_function
from vital.utils.format.native import flatten


class StructurePostProcessing:
    """Base post-processing class for post-processing applied to each individual structure of a segmentation map."""

    def __init__(self, labels: Sequence[SemanticStructureId]):
        """Initializes class instance.

        Args:
            labels: Labels of the structures to process in the segmentations.
        """
        self._labels = labels

    def __call__(self, seg: np.ndarray, **kwargs) -> np.ndarray:
        """Applies a specific post-processing algorithm on each individual structure in a segmentation map.

        Args:
            seg: Segmentation to process.
            **kwargs: Capture non-used parameters to get a callable API compatible with similar callables.

        Returns:
            (H, W), Processed segmentation.
        """
        if seg.dtype != np.bool:  # If it is a categorical image containing multiple structures
            post_img = np.zeros_like(seg)
            for class_label in self._labels:
                label_image = self._process_structure(np.isin(seg, class_label))
                post_img[label_image] = class_label
        else:  # If it is a binary image containing only one structure
            post_img = self._process_structure(seg)
        return post_img

    @abstractmethod
    def _process_structure(self, mask: np.ndarray) -> np.ndarray:
        """Applies a post-processing algorithm on the binary mask of a single structure from a segmentation map.

        Args:
            mask: Binary mask of a semantic structure.

        Returns:
            Processed binary mask of the semantic structure.
        """


class Post2DBigBlob(StructurePostProcessing):
    """Post-processing that returns only the biggest blob of non-zero value in a binary mask."""

    @staticmethod
    @batch_function(item_ndim=2)
    def _process_structure(mask: np.ndarray) -> np.ndarray:
        """Keeps only the biggest blob of non-zero value in a binary mask.

        Args:
            mask: ([N], H, W), Binary mask of a semantic structure.

        Returns:
            ([N], H, W), Processed binary mask.
        """
        # Find each blob in the image
        lbl, num = ndimage.measurements.label(mask)

        # Count the number of elements per label
        count = np.bincount(lbl.flat)

        if not np.any(count[1:]):
            return mask

        # Select the largest blob
        maxi = np.argmax(count[1:]) + 1

        # Remove the other blobs
        lbl[lbl != maxi] = 0

        return lbl.astype(bool)


class Post2DFillIntraHoles(StructurePostProcessing):
    """Post-processing that fills holes inside the non-zero area of a binary mask."""

    @staticmethod
    @batch_function(item_ndim=2)
    def _process_structure(mask: np.ndarray) -> np.ndarray:
        """Fills holes inside the non-zero area of a binary mask..

        Args:
            mask: ([N], H, W), Binary mask of a semantic structure.

        Returns:
            ([N], H, W), Processed binary mask.
        """
        return ndimage.binary_fill_holes(mask)


class Post2DFillInterHoles:
    """Post-processing that fills holes between two classes of a multi-class segmentation map with a new label."""

    def __init__(self, struct1_label: SemanticStructureId, struct2_label: SemanticStructureId, fill_label: int):
        """Initializes class instance.

        Args:
            struct1_label: Label(s) of one of the semantic structure to postprocess.
            struct2_label: Label(s) of the other semantic structure to postprocess.
            fill_label: New label with which to fill holes between the semantic structures.
        """
        self._flattened_structs_labels = flatten([struct1_label, struct2_label])
        self._fill_label = fill_label

    @batch_function(item_ndim=2)
    def __call__(self, seg: np.ndarray, **kwargs) -> np.ndarray:
        """Fills holes between two classes of a multi-class segmentation map with a new label.

        Args:
            seg: ([N], H, W), Segmentation to process.
            **kwargs: Capture non-used parameters to get a callable API compatible with similar callables.

        Returns:
            ([N], H, W), Processed segmentation.
        """
        post_img = seg.copy()

        # Fill the holes between the two anatomical structures
        binary_merged_structs = np.isin(seg, self._flattened_structs_labels)
        binary_merged_structs_filled_holes = ndimage.binary_fill_holes(binary_merged_structs)

        # Identify the newly filled holes and assign them to the desired label
        inter_holes_mask = binary_merged_structs_filled_holes & ~binary_merged_structs
        post_img[inter_holes_mask] = self._fill_label

        return post_img
