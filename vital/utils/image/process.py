from abc import abstractmethod
from typing import Sequence

import numpy as np
from scipy import ndimage

from vital.data.config import SemanticStructureId
from vital.utils.format.native import flatten


class StructurePostProcessing:
    """Base post-processing class for post-processing applied to each individual structure of a segmentation map."""

    def __init__(self, labels: Sequence[SemanticStructureId]):  # noqa: D205,D212,D415
        """
        Args:
            labels: Labels of the structures to process in the segmentations.
        """
        self._labels = labels

    def __call__(self, seg: np.ndarray, **kwargs) -> np.ndarray:
        """Applies a specific post-processing algorithm on each individual structure in a segmentation map.

        Args:
            seg: (H, W), Segmentation to process.
            **kwargs: Capture non-used parameters to get a callable API compatible with similar callables.

        Returns:
            (H, W), Processed segmentation.
        """
        if seg.dtype != np.bool:  # If it is a categorical image containing multiple structures
            labels = np.unique(seg[seg.nonzero()])
            post_img = np.zeros_like(seg)
            i = 0
            for class_label in labels:
                label_image = np.isin(seg, class_label)
                if class_label in self._labels:
                    label_image = self._process_structure(label_image, class_label=i)
                    i += 1
                post_img[label_image] = class_label
        else:  # If it is a binary image containing only one structure
            post_img = self._process_structure(seg, )
        return post_img

    @abstractmethod
    def _process_structure(self, mask: np.ndarray, **kwargs) -> np.ndarray:
        """Applies a post-processing algorithm on the binary mask of a single structure from a segmentation map.

        Args:
            **kwargs:
            mask: (H, W), Binary mask of a semantic structure.

        Returns:
            (H, W), Processed binary mask of the semantic structure.
        """


class PostBigBlob(StructurePostProcessing):
    """Post-processing that returns only the biggest blob of non-zero value in a binary mask."""

    def __init__(self, labels: Sequence[SemanticStructureId], nb_blobs: Sequence[int] = None):
        super().__init__(labels)
        self.nb_blobs = nb_blobs or list(np.ones(len(self._labels), dtype=np.int8))
        assert len(self.nb_blobs) == len(self._labels)

    def _process_structure(self, mask: np.ndarray, class_label=None, **kwargs) -> np.ndarray:
        # Find each blob in the image
        lbl, num = ndimage.measurements.label(mask)

        # Count the number of elements per label
        count = np.bincount(lbl.flat)

        if not np.any(count[1:]):
            return mask

        # Sort the largest blobs
        ind = np.argsort(count[1:])[::-1]

        img = np.zeros_like(lbl)

        # Select only nb_blobs[class_label] largest blobs
        for i in ind[:min(self.nb_blobs[class_label], len(ind))]:
            img[lbl == i+1] = 1

        return img.astype(bool)


class PostFillIntraHoles(StructurePostProcessing):
    """Post-processing that fills holes inside the non-zero area of a binary mask."""

    def _process_structure(self, mask: np.ndarray, **kwargs) -> np.ndarray:
        return ndimage.binary_fill_holes(mask)


class PostFillInterHoles:
    """Post-processing that fills holes between two classes of a multi-class segmentation map with a new label."""

    def __init__(
        self, struct1_label: SemanticStructureId, struct2_label: SemanticStructureId, fill_label: int
    ):  # noqa: D205,D212,D415
        """
        Args:
            struct1_label: Label(s) of one of the semantic structure to postprocess.
            struct2_label: Label(s) of the other semantic structure to postprocess.
            fill_label: New label with which to fill holes between the semantic structures.
        """
        self._flattened_structs_labels = flatten([struct1_label, struct2_label])
        self._fill_label = fill_label

    def __call__(self, seg: np.ndarray, **kwargs) -> np.ndarray:
        """Fills holes between two classes of a multi-class segmentation map with a new label.

        Args:
            seg: (H, W), Segmentation to process.
            **kwargs: Capture non-used parameters to get a callable API compatible with similar callables.

        Returns:
            (H, W), Processed segmentation.
        """
        post_img = seg.copy()

        # Fill the holes between the two anatomical structures
        binary_merged_structs = np.isin(seg, self._flattened_structs_labels)
        binary_merged_structs_filled_holes = ndimage.binary_fill_holes(binary_merged_structs)

        # Identify the newly filled holes and assign them to the desired label
        inter_holes_mask = binary_merged_structs_filled_holes & ~binary_merged_structs
        post_img[inter_holes_mask] = self._fill_label

        return post_img