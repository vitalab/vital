import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import morphology

from vital.data.mri.config import Label
from vital.metrics.evaluate.segmentation import Segmentation2DMetrics


class FrontierMetrics:
    """Class to compute metrics on the frontiers between segmentations of multiple anatomical structures."""

    def __init__(self, segmentation_metrics: Segmentation2DMetrics):  # noqa: D205,D212,D415
        """
        Args:
            segmentation_metrics: Segmentation2DMetrics, an instance, based on the segmentation for which to compute
                                  anatomical metrics, of the class implementing various segmentation metrics.
        """
        self.segmentation_metrics = segmentation_metrics

    def holes_between_lv_and_myo(self):
        """Counts the pixels in the gap between the two areas segmented as LV and MYO.

        Returns:
             int, the count of pixels in the gap between the two areas segmented as LV and MYO.
        """
        return self.segmentation_metrics.count_holes_between_regions(Label.LV.value, Label.MYO.value)

    def holes_between_rv_and_myo(self):
        """Counts the pixels in the gap between the two areas segmented as RV and MYO.

        Returns:
             int, the count of pixels in the gap between the two areas segmented as RV and MYO.
        """
        return self.segmentation_metrics.count_holes_between_regions(Label.RV.value, Label.MYO.value)

    def rv_disconnected_from_myo(self):
        """Measures the gap (in mm) between the areas segmented as RV and MYO.

        Returns:
             float, the width of the gap (in mm) between the areas segmented as RV and MYO.
        """
        infinity = (
            1.5 * self.segmentation_metrics.segmentation.size * max(self.segmentation_metrics.voxelspacing)
        )  # this should be bigger than any distance between 2 pixels!
        rv_as_1 = morphology.dilation(
            np.isin(self.segmentation_metrics.segmentation, Label.RV.value).astype(dtype=np.uint8), np.ones((3, 3))
        )

        myo_as_0 = np.isin(self.segmentation_metrics.segmentation, Label.MYO.value, invert=True).astype(dtype=np.uint8)
        if rv_as_1.sum() > 0 and (1 - myo_as_0).sum() > 0:  # If both rv and myo are present in the image
            distance_to_myo = distance_transform_edt(myo_as_0, self.segmentation_metrics.voxelspacing)
            min_distance_rv_to_myo = (rv_as_1 * distance_to_myo + infinity * (1 - rv_as_1)).min()
            return min_distance_rv_to_myo
        else:  # If the image has no rv and myo
            return 0

    def frontier_between_lv_and_rv(self):
        """Counts the pixels that touch between the areas segmented as LV and RV.

        Returns:
             int, the count of pixels that touch between the areas segmented as LV and RV.
        """
        return self.segmentation_metrics.count_frontier_between_regions(Label.LV.value, Label.RV.value)

    def frontier_between_lv_and_background(self):
        """Counts the pixels that touch between the areas segmented as LV and background.

        Returns:
             int, the count of pixels that touch between the areas segmented as LV and background.
        """
        return self.segmentation_metrics.count_frontier_between_regions(Label.LV.value, Label.BG.value)
