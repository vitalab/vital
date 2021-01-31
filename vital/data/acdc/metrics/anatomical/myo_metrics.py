from skimage import measure

from vital.data.acdc.config import Label
from vital.metrics.evaluate.anatomical_structure import Anatomical2DStructureMetrics
from vital.metrics.evaluate.segmentation import Segmentation2DMetrics


class MyocardiumMetrics(Anatomical2DStructureMetrics):
    """Class to compute metrics on the segmentation of the myocardium."""

    def __init__(self, segmentation_metrics: Segmentation2DMetrics):  # noqa: D205,D212,D415
        """
        Args:
            segmentation_metrics: Segmentation2DMetrics, an instance, based on the segmentation for which to compute
                                  anatomical metrics, of the class implementing various segmentation metrics.
        """
        super().__init__(segmentation_metrics, Label.MYO.value)

    def count_holes(self):
        """Counts the pixels that form holes in the supposedly contiguous area segmented as myocardium.

        Returns:
            int, number of pixels that form holes in the segmented area as myocardium.
        """
        holes_on_myo_count = super().count_holes()

        if holes_on_myo_count > 0:  # If myo has holes, assume it is closed and account for lv amongst the holes

            # Count the number of pixels that make up the lv segmented area (that should form a hole inside myo)
            lv_binary_struct = self.segmentation_metrics.binary_structs[Label.LV.value]
            lv_props = measure.regionprops(measure.label(lv_binary_struct, connectivity=2))
            lv_pixel_count = 0

            if lv_props:  # If the lv is present in the image
                lv_pixel_count = lv_props[0].area

            # Subtract the number of pixels making up the lv segmented area from the holes
            return holes_on_myo_count - lv_pixel_count

        else:  # If myo has no holes, it is open and other metrics should indicate an anatomical error
            return 0
