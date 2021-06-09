from vital.data.camus.config import Label
from vital.metrics.evaluate.segmentation import Segmentation2DMetrics


class FrontierMetrics:
    """Class to compute metrics on the frontiers between segmentations of multiple anatomical structures."""

    def __init__(self, segmentation_metrics: Segmentation2DMetrics):
        """Initializes class instance.

        Args:
            segmentation_metrics: Instance, based on the segmentation for which to compute anatomical metrics, of the
                class implementing various segmentation metrics.
        """
        self.segmentation_metrics = segmentation_metrics

    def count_holes_between_endo_and_epi(self) -> int:
        """Counts the pixels in the gap between the left ventricle endocardium and left ventricle epicardium.

        Returns:
            Count of pixels in the gap between the two areas segmented as left ventricle endocardium and left ventricle
            epicardium.
        """
        return self.segmentation_metrics.count_holes_between_regions(Label.ENDO.value, Label.EPI.value)

    def count_holes_between_endo_and_atrium(self) -> int:
        """Counts the pixels in the gap between the left ventricle endocardium and left atrium.

        Returns:
            Count of pixels in the gap between the two areas segmented as left ventricle endocardium and left atrium.
        """
        return self.segmentation_metrics.count_holes_between_regions(Label.ENDO.value, Label.EPI.value)

    def measure_frontier_ratio_between_endo_and_background(self) -> float:
        """Measures the ratio between the length of the frontier between the ENDO and BG and the size of the ENDO.

        Notes:
            - `ENDO` stands for "left ventricle endocardium", and `BG` stands for "background".

        Returns:
            Ratio between the length of the frontier between the ENDO and BG and the size of the ENDO.
        """
        return self.segmentation_metrics.measure_frontier_ratio_between_regions(Label.ENDO.value, Label.BG.value)

    def measure_frontier_ratio_between_epi_and_atrium(self) -> float:
        """Measures the ratio between the length of the frontier between the EPI and ENDO and the size of the EPI.

        Notes:
            - `EPI` stands for "left ventricle epicardium", and `ENDO` stands for "left ventricle endocardium".

        Returns:
            Ratio between the length of the frontier between the EPI and ENDO and the size of the EPI.
        """
        return self.segmentation_metrics.measure_frontier_ratio_between_regions(Label.EPI.value, Label.ATRIUM.value)
