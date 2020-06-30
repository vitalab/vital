from vital.data.config import SemanticStructureId
from vital.metrics.evaluate.segmentation import Segmentation2DMetrics


class Anatomical2DStructureMetrics:
    """ Abstract class that implements overhead code to compute metrics for a specific anatomical structure in a
    segmentation.
    """

    def __init__(self, segmentation_metrics: Segmentation2DMetrics, struct_label: SemanticStructureId):
        """
        Args:
            segmentation_metrics: an instance, based on the segmentation for which to compute anatomical metrics, of
                                  the class implementing various segmentation metrics.
            struct_label: the label(s) of the class(es) making up the anatomical structure for which to compute
                          anatomical metrics.
        """
        self.segmentation_metrics = segmentation_metrics
        self.struct_label = struct_label
        self.no_structure_flag = float("nan")

    def count_holes(self) -> int:
        """ Counts the pixels that form holes in the segmentation of the anatomical structure.

        Returns:
            the count of pixels that form holes in the segmentation of the anatomical structure.
        """
        return self.segmentation_metrics.count_holes(self.struct_label)

    def count_disconnectivity(self) -> int:
        """ Counts the pixels that are disconnected from the main area segmented as the anatomical structure.

        Returns:
            the count of pixels that are disconnected from the main area segmented as the anatomical structure.
        """
        return self.segmentation_metrics.count_disconnectivity(self.struct_label)

    def measure_concavity(self) -> float:
        """ Measures the depth of a concavity in a supposedly convex anatomical structure.

        Returns:
             the depth (in mm) of a concavity in the anatomical structure.
        """
        return self.segmentation_metrics.measure_concavity(self.struct_label, no_structure_flag=self.no_structure_flag)

    def measure_circularity(self) -> float:
        """ Measures the isoperimetric ratio for an anatomical structure assuming the structure is contiguous.

        Returns:
            the isoperimetric ratio of the anatomical structure.
        """
        return self.segmentation_metrics.measure_circularity(
            self.struct_label, no_structure_flag=self.no_structure_flag
        )

    def measure_erosion_ratio_before_split(self) -> float:
        """ Measures the ratio between the depth of erosion necessary to divide a continuous anatomical structure in at
        least two fragments and the maximum thickness (in pixels) of the structure.

        Returns:
            the ratio between the depth of erosion necessary to divide a continuous anatomical structure in at least
            two fragments and the maximum thickness (in pixels) of the structure.
        """
        return self.segmentation_metrics.measure_erosion_ratio_before_split(
            self.struct_label, no_structure_flag=self.no_structure_flag
        )

    def measure_convexity(self) -> float:
        """  Measures the shape convexity of the anatomical structure's segmentation.

        Returns:
            the value of the shape convexity metric for the anatomical structure's segmentation.
        """
        return self.segmentation_metrics.measure_convexity(self.struct_label, no_structure_flag=self.no_structure_flag)
