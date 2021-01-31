from vital.data.acdc.config import Label
from vital.metrics.evaluate.anatomical_structure import Anatomical2DStructureMetrics
from vital.metrics.evaluate.segmentation import Segmentation2DMetrics


class RightVentricleMetrics(Anatomical2DStructureMetrics):
    """Class to compute metrics on the segmentation of the right ventricle."""

    def __init__(self, segmentation_metrics: Segmentation2DMetrics):  # noqa: D205,D212,D415
        """
        Args:
            segmentation_metrics: Segmentation2DMetrics, an instance, based on the segmentation for which to compute
                                  anatomical metrics, of the class implementing various segmentation metrics.
        """
        super().__init__(segmentation_metrics, Label.RV.value)
