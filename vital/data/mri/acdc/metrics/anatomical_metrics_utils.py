from vital.data.mri.acdc.metrics.anatomical.config import domains, ideal_value, thresholds
from vital.data.mri.acdc.metrics.anatomical.frontier_metrics import FrontierMetrics
from vital.data.mri.acdc.metrics.anatomical.lv_metrics import LeftVentricleMetrics
from vital.data.mri.acdc.metrics.anatomical.myo_metrics import MyocardiumMetrics
from vital.data.mri.acdc.metrics.anatomical.rv_metrics import RightVentricleMetrics
from vital.data.mri.acdc.metrics.anatomical.score import score_metric
from vital.data.mri.config import Label
from vital.metrics.evaluate.segmentation import Segmentation2DMetrics, check_metric_validity


def check_segmentation_validity(segmentation, voxelspacing, **kwargs):
    """Wrapper around the function that computes anatomical metrics for segmentations.

    Is only concerned with whether any anatomical errors where detected in the segmentation or not.

    Args:
        segmentation: ndarray, segmentation for which to compute anatomical metrics.
        voxelspacing: tuple, the size of the segmentation's voxels along each (height, width) dimension (in mm).
        **kwargs: additional parameters.

    Returns:
        bool, whether the segmentation is anatomically plausible (True) or not (False).
    """
    return not _extract_anatomical_metrics_by_segmentation(segmentation, voxelspacing)["anatomical_errors"]


def check_segmentation_score(segmentation, voxelspacing, **kwargs):
    """Wrapper around the function that computes anatomical metrics for segmentations.

    Only concerned with whether any anatomical errors where detected in the segmentation or not.

    Args:
        segmentation: ndarray, segmentation for which to compute anatomical metrics.
        voxelspacing: tuple, the size of the segmentation's voxels along each (height, width) dimension (in mm).
        **kwargs: additional parameters.

    Returns:
        bool, whether the segmentation is anatomically plausible (True) or not (False).
    """
    return _extract_anatomical_metric_scores_by_segmentation(segmentation, voxelspacing)["anatomical_score"]


def _extract_anatomical_metrics_by_segmentation_wrapper(parameters):
    """Wraps `extract_anatomical_metrics_by_segmentation` so that it can be called from the multiprocessing API.

    Args:
        parameters: tuple, parameters for `extract_anatomical_metrics_by_segmentation` packed in a tuple

    Returns:
        dict, a dictionary listing the segmentation map's anatomical metrics.
    """
    img_id, (segmentation, voxelspacing) = parameters

    # First create the dictionary with index key, then update it with values of different metrics
    # This is done so that the insertion order of the keys is the same as the order in which we want them displayed
    img_id_metrics = {"patientName_frame_slice": img_id}
    img_id_metrics.update(_extract_anatomical_metrics_by_segmentation(segmentation, voxelspacing=voxelspacing))
    img_id_metrics.update(_extract_anatomical_metric_scores_by_segmentation(segmentation, voxelspacing=voxelspacing))
    return img_id_metrics


def _extract_anatomical_metrics_by_segmentation(segmentation, voxelspacing):
    """Computes the anatomical metrics for a single segmentation.

    Args:
        segmentation: ndarray, the segmentation as a 2d array.
        voxelspacing: tuple, the size of the segmentation's voxels along each (height, width) dimension (in mm).

    Returns:
        dict, mapping between the anatomical metrics' names and their value for the segmentation.
    """
    segmentation_metrics = Segmentation2DMetrics(
        segmentation, [Label.BG.value, Label.LV.value, Label.MYO.value, Label.RV.value], voxelspacing=voxelspacing
    )
    lv_metrics = LeftVentricleMetrics(segmentation_metrics)
    myo_metrics = MyocardiumMetrics(segmentation_metrics)
    rv_metrics = RightVentricleMetrics(segmentation_metrics)
    frontier_metrics = FrontierMetrics(segmentation_metrics)
    metrics = {
        "holes_in_lv": lv_metrics.count_holes(),
        "holes_in_rv": rv_metrics.count_holes(),
        "holes_in_myo": myo_metrics.count_holes(),
        "disconnectivity_in_lv": lv_metrics.count_disconnectivity(),
        "disconnectivity_in_rv": rv_metrics.count_disconnectivity(),
        "disconnectivity_in_myo": myo_metrics.count_disconnectivity(),
        "holes_between_lv_and_myo": frontier_metrics.holes_between_lv_and_myo(),
        "holes_between_rv_and_myo": frontier_metrics.holes_between_rv_and_myo(),
        "rv_disconnected_from_myo": frontier_metrics.rv_disconnected_from_myo(),
        "frontier_between_lv_and_rv": frontier_metrics.frontier_between_lv_and_rv(),
        "frontier_between_lv_and_background": frontier_metrics.frontier_between_lv_and_background(),
        "lv_concavity": lv_metrics.measure_concavity(),
        "rv_concavity": rv_metrics.measure_concavity(),
        "myo_concavity": myo_metrics.measure_concavity(),
        "lv_circularity": lv_metrics.measure_circularity(),
        "myo_circularity": myo_metrics.measure_circularity(),
    }
    metrics["anatomical_errors"] = any(
        not check_metric_validity(value, thresholds.get(name), optional_structure=True)
        for name, value in metrics.items()
    )
    return metrics


def _extract_anatomical_metric_scores_by_segmentation(segmentation, voxelspacing):
    """Computes the anatomical metrics for a single segmentation.

    Args:
        segmentation: ndarray, the segmentation as a 2d array.
        voxelspacing: tuple, the size of the segmentation's voxels along each (height, width) dimension (in mm).

    Returns:
        dict, mapping between the anatomical metrics' names and their value for the segmentation.
    """
    segmentation_metrics = Segmentation2DMetrics(
        segmentation, [Label.BG.value, Label.LV.value, Label.MYO.value, Label.RV.value], voxelspacing=voxelspacing
    )
    lv_metrics = LeftVentricleMetrics(segmentation_metrics)
    myo_metrics = MyocardiumMetrics(segmentation_metrics)
    rv_metrics = RightVentricleMetrics(segmentation_metrics)
    frontier_metrics = FrontierMetrics(segmentation_metrics)
    metrics = {
        "holes_in_lv": lv_metrics.count_holes(),
        "holes_in_rv": rv_metrics.count_holes(),
        "holes_in_myo": myo_metrics.count_holes(),
        "disconnectivity_in_lv": lv_metrics.count_disconnectivity(),
        "disconnectivity_in_rv": rv_metrics.count_disconnectivity(),
        "disconnectivity_in_myo": myo_metrics.count_disconnectivity(),
        "holes_between_lv_and_myo": frontier_metrics.holes_between_lv_and_myo(),
        "holes_between_rv_and_myo": frontier_metrics.holes_between_rv_and_myo(),
        "rv_disconnected_from_myo": frontier_metrics.rv_disconnected_from_myo(),
        "frontier_between_lv_and_rv": frontier_metrics.frontier_between_lv_and_rv(),
        "frontier_between_lv_and_background": frontier_metrics.frontier_between_lv_and_background(),
        "lv_concavity": lv_metrics.measure_concavity(),
        "rv_concavity": rv_metrics.measure_concavity(),
        "myo_concavity": myo_metrics.measure_concavity(),
        "lv_circularity": lv_metrics.measure_circularity(),
        "myo_circularity": myo_metrics.measure_circularity(),
    }

    metrics["anatomical_score"] = sum(
        score_metric(value, thresholds.get(name), domains.get(name), ideal_value.get(name, 0), optional_structure=True)
        / len(metrics)
        for name, value in metrics.items()
    )
    return metrics
