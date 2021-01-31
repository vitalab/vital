import json
import os
from multiprocessing import Pool
from os.path import join as pjoin

import h5py
import pandas as pd
from tqdm import tqdm

from vital.data.acdc.config import Label
from vital.data.acdc.metrics.anatomical.config import domains, ideal_value, thresholds
from vital.data.acdc.metrics.anatomical.frontier_metrics import FrontierMetrics
from vital.data.acdc.metrics.anatomical.lv_metrics import LeftVentricleMetrics
from vital.data.acdc.metrics.anatomical.myo_metrics import MyocardiumMetrics
from vital.data.acdc.metrics.anatomical.rv_metrics import RightVentricleMetrics

# from vital.data.acdc.preprocess import PreProcessResizeSeg
from vital.data.acdc.metrics.anatomical.score import score_metric
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


def extract_anatomical_metrics(path, mode="pred_post", metrics_dir="METRICS", post_process=None):
    """Computes and outputs to csv the anatomical metrics of the predictions made by a trained network.

     Computed over training, validation and testing datasets.

    NOTE: The files TRAIN_PREDICTION, VALID_PREDICTION and TEST_PREDICTION must be present in the `path` directory
    when calling this script.

    Args:
        path: string, the to the output directory of an ACDC segmentation model.
        mode: string, segmentation maps on which to compute metrics.
        metrics_dir: string, path to the directory in which to save the metrics' csv files.
        post_process: list, list of post processing to apply to predictions.
    """
    if os.path.isdir(path):
        predictions_fnames = ["TRAIN_PREDICTION", "VALID_PREDICTION", "TEST_PREDICTION"]
        metrics_fnames = [
            "TRAIN_{}_METRICS.csv".format(mode.upper()),
            "VALID_{}_METRICS.csv".format(mode.upper()),
            "TEST_{}_METRICS.csv".format(mode.upper()),
        ]
        for predictions_fname, metrics_fname in zip(predictions_fnames, metrics_fnames):
            extract_anatomical_metrics_by_dataset(
                pjoin(path, predictions_fname), mode, pjoin(metrics_dir, metrics_fname)
            )
    elif os.path.isfile(path):
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
        extract_anatomical_metrics_by_dataset(path, mode, pjoin(metrics_dir, "METRICS.CSV"), post_process)
    else:
        print(path, " Invalid")


def extract_anatomical_metrics_by_dataset(
    predictions_fname, mode="pred_post", metrics_fname="METRICS.csv", post_process=None
):
    """Computes the anatomical metrics over the predictions made by an ACDC segmentation model and outputs them to csv.

    Args:
        predictions_fname: string, the name of the hd5 file containing the segmentation model's predictions for a
                           dataset.
        mode: string, segmentation maps on which to compute metrics.
        metrics_fname: string, the name of the metrics' csv file to be produced as output.
        post_process: list, list of post processing to apply to predictions.
    """
    file = h5py.File(predictions_fname, "r")
    if post_process:
        nifti_dir = pjoin(os.path.dirname(metrics_fname), "NIFTI/")
        if not os.path.isdir(nifti_dir):
            os.makedirs(nifti_dir)
        map_processed = {}
    # A list of identifiers for all the images in the dataset
    img_ids = ["%s_%s" % (group, slice_idx) for group in file.keys() for slice_idx in range(file[group][mode].shape[0])]

    def _segmentations():
        for group_key in file.keys():
            voxels = file[group_key].attrs.get("voxel_size")
            if len(voxels) == 4:
                voxels = voxels[0:3]
            segs_3d = file[group_key][mode]
            if post_process:
                p_slices = []
                # segs_3d = PreProcessResizeSeg(size=(256, 256), num_classes=len(Label))(segs_3d)
                segs_3d = post_process(segs_3d, voxels, processed_slices=p_slices)
                map_processed[group_key] = p_slices
                # save_nii_file(segs_3d.transpose((1, 2, 0)),
                #               pjoin(nifti_dir, group_key + "_post_pred.nii.gz"),
                #               zoom=voxels,
                #               dtype=np.uint8)
            for slice_idx in range(segs_3d.shape[0]):
                yield segs_3d[slice_idx], voxels[0:2]

    # Computes the number of segmentations iterated over by `_segmentations()`
    segmentations_count = sum(file[patient_id][mode].shape[0] for patient_id in file.keys())

    with Pool() as pool:
        # The detailed metrics for each image
        # Each list element is a dictionary containing the values for all the metrics for an image in the subfold
        metrics = list(
            tqdm(
                pool.imap(_extract_anatomical_metrics_by_segmentation_wrapper, zip(img_ids, _segmentations())),
                total=segmentations_count,
                unit="img",
                desc="Computing metrics for {} dataset".format(predictions_fname),
            )
        )

    _output_anatomical_metrics_to_csv(metrics, metrics_fname)
    if post_process:
        with open(pjoin(os.path.dirname(metrics_fname), "processed_slices.json"), "w") as fj:
            json.dump(map_processed, fj, indent=2)


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


def _output_anatomical_metrics_to_csv(metrics, metrics_fname):
    """Saves the computed metrics, with the aggregated results at the top, to csv format.

    Args:
        metrics: a list of dictionaries containing the computed metrics for each image in the dataset.
        metrics_fname: the name of the metrics' csv file to be produced as output.
    """
    # The columns' dictionary must be specified if we do not want the columns to be ordered by lexicographically
    metrics_columns = metrics[0].keys()
    metrics = pd.DataFrame(metrics, columns=metrics_columns)

    # Aggregate the metrics
    def count_metric_errors(metric_name):
        return lambda series: sum(
            not check_metric_validity(metric_value, thresholds.get(metric_name), optional_structure=True)
            for metric_value in series
        )

    aggregation_dict = {
        metric_name: count_metric_errors(metric_name)
        for metric_name in metrics.keys()
        if metric_name != "patientName_frame_slice"
    }
    aggregated_metrics = metrics.aggregate(aggregation_dict)
    aggregated_metrics["patientName_frame_slice"] = "anatomical_errors_count"

    # Write aggregated metrics at the top of the file
    pd.DataFrame(aggregated_metrics).T.to_csv(metrics_fname, index=False, columns=metrics_columns)

    # Append the detailed metrics for each image after the aggregated results
    pd.DataFrame(metrics).to_csv(metrics_fname, mode="a", header=False, index=False, na_rep="Nan")


if __name__ == "__main__":
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument(
        "--path", type=str, required=True, help="Path to the output directory of an ACDC segmentation model."
    )
    aparser.add_argument(
        "--mode",
        type=str,
        choices=["pred_post", "pred_m", "gt_m"],
        default="pred_post",
        help="Segmentation maps on which to compute metrics.",
    )
    aparser.add_argument(
        "--metrics_folder", type=str, default="", help="Path to the directory in which to save the metrics' csv files."
    )

    args = aparser.parse_args()
    extract_anatomical_metrics(args.path, args.mode, args.metrics_folder, post_process=args.postproc_method)
