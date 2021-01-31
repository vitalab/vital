import json
import os
from os.path import join as pjoin

import h5py
import pandas as pd
from tqdm import tqdm

# from VITALabAI.utils.export import save_nii_file
from vital.data.acdc.metrics.clinical.clinical_metrics import (
    measure_ejection_fraction_error_lv,
    measure_ejection_fraction_error_rv,
    measure_volume_lv,
)

# from vital.datasets.acdc.preprocess import PreProcessResizeSeg


def extract_clinical_metrics(path, mode="pred_post", metrics_dir="CLIN_METRICS", post_process=None):
    """Computes and outputs to csv the clinical metrics of the predictions made by a trained network.

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
            "TRAIN_{}_CLIN_METRICS.csv".format(mode.upper()),
            "VALID_{}_CLIN_METRICS.csv".format(mode.upper()),
            "TEST_{}_CLIN_METRICS.csv".format(mode.upper()),
        ]
        for predictions_fname, metrics_fname in zip(predictions_fnames, metrics_fnames):
            extract_clinical_metrics_by_dataset(pjoin(path, predictions_fname), mode, pjoin(metrics_dir, metrics_fname))
    elif os.path.isfile(path):
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
        extract_clinical_metrics_by_dataset(path, mode, pjoin(metrics_dir, "CLIN_METRICS.CSV"), post_process)
    else:
        print(path, " Invalid")


def extract_clinical_metrics_by_dataset(
    predictions_fname, mode="pred_post", metrics_fname="CLIN_METRICS.csv", post_process=None
):
    """Computes the clinical metrics over the predictions made by an ACDC segmentation model and outputs them to csv.

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
    patient_ids = list(dict.fromkeys(group.split("_")[0] for group in file.keys()))

    def _segmentations():
        for patient_id in patient_ids:
            group_keys = map(lambda suffix: patient_id + suffix, ["_ED_0", "_ES_0"])
            patient_segs = []
            patient_gt_segs = []
            patient_voxels = []
            for group_key in group_keys:
                voxels = file[group_key].attrs.get("voxel_size")
                patient_height = file[group_key].attrs.get("height", 0)
                patient_weight = file[group_key].attrs.get("weight", 0)
                if len(voxels) == 4:
                    voxels = voxels[0:3]
                segs_3d = file[group_key][mode]
                gt_segs_3d = file[group_key]["gt_m"]
                if post_process:
                    p_slices = []
                    # segs_3d = PreProcessResizeSeg(size=(256, 256), num_classes=4)(segs_3d)
                    segs_3d = post_process(segs_3d, voxels, processed_slices=p_slices)
                    map_processed[group_key] = p_slices
                    # save_nii_file(segs_3d.transpose((1, 2, 0)),
                    #               pjoin(nifti_dir, group_key + "_post_pred.nii.gz"),
                    #               zoom=voxels,
                    #               dtype=np.uint8)
                patient_segs.append(segs_3d[()])
                patient_gt_segs.append(gt_segs_3d[()])
                patient_voxels.append(voxels)
            yield patient_segs, patient_voxels, patient_gt_segs, patient_height, patient_weight

    def _extract_clinical_metrics_by_segmentation(patient_id, segmentations, voxel_sizes, gt_segs, p_height, p_weight):
        patient_id_metrics = extract_clinical_metrics_by_patient(
            segmentations, voxel_sizes, gt_segs, p_height, p_weight
        )
        patient_id_metrics["patientName"] = patient_id
        return patient_id_metrics

    # The detailed metrics for each image
    # Each list element is a dictionary containing the values for all the metrics for an image in the dataset
    pbar = tqdm(
        zip(patient_ids, _segmentations()),
        unit="patient",
        desc="Computing metrics for dataset: {}".format(predictions_fname),
    )
    metrics = [
        _extract_clinical_metrics_by_segmentation(patient_id, segmentations, voxel_sizes, gt_segs, p_height, p_weight)
        for patient_id, (segmentations, voxel_sizes, gt_segs, p_height, p_weight) in pbar
    ]

    output_clinical_metrics_to_csv(metrics, metrics_fname)
    if post_process:
        with open(pjoin(os.path.dirname(metrics_fname), "processed_slices.json"), "w") as fj:
            json.dump(map_processed, fj, indent=2)


def extract_clinical_metrics_by_patient(segmentations, voxel_sizes, gt_segs, p_height=0, p_weight=0):
    """Computes the clinical metrics for a single patient.

    Args:
        segmentations: tuple of all the 2d array segmentation maps for a single patient.
        voxel_sizes: tuple of the segmentation maps' voxel sizes
                     (must contain the same number of elements as there are segmentations)
                     (it is an attribute in the ACDC segmentation models' predictions files).
        gt_segs:  tuple of all the 2d array ground truth for a single patient.
        p_height: height of the patient.
        p_weight: weight of the patient.


    Returns:
        A dictionary listing the patients' clinical metrics.
    """
    ed_seg, es_seg = segmentations
    gt_ed_seg, gt_es_seg = gt_segs
    ed_voxel_size, es_voxel_size = voxel_sizes
    # body_surface = 0.007184 * (p_height ** 0.725) * (p_weight ** 0.425)
    # body surface could be useful for computing other clinical parameters. kept for future use.
    metrics = {
        "ejection_fraction_error_lv": measure_ejection_fraction_error_lv(
            ed_seg, es_seg, gt_ed_seg, gt_es_seg, ed_voxel_size, es_voxel_size
        ),
        "ejection_fraction_error_rv": measure_ejection_fraction_error_rv(
            ed_seg, es_seg, gt_ed_seg, gt_es_seg, ed_voxel_size, es_voxel_size
        ),
        "end_diastolic_volume": measure_volume_lv(ed_seg, ed_voxel_size),
        "end_systolic_volume": measure_volume_lv(es_seg, es_voxel_size),
    }
    return metrics


def output_clinical_metrics_to_csv(metrics, metrics_fname):
    """Saves the computed metrics, with the aggregated results at the top, to csv format.

    Args:
        metrics: a list of dictionaries containing the computed metrics for each image in the dataset.
        metrics_fname: the name of the metrics' csv file to be produced as output.
    """
    # The columns' dictionary must be specified if we do not want the columns to be ordered by lexicographically
    metrics_columns = [
        "patientName",
        "ejection_fraction_error_lv",
        "ejection_fraction_error_rv",
        "end_diastolic_volume",
        "end_systolic_volume",
    ]
    metrics = pd.DataFrame(metrics, columns=metrics_columns)
    metrics.loc["average"] = metrics.mean(numeric_only=True)
    pd.DataFrame(metrics).to_csv(metrics_fname, index=False, columns=metrics_columns)


# if __name__ == "__main__":
#     import argparse
#     from VITALabAI.project.acdc.acdc_base import ACDCBase
#     from VITALabAI.dataset.semanticsegmentation.acdc.postprocess import PostProcessing
#
#     aparser = argparse.ArgumentParser()
#     aparser.add_argument("--path", type=str, required=True,
#                          help="Path to the output directory of an ACDC segmentation model.")
#     aparser.add_argument("--mode", type=str, choices=['pred_post', 'pred_m', 'gt_m'], default='pred_post',
#                          help="Segmentation maps on which to compute metrics.")
#     aparser.add_argument("--metrics_folder", type=str, default='',
#                          help="Path to the directory in which to save the metrics' csv files.")
#     ACDCBase.parse_segmentation_postproc(aparser)
#
#     args = aparser.parse_args()
#     if args.postproc_method is not None:
#         post_proc = PostProcessing.factory(**vars(args))
#     else:
#         post_proc = None
#     extract_clinical_metrics(args.path, args.mode, args.metrics_folder, post_process=post_proc)
