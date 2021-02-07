import numpy as np

from vital.data.acdc.config import Label


def measure_ejection_fraction_error(ed_seg, es_seg, gt_ed_seg, gt_es_seg, ed_voxel, es_voxel, seg_class):
    """Measures the ejection fraction absolute error for a patient based on the 3D segmentations at ED and ES.

    Args:
        ed_seg: a 3D array of dimensions (slices, height, width) that correspond to the segmentations at diastole end.
        es_seg: a 3D array of dimensions (slices, height, width) that correspond to the segmentations at systole end.
        gt_ed_seg: a 3D array of dimensions (slices, height, width) that correspond to the ground truth at diastole end.
        gt_es_seg: a 3D array of dimensions (slices, height, width) that correspond to the ground truth at systole end.
        ed_voxel: voxel size of the systolic segmentations.
        es_voxel: voxel size of the diastolic segmentations.
        seg_class: int, class label to compute the EF absolute error for.

    Returns:
        The ejection fraction absolute error for the patient's LV.
    """
    pixels_count_ed = np.count_nonzero(ed_seg == seg_class)
    pixels_count_es = np.count_nonzero(es_seg == seg_class)
    ef_pred = np.round(100 * (pixels_count_ed - pixels_count_es) / pixels_count_ed, decimals=3)
    gt_pixels_count_ed = np.count_nonzero(gt_ed_seg == seg_class)
    gt_pixels_count_es = np.count_nonzero(gt_es_seg == seg_class)
    ef_gt = np.round(100 * (gt_pixels_count_ed - gt_pixels_count_es) / gt_pixels_count_ed, decimals=3)
    return np.abs(ef_gt - ef_pred)


def measure_ejection_fraction_error_lv(ed_seg, es_seg, gt_ed_seg, gt_es_seg, ed_voxel, es_voxel):
    """Measures the ejection fraction absolute error for a patient based on the 3D segmentations at ED and ES.

    Args:
        ed_seg: a 3D array of dimensions (slices, height, width) that correspond to the segmentations at diastole end.
        es_seg: a 3D array of dimensions (slices, height, width) that correspond to the segmentations at systole end.
        gt_ed_seg: a 3D array of dimensions (slices, height, width) that correspond to the ground truth at diastole end.
        gt_es_seg: a 3D array of dimensions (slices, height, width) that correspond to the ground truth at systole end.
        ed_voxel: voxel size of the systolic segmentations.
        es_voxel: voxel size of the diastolic segmentations.

    Returns:
        The ejection fraction absolute error for the patient's LV.
    """
    return measure_ejection_fraction_error(ed_seg, es_seg, gt_ed_seg, gt_es_seg, ed_voxel, es_voxel, Label.LV)


def measure_ejection_fraction_error_rv(ed_seg, es_seg, gt_ed_seg, gt_es_seg, ed_voxel, es_voxel):
    """Measures the ejection fraction absolute error for a patient based on the 3D segmentations at ED and ES.

    Args:
        ed_seg: a 3D array of dimensions (slices, height, width) that correspond to the segmentations at diastole end.
        es_seg: a 3D array of dimensions (slices, height, width) that correspond to the segmentations at systole end.
        gt_ed_seg: a 3D array of dimensions (slices, height, width) that correspond to the ground truth at diastole end.
        gt_es_seg: a 3D array of dimensions (slices, height, width) that correspond to the ground truth at systole end.
        ed_voxel: voxel size of the systolic segmentations.
        es_voxel: voxel size of the diastolic segmentations.

    Returns:
        The ejection fraction absolute error for the patient's RV.
    """
    return measure_ejection_fraction_error(ed_seg, es_seg, gt_ed_seg, gt_es_seg, ed_voxel, es_voxel, Label.RV)


def measure_volume_lv(seg, voxel):
    """Measure the volume of the 3D segmentation.

    Args:
        seg: ndarray, a 3D array of dimensions (slices, height, width) that correspond to the segmentations.
        voxel: list, voxel size corresponding to the 3D segmentation.

    Returns:
        float, volume of the 3d segmentation.
    """
    lv_vol = np.count_nonzero(seg == Label.LV) * np.prod(voxel) / 1000
    return lv_vol
