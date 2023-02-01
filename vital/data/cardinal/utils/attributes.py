from typing import Dict, Tuple, Union

import numpy as np

from vital.data.cardinal.config import ClinicalAttribute, ImageAttribute, Label
from vital.data.cardinal.config import View as ViewEnum
from vital.metrics.evaluate.clinical.heart_us import compute_left_ventricle_volumes
from vital.utils.image.measure import T
from vital.utils.image.us.measure import EchoMeasure


def compute_mask_attributes(mask: T, voxelspacing: Tuple[float, float]) -> Dict[ImageAttribute, T]:
    """Measures a variety of attributes on a (batch of) mask(s).

    Args:
        mask: ([N], H, W), Mask(s) on which to compute the attributes.
        voxelspacing: Size of the mask's voxels along each (height, width) dimension (in mm).

    Returns:
        Mapping between the attributes and ([N], 1) arrays of their values for each mask in the batch.
    """
    voxelarea = voxelspacing[0] * voxelspacing[1]
    return {
        ImageAttribute.gls: EchoMeasure.gls(mask, Label.LV, Label.MYO, voxelspacing=voxelspacing),
        ImageAttribute.lv_area: EchoMeasure.structure_area(mask, labels=Label.LV, voxelarea=voxelarea),
        ImageAttribute.lv_length: EchoMeasure.lv_length(mask, Label.LV, Label.MYO, voxelspacing=voxelspacing),
        ImageAttribute.myo_area: EchoMeasure.structure_area(mask, labels=Label.MYO, voxelarea=voxelarea),
    }


def compute_clinical_attributes(
    a4c_mask: np.ndarray,
    a4c_voxelspacing: Tuple[float, float],
    a2c_mask: np.ndarray,
    a2c_voxelspacing: Tuple[float, float],
    a4c_ed_frame: int = None,
    a4c_es_frame: int = None,
    a2c_ed_frame: int = None,
    a2c_es_frame: int = None,
) -> Dict[ClinicalAttribute, Union[int, float]]:
    """Measures a variety of clinical attributes based on masks from orthogonal views, i.e. A4C and A2C.

    Args:
        a4c_mask: (N1, H1, W1), Mask of the A4C view.
        a4c_voxelspacing: Size of the A4C mask's voxels along each (height, width) dimension (in mm).
        a2c_mask: (N2, H2, W2), Mask of the A2C view.
        a2c_voxelspacing: Size of the A2C mask's voxels along each (height, width) dimension (in mm).
        a4c_ed_frame: If the clinical attribute are computed on predictions rather than on reference, this is used to
            specify the index of the ED frame in the reference segmentation of the A4C view.
        a4c_es_frame: If the clinical attribute are computed on predictions rather than on reference, this is used to
            specify the index of the ES frame in the reference segmentation of the A4C view.
        a2c_ed_frame: If the clinical attribute are computed on predictions rather than on reference, this is used to
            specify the index of the ED frame in the reference segmentation of the A2C view.
        a2c_es_frame: If the clinical attribute are computed on predictions rather than on reference, this is used to
            specify the index of the ES frame in the reference segmentation of the A2C view.

    Returns:
        Mapping between the clinical attributes and their scalar values.
    """
    # Extract the relevant frames from the sequences (i.e. ED and ES) to compute clinical attributes
    lv_volumes_fn_kwargs = {}
    for view, lv_mask, voxelspacing, ed_frame, es_frame in [
        (ViewEnum.A4C, np.isin(a4c_mask, Label.LV), a4c_voxelspacing, a4c_ed_frame, a4c_es_frame),
        (ViewEnum.A2C, np.isin(a2c_mask, Label.LV), a2c_voxelspacing, a2c_ed_frame, a2c_es_frame),
    ]:
        voxelarea = voxelspacing[0] * voxelspacing[1]

        # Identify the ES frame in a view as the frame where the LV is the smallest (in 2D)
        if ed_frame is None:
            ed_frame = 0
        if es_frame is None:
            es_frame = np.argmin(EchoMeasure.structure_area(lv_mask, voxelarea=voxelarea))
        view_prefix = view.lower() + "_"
        lv_volumes_fn_kwargs.update(
            {
                view_prefix + "ed": lv_mask[ed_frame],
                view_prefix + "es": lv_mask[es_frame],
                view_prefix + "voxelspacing": voxelspacing,
            }
        )

    # Compute the clinical attributes
    edv, esv = compute_left_ventricle_volumes(**lv_volumes_fn_kwargs)
    ef = int(round(100 * (edv - esv) / edv))

    return {ClinicalAttribute.ef: ef, ClinicalAttribute.edv: edv, ClinicalAttribute.esv: esv}
