from typing import Dict, Tuple

from vital.data.cardinal.config import ImageAttribute, Label
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
