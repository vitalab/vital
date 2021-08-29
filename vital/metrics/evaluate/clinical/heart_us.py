# Disable flak8 entirely because this file was mostly copied over from external sources and
# the quality of the codebase and documentation is not up to par with the project's standards
# flake8: noqa
from numbers import Real
from typing import Tuple

import numpy as np
from skimage.measure import find_contours

from vital.utils.image.transform import resize_image


def compute_left_ventricle_volumes(
    a2c_ed: np.ndarray,
    a2c_es: np.ndarray,
    a2c_voxelspacing: Tuple[Real, Real],
    a4c_ed: np.ndarray,
    a4c_es: np.ndarray,
    a4c_voxelspacing: Tuple[Real, Real],
) -> Tuple[Real, Real]:
    """Computes the ED and ES volumes of the left ventricle from 2 orthogonal 2D views (A2C and A4C).

    Args:
        a2c_ed: (H,W), Binary segmentation map of the left ventricle from the end-diastole (ED) instant of the 2-chamber
            apical view (A2C).
        a2c_es: (H,W), Binary segmentation map of the left ventricle from the end-systole (ES) instant of the 2-chamber
            apical view (A2C).
        a2c_voxelspacing: Size (in mm) of the 2-chamber apical view's voxels along each (height, width) dimension.
        a4c_ed: (H,W), Binary segmentation map of the left ventricle from the end-diastole (ED) instant of the 4-chamber
            apical view (A4C).
        a4c_es: (H,W), Binary segmentation map of the left ventricle from the end-systole (ES) instant of the 4-chamber
            apical view (A4C).
        a4c_voxelspacing: Size (in mm) of the 4-chamber apical view's voxels along each (height, width) dimension.

    Returns:
        Left ventricle ED and ES volumes.
    """
    a2c_ed_diameters, a2c_ed_step_size = _compute_diameters(a2c_ed, a2c_voxelspacing)
    a2c_es_diameters, a2c_es_step_size = _compute_diameters(a2c_es, a2c_voxelspacing)
    a4c_ed_diameters, a4c_ed_step_size = _compute_diameters(a4c_ed, a4c_voxelspacing)
    a4c_es_diameters, a4c_es_step_size = _compute_diameters(a4c_es, a4c_voxelspacing)
    step_size = max((a2c_ed_step_size, a2c_es_step_size, a4c_ed_step_size, a4c_es_step_size))

    ed_volume = _compute_left_ventricle_volume_by_instant(a2c_ed_diameters, a4c_ed_diameters, step_size)
    es_volume = _compute_left_ventricle_volume_by_instant(a2c_es_diameters, a4c_es_diameters, step_size)
    return ed_volume, es_volume


def _compute_left_ventricle_volume_by_instant(
    a2c_diameters: np.ndarray, a4c_diameters: np.ndarray, step_size: Real
) -> Real:
    """Compute left ventricle volume using Biplane Simpson's method.

    Args:
        a2c_diameters: Diameters measured at each key instant of the cardiac cycle, from the 2-chamber apical view.
        a4c_diameters: Diameters measured at each key instant of the cardiac cycle, from the 4-chamber apical view.
        step_size:

    Returns:
        Left ventricle volume (in millilitres).
    """
    # All measures are now in millimeters, convert to meters by dividing by 1000
    a2c_diameters /= 1000
    a4c_diameters /= 1000
    step_size /= 1000

    # Estimate left ventricle volume from orthogonal disks
    lv_volume = np.sum(a2c_diameters * a4c_diameters) * step_size * np.pi / 4

    # Volume is now in cubic meters, so convert to milliliters (1 cubic meter = 1_000_000 milliliters)
    return round(lv_volume * 1e6)


def _find_distance_to_edge(
    segmentation: np.ndarray, point_on_mid_line: np.ndarray, normal_direction: np.ndarray
) -> Real:
    distance = 8  # start a bit in to avoid line stopping early at base
    while True:
        current_position = point_on_mid_line + distance * normal_direction

        y, x = np.round(current_position).astype(int)
        if segmentation.shape[0] <= y or y < 0 or segmentation.shape[1] <= x or x < 0:
            # out of bounds
            return distance

        elif segmentation[y, x] == 0:
            # Edge found
            return distance

        distance += 0.5


def _distance_line_to_points(line_point_0: np.ndarray, line_point_1: np.ndarray, points: np.ndarray) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    return np.absolute(np.cross(line_point_1 - line_point_0, line_point_0 - points)) / np.linalg.norm(
        line_point_1 - line_point_0
    )


def _get_angle_of_lines_to_point(reference_point: np.ndarray, moving_points: np.ndarray) -> np.ndarray:
    diff = moving_points - reference_point
    return abs(np.degrees(np.arctan2(diff[:, 0], diff[:, 1])))


def _reshape_image_to_isotropic(image: np.ndarray, spacing: Tuple[Real, Real]) -> Tuple[np.ndarray, Real]:
    real_aspect = (image.shape[0] * spacing[0]) / (image.shape[1] * spacing[1])
    current_aspect = (image.shape[0]) / (image.shape[1])
    new_height = int(image.shape[0] * (real_aspect / current_aspect))
    new_width = image.shape[1]
    return resize_image(image, (new_width, new_height)), spacing[1]


def _compute_diameters(segmentation: np.ndarray, voxelspacing: Tuple[Real, Real]) -> Tuple[np.ndarray, Real]:
    """

    Args:
        segmentation: Binary segmentation of the structure for which to find the diameter.
        voxelspacing: Size of the segmentations' voxels along each (height, width) dimension (in mm).

    Returns:
    """

    # Make image isotropic, have same spacing in both directions.
    # The spacing can be multiplied by the diameter directly.
    segmentation, isotropic_spacing = _reshape_image_to_isotropic(segmentation, voxelspacing)

    # Go through entire contour to find AV plane
    contour = find_contours(segmentation, 0.5)[0]

    # For each pair of contour points
    # Check if angle is ok
    # If angle is ok, check that almost all other contour points are above the line
    # Or check that all points between are close to the line
    # If so, it is accepted, select the longest stretch
    best_length = 0
    for point_idx in range(2, len(contour)):
        previous_points = contour[:point_idx]
        angles_to_previous_points = _get_angle_of_lines_to_point(contour[point_idx], previous_points)

        for acute_angle_idx in np.nonzero(angles_to_previous_points <= 45)[0]:

            intermediate_points = contour[acute_angle_idx + 1 : point_idx]
            distance_to_intermediate_points = _distance_line_to_points(
                contour[point_idx], contour[acute_angle_idx], intermediate_points
            )
            if np.all(distance_to_intermediate_points <= 8):
                distance = np.linalg.norm(contour[point_idx] - contour[acute_angle_idx])
                if best_length < distance:
                    best_length = distance
                    best_i = point_idx
                    best_j = acute_angle_idx

    mid_point = int(best_j + round((best_i - best_j) / 2))
    # Apex is longest from midpoint
    mid_line_length = 0
    apex = 0
    for i in range(len(contour)):
        length = np.linalg.norm(contour[mid_point] - contour[i])
        if mid_line_length < length:
            mid_line_length = length
            apex = i

    direction = contour[apex] - contour[mid_point]
    normal_direction = np.array([-direction[1], direction[0]])
    normal_direction = normal_direction / np.linalg.norm(normal_direction)  # Normalize
    diameters = []
    for fraction in np.linspace(0, 1, 20, endpoint=False):
        point_on_mid_line = contour[mid_point] + direction * fraction

        distance1 = _find_distance_to_edge(segmentation, point_on_mid_line, normal_direction)
        distance2 = _find_distance_to_edge(segmentation, point_on_mid_line, -normal_direction)
        diameters.append((distance1 + distance2) * isotropic_spacing)

    step_size = (mid_line_length * isotropic_spacing) / 20
    return np.array(diameters), step_size
