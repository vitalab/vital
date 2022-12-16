import functools
import itertools
from typing import Callable, List, Literal, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from skimage.measure import find_contours

from vital.data.config import SemanticStructureId
from vital.utils.decorators import auto_cast_data, batch_function
from vital.utils.image.coord import cart2pol
from vital.utils.image.measure import Measure, T


class EchoMeasure(Measure):
    """Implementation of various echocardiography-specific measures on images."""

    @staticmethod
    def _extract_landmarks_from_polar_contour(
        segmentation: np.ndarray,
        labels: SemanticStructureId,
        polar_smoothing_factor: float = 0,
        debug_plots: bool = False,
        apex: bool = True,
        base: bool = True,
    ) -> List[np.ndarray]:
        """Extracts a structure's landmarks that produce characteristic peaks in the polar projection of the contour.

        Args:
            segmentation: (H, W), Segmentation map.
            labels: Labels of the classes that are part of the structure for which to extract landmarks.
            polar_smoothing_factor: Multiplicative factor (for the number of points along the contour), to determine the
                standard deviation of a Gaussian kernel to smooth the projection of the contour points in polar
                coordinates.
            debug_plots: Whether to plot the peaks found in the projection of the contour points in polar coordinates +
                where the selected peaks map back on the segmentation. These plots should only be used for debugging the
                identification of the landmarks.
            apex: Whether to try and extract the apex of the structure.
            base: Whether to try and extract the left and right corners at the base of the structure.

        Returns:
            The coordinates of the structure's apex (if `apex==True`) and left and right corners at the base
            (if `base==True`).
        """
        structure_mask = np.isin(segmentation, labels)

        # Extract all the points on the contour of the structure of interest
        # Use `level=0.9` to force the contour to be closer to the structure of interest than the background
        contour = find_contours(structure_mask, level=0.9)[0]

        # Shift the contour, so it's centered around the center of mass of the structure
        contour_centered = contour - ndimage.center_of_mass(structure_mask)

        # Obtain the projection of the contour in polar coordinates
        theta, rho, sort_indices = cart2pol(contour_centered[:, 1], contour_centered[:, 0], sort_by_theta=True)

        if polar_smoothing_factor:
            # Smooth the signal to avoid finding peaks for small localities
            rho = gaussian_filter1d(rho, len(contour) * polar_smoothing_factor)

        # Detect peaks that correspond to endo/epi base and apex
        peaks, properties = find_peaks(rho, height=0)
        peak_heights = properties["peak_heights"]

        landmarks_polar_indices = []

        if apex:
            # Discard base peaks by only keeping peaks found in the upper half of the mask
            # (by discarding peaks found where theta < 0)
            apex_peaks_mask = theta[peaks] < 0
            apex_peaks = peaks[apex_peaks_mask]
            apex_peak_heights = peak_heights[apex_peaks_mask]

            if not len(apex_peaks):
                raise RuntimeError("Unable to identify the apex of the endo/epi.")

            # Keep only the highest peak as the peak corresponding to the apex
            landmarks_polar_indices.append(apex_peaks[apex_peak_heights.argmax()])

        if base:
            # Discard apex peak by only keeping peaks found in the lower half of the mask
            # (by discarding peaks found where theta > 0)
            base_peaks_mask = theta[peaks] > 0
            base_peaks = peaks[base_peaks_mask]
            base_peak_heights = peak_heights[base_peaks_mask]

            if (num_peaks := len(base_peaks)) < 2:
                raise RuntimeError(
                    f"Identified {num_peaks} corner(s) for the endo/epi base. We needed to find at least 2 corners to "
                    f"identify the corners at the base of the endo/epi."
                )

            # Identify the indices of the 2 highest peaks in the list of peaks
            base_highest_peaks = base_peak_heights.argsort()[-2:]
            # Sort the indices of the 2 highest peaks to make sure they stay ordered by descending theta
            # (so that the peak of the left corner comes first) regardless of their heights
            base_highest_peaks = sorted(base_highest_peaks, reverse=True)

            landmarks_polar_indices.extend(base_peaks[base_highest_peaks])

        if debug_plots:
            # Display contour curve in polar coordinates
            import seaborn as sns
            from matplotlib import pyplot as plt

            with sns.axes_style("darkgrid"):
                plot = sns.lineplot(data=pd.DataFrame({"theta": theta, "rho": rho}), x="theta", y="rho")

            # Annotate the peaks with their respective index
            for peak_idx, peak in enumerate(peaks):
                plot.annotate(f"{peak_idx}", (theta[peak], rho[peak]), xytext=(1, 4), textcoords="offset points")

            # Plot lines pointing to the peaks to make them more visible
            plot.vlines(x=theta[peaks], ymin=rho.min(), ymax=rho[peaks], linestyles="dashed")

            plt.show()

        # Map the indices of the peaks in polar coordinates back to the indices in the list of contour points
        contour_indices = sort_indices[landmarks_polar_indices]
        landmarks = contour[contour_indices]

        if debug_plots:
            plt.imshow(structure_mask)
            for landmark in landmarks:
                plt.scatter(landmark[1], landmark[0], c="r", marker="o", s=3)
            plt.show()

        return landmarks

    @staticmethod
    def _endo_epi_contour(
        segmentation: np.ndarray,
        labels: SemanticStructureId,
        base_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Lists points on the contour of the endo/epi (excluding the base), from the left of the base to its right.

        Args:
            segmentation: (H, W), Segmentation map.
            labels: Labels of the classes that are part of the endocardium/epicardium.
            base_fn: Function that identifies the left and right corners at the base of the endocardium/epicardium in a
                segmentation mask.

        Returns:
            Coordinates of points on the contour of the endo/epi (excluding the base), from the left of the base to its
            right.
        """
        structure_mask = np.isin(segmentation, labels)

        # Identify the left/right markers at the base of the endo/epi
        left_corner, right_corner = base_fn(segmentation)

        # Extract all the points on the contour of the structure of interest
        # Use `level=0.9` to force the contour to be closer to the structure of interest than the background
        contour = find_contours(structure_mask, level=0.9)[0]

        # Shift the contour so that they start at the left corner
        # To detect the contour coordinates that match the corner, we use the closest match since skimage's
        # `find_contours` coordinates are interpolated between pixels, so they won't match exactly corner coordinates
        dist_to_left_corner = np.linalg.norm(left_corner - contour, axis=1)
        left_corner_contour_idx = np.argmin(dist_to_left_corner)
        contour = np.roll(contour, -left_corner_contour_idx, axis=0)

        # Filter the full contour to discard points along the base
        # We implement this by slicing the contours from the left corner to the right corner, since the contour returned
        # by skimage's `find_contours` is oriented clockwise
        dist_to_right_corner = np.linalg.norm(right_corner - contour, axis=1)
        right_corner_contour_idx = np.argmin(dist_to_right_corner)
        contour_without_base = contour[: right_corner_contour_idx + 1]

        return contour_without_base

    @staticmethod
    def _endo_base(
        segmentation: np.ndarray, lv_labels: SemanticStructureId, myo_labels: SemanticStructureId
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Finds the left/right markers at the base of the endocardium.

        Notes:
            - This implementation exists because it is more reliable for the endo than the more general algorithm that
              tries to identify peaks in the polar projection of a contour. As such, in cases where the latter gives
              acceptable results, it should be preferred.

        Args:
            segmentation: (H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the left ventricle.

        Returns:
            Coordinates of the left/right markers at the base of the endocardium.
        """
        struct = ndimage.generate_binary_structure(2, 2)
        left_ventricle = np.isin(segmentation, lv_labels)
        myocardium = np.isin(segmentation, myo_labels)
        others = ~(left_ventricle + myocardium)
        dilated_myocardium = ndimage.binary_dilation(myocardium, structure=struct)
        dilated_others = ndimage.binary_dilation(others, structure=struct)
        y_coords, x_coords = np.nonzero(left_ventricle * dilated_myocardium * dilated_others)

        if (num_markers := len(y_coords)) < 2:
            raise RuntimeError(
                f"Identified {num_markers} marker(s) at the edges of the left ventricle/myocardium frontier. We need "
                f"to identify at least 2 such markers to determine the base of the left ventricle."
            )

        if np.all(x_coords == x_coords.mean()):
            # Edge case where the base points are aligned vertically
            # Divide frontier into bottom and top halves.
            coord_mask = y_coords > y_coords.mean()
            left_point_idx = y_coords[coord_mask].argmin()
            right_point_idx = y_coords[~coord_mask].argmax()
        else:
            # Normal case where there is a clear divide between left and right markers at the base
            # Divide frontier into left and right halves.
            coord_mask = x_coords < x_coords.mean()
            left_point_idx = y_coords[coord_mask].argmax()
            right_point_idx = y_coords[~coord_mask].argmax()
        return (
            np.array([y_coords[coord_mask][left_point_idx], x_coords[coord_mask][left_point_idx]]),
            np.array([y_coords[~coord_mask][right_point_idx], x_coords[~coord_mask][right_point_idx]]),
        )

    @staticmethod
    @auto_cast_data
    @batch_function(item_ndim=2)
    def endo_epi_control_points(
        segmentation: T,
        lv_labels: SemanticStructureId,
        myo_labels: SemanticStructureId,
        structure: Literal["endo", "epi"],
        num_control_points: int,
        voxelspacing: Tuple[float, float] = (1, 1),
    ) -> T:
        """Lists uniformly distributed control points along the contour of the endocardium/epicardium.

        Args:
            segmentation: (H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the myocardium.
            structure: Structure for which to identify the control points.
            num_control_points: Number of control points to sample along the contour of the endocardium/epicardium. The
                number of control points should be odd to be divisible evenly between the base -> apex and apex -> base
                segments.
            voxelspacing: Size of the segmentation's voxels along each (height, width) dimension (in mm).

        Returns:
            Coordinates of the control points along the contour of the endocardium/epicardium.
        """
        voxelspacing = np.array(voxelspacing)

        # "Backend" function used to find the corners at the base of the structure depends on the structure
        match structure:
            case "endo":
                struct_labels = lv_labels
                base_fn = functools.partial(EchoMeasure._endo_base, lv_labels=lv_labels, myo_labels=myo_labels)
            case "epi":
                struct_labels = [lv_labels, myo_labels]
                base_fn = functools.partial(
                    EchoMeasure._extract_landmarks_from_polar_contour,
                    labels=struct_labels,
                    polar_smoothing_factor=5e-3,  # 5e-3 was determined empirically
                    apex=False,
                )
            case _:
                raise ValueError(f"Unexpected value for 'mode': {structure}. Use either 'endo' or 'epi'.")

        # Find the points along the contour of the endo/epi excluding the base
        contour = EchoMeasure._endo_epi_contour(segmentation, struct_labels, base_fn)

        # Identify the apex from the points within the contour
        apex = EchoMeasure._extract_landmarks_from_polar_contour(
            segmentation, struct_labels, polar_smoothing_factor=5e-2, base=False  # 5e-2 was determined empirically
        )[0]

        # Round the contour's coordinates, so they don't fall between pixels anymore
        contour = contour.round().astype(int)

        # Break the contour down into independent segments (base -> apex, apex -> base) along which to uniformly
        # distribute control points
        apex_idx_in_contour = np.linalg.norm((contour - apex) * voxelspacing, axis=1).argmin()
        segments = [0, apex_idx_in_contour, len(contour) - 1]

        if (num_control_points - 1) % (num_segments := len(segments) - 1):
            raise ValueError(
                f"The number of requested control points: {num_control_points}, cannot be divided evenly across the "
                f"{num_segments} contour segments. Please set a number of control points that, when subtracted by 1, "
                f"is divisible by {num_segments}."
            )
        num_control_points_per_segment = (num_control_points - 1) // num_segments

        # Simplify the general case for handling th
        control_points_indices = [0]
        for segment_start, segment_stop in itertools.pairwise(segments):
            # Slice segment so that both the start and stop points are included in the segment
            segment = contour[segment_start : segment_stop + 1]

            # Compute the geometric distances between each point along the segment and the previous point.
            # This allows to then simply compute the cumulative distance from the left corner to each segment point
            segment_dist_to_prev = [0.0] + [
                np.linalg.norm((p1 - p0) * voxelspacing) for p0, p1 in itertools.pairwise(segment)
            ]
            segment_cum_dist = np.cumsum(segment_dist_to_prev)

            # Select points along the segment that are equidistant (by selecting points that are closest to where
            # steps of `perimeter / num_control_points` would expect to find points)
            control_points_step = np.linspace(0, segment_cum_dist[-1], num=num_control_points_per_segment + 1)
            segment_control_points = [
                segment_start + np.argmin(np.abs(point_cum_dist - segment_cum_dist))
                for point_cum_dist in control_points_step
            ]
            # Skip the first control point in the current segment, because its already included as the last control
            # point of the previous segment
            control_points_indices += segment_control_points[1:]

        return contour[control_points_indices]

    @staticmethod
    @auto_cast_data
    def gls(
        segmentation: T,
        lv_labels: SemanticStructureId,
        myo_labels: SemanticStructureId,
        voxelspacing: Tuple[float, float] = (1, 1),
    ) -> T:
        """Global Longitudinal Strain (GLS) for each frame in the sequence, compared to the first frame.

        Args:
            segmentation: (N, H, W), Segmentation map for a whole sequence, where the first frame is assumed to be an
                ED instant.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the myocardium.
            voxelspacing: Size of the segmentation's voxels along each (height, width) dimension (in mm).

        Returns:
            (N,), Global Longitudinal Strain (GLS) curve, where the value (in percentage) is the relative difference
            in length of the endocardium compared to the first frame.
        """
        voxelspacing = np.array(voxelspacing)

        def _lv_longitudinal_length(frame: np.ndarray) -> float:
            # Find the points along the contour of the LV excluding the base
            contour = EchoMeasure._endo_epi_contour(
                frame, lv_labels, functools.partial(EchoMeasure._endo_base, lv_labels=lv_labels, myo_labels=myo_labels)
            )

            # Compute the perimeter as the sum of distances between each point along the contour and the previous one
            return sum(np.linalg.norm((p1 - p0) * voxelspacing) for p0, p1 in itertools.pairwise(contour))

        # Compute the longitudinal length of the LV for each frame in the sequence
        lv_longitudinal_lengths = np.array([_lv_longitudinal_length(frame) for frame in segmentation])

        # Compute the GLS for each frame in the sequence
        ed_lv_longitudinal_length = lv_longitudinal_lengths[0]
        gls = ((lv_longitudinal_lengths - ed_lv_longitudinal_length) / ed_lv_longitudinal_length) * 100

        return gls

    @staticmethod
    @auto_cast_data
    @batch_function(item_ndim=2)
    def lv_base_width(
        segmentation: T,
        lv_labels: SemanticStructureId,
        myo_labels: SemanticStructureId,
        voxelspacing: Tuple[float, float] = (1, 1),
    ) -> T:
        """Measures the distance between the left and right markers at the base of the left ventricle.

        Args:
            segmentation: ([N], H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the myocardium.
            voxelspacing: Size of the segmentation's voxels along each (height, width) dimension (in mm).

        Returns:
            ([N]), Distance between the left and right markers at the base of the left ventricle, or NaNs for the
            images where those 2 points cannot be reliably estimated.
        """
        voxelspacing = np.array(voxelspacing)

        # Identify the base of the left ventricle
        left_corner, right_corner = EchoMeasure._endo_base(segmentation, lv_labels, myo_labels)

        # Compute the distance between the points at the base
        return np.linalg.norm((left_corner - right_corner) * voxelspacing)

    @staticmethod
    @auto_cast_data
    @batch_function(item_ndim=2)
    def lv_length(
        segmentation: T,
        lv_labels: SemanticStructureId,
        myo_labels: SemanticStructureId,
        voxelspacing: Tuple[float, float] = (1, 1),
    ) -> T:
        """Measures the LV length as the distance between the base's midpoint and the apex.

        Args:
            segmentation: ([N], H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the myocardium.
            voxelspacing: Size of the segmentation's voxels along each (height, width) dimension (in mm).

        Returns:
            ([N]), Length of the left ventricle.
        """
        voxelspacing = np.array(voxelspacing)

        # Identify major landmarks of the left ventricle (i.e. base corners, base's midpoint and apex)
        left_corner, right_corner = EchoMeasure._endo_base(segmentation, lv_labels, myo_labels)
        base_mid = (left_corner + right_corner) / 2
        apex = EchoMeasure._extract_landmarks_from_polar_contour(
            segmentation, lv_labels, polar_smoothing_factor=5e-2, base=False
        )[0]

        # Compute the distance between the apex and the base's midpoint
        return np.linalg.norm((apex - base_mid) * voxelspacing)
