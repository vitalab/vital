import functools
import itertools
from typing import Callable, List, Literal, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import distance
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
    def _split_along_endo_axis(
        segmentation: np.ndarray,
        lv_labels: SemanticStructureId,
        myo_labels: SemanticStructureId,
        voxelspacing: Tuple[float, float] = (1, 1),
    ) -> np.ndarray:
        """Computes a mask that splits the image along a line between the endocardium's apex and middle of the base.

        Args:
            segmentation: (H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the left ventricle.
            voxelspacing: Size of the segmentation's voxels along each (height, width) dimension (in mm).

        Returns:
            Mask that splits the image along a line between the endocardium's apex and middle of the base.
        """
        # Identify major landmarks of the left ventricle (i.e. base's corners + midpoint and apex)
        left_corner, apex, right_corner = EchoMeasure._endo_epi_control_points(
            segmentation, lv_labels, myo_labels, "endo", 3, voxelspacing
        )
        base_mid = (left_corner + right_corner) / 2

        # Compute x = ay + b form of the equation of the LV center line
        a = (apex[1] - base_mid[1]) / (apex[0] - base_mid[0])
        b = -(a * base_mid[0] - base_mid[1])

        def _is_left_of_lv_center_line(y: int, x: int) -> bool:
            """Whether the y,x coordinates provided fall left of the LV center line."""
            return x < a * y + b

        # Create a binary mask for the whole image that is positive above the LV base's line
        left_of_lv_center_line_mask = np.fromfunction(_is_left_of_lv_center_line, segmentation.shape)
        return left_of_lv_center_line_mask

    @staticmethod
    def _endo_epi_control_points(
        segmentation: T,
        lv_labels: SemanticStructureId,
        myo_labels: SemanticStructureId,
        structure: Literal["endo", "epi", "myo"],
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
            (`num_control_points`, 2) Coordinates of the control points along the contour of the endo/epicardium.
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
    @batch_function(item_ndim=2)
    def structure_area_split_by_endo_center_line(
        segmentation: T,
        lv_labels: SemanticStructureId,
        myo_labels: SemanticStructureId,
        half: Literal["left", "right"],
        labels: SemanticStructureId = None,
        voxelspacing: Tuple[float, float] = (1, 1),
    ) -> T:
        """Computes the area of a structure that falls on the left/right side of the endo center line.

        Args:
            segmentation: ([N], H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the left ventricle.
            half: The side of the image to consider when computing the area of the structure. Either "left" or "right".
            labels: Labels of the classes that are part of the structure for which to count the number of pixels. If
                `None`, all truthy values will be considered part of the structure.
            voxelspacing: Size of the segmentation's voxels along each (height, width) dimension (in mm).

        Returns:
            ([N]), Surface associated with the structure (in mm² if `voxelspacing` and pixels otherwise) that falls on
            the left/right side of the endo center line, in each segmentation of the batch.
        """
        # Find the binary mask of the structure
        if labels:
            mask = np.isin(segmentation, labels)
        else:
            mask = segmentation.astype(bool)

        # Find the mask of the left/right split along the endo center line
        half_mask = EchoMeasure._split_along_endo_axis(segmentation, lv_labels, myo_labels, voxelspacing)
        if half == "right":
            half_mask = ~half_mask

        # Only keep the part of the structure that falls on the requested side of the endo center line
        mask = mask * half_mask

        return mask.sum((-2, -1)) * (voxelspacing[0] * voxelspacing[1])

    @staticmethod
    @auto_cast_data
    @batch_function(item_ndim=2)
    def control_points(
        segmentation: T,
        lv_labels: SemanticStructureId,
        myo_labels: SemanticStructureId,
        structure: Literal["endo", "epi", "myo"],
        num_control_points: int,
        voxelspacing: Tuple[float, float] = (1, 1),
    ) -> T:
        """Lists uniformly distributed control points along the contour of the endo/epi or in the center of the myo.

        Args:
            segmentation: ([N], H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the myocardium.
            structure: Structure for which to identify the control points.
            num_control_points: Number of control points to sample. The number of control points should be odd to be
                divisible evenly between the base -> apex and apex -> base segments.
            voxelspacing: Size of the segmentation's voxels along each (height, width) dimension (in mm).

        Returns:
            ([N], `num_control_points`, 2) Coordinates of the control points.
        """
        match structure:
            case "endo" | "epi":
                control_points = EchoMeasure._endo_epi_control_points(
                    segmentation, lv_labels, myo_labels, structure, num_control_points, voxelspacing
                )
            case "myo":
                # Define myocardium control points as the average of corresponding points along the endo/epi contours
                endo_control_points, epicontrol_points = [
                    EchoMeasure._endo_epi_control_points(
                        segmentation, lv_labels, myo_labels, struct, num_control_points, voxelspacing=voxelspacing
                    )
                    for struct in ("endo", "epi")
                ]
                control_points = (endo_control_points + epicontrol_points) // 2

        return control_points

    @staticmethod
    @auto_cast_data
    def longitudinal_strain(
        segmentation: T,
        lv_labels: SemanticStructureId,
        myo_labels: SemanticStructureId,
        num_control_points: int = 31,
        control_points_slice: slice = None,
        correct_drift: bool = True,
        voxelspacing: Tuple[float, float] = (1, 1),
    ) -> T:
        """Global Longitudinal Strain (GLS) for each frame in the sequence, compared to the first frame.

        Args:
            segmentation: (N, H, W), Segmentation map for a whole sequence, where the first frame is assumed to be an
                ED instant.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the myocardium.
            num_control_points: Number of control points to sample along the contour of the endocardium. The number of
                control points should be odd to be divisible evenly between the base -> apex and apex -> base segments.
            control_points_slice: Slice of control points to consider when computing the strain. This is useful to
                compute the strain over a subset of the control points, e.g. over the basal septum in A4C.
            correct_drift: Whether to correct the strain line to ensure that it returns to the baseline, i.e. the
                value of the first frame, at the end of the cycle, i.e. the last frame.
            voxelspacing: Size of the segmentation's voxels along each (height, width) dimension (in mm).

        Returns:
            (N,), Global Longitudinal Strain (GLS) curve, where the value (in percentage) is the relative difference
            in length of the endocardium compared to the first frame.
        """
        voxelspacing = np.array(voxelspacing)

        def _lv_longitudinal_length(frame: np.ndarray) -> float:
            # Find sample points along the contour of the LV
            control_points = EchoMeasure.control_points(
                frame, lv_labels, myo_labels, "endo", num_control_points, voxelspacing=voxelspacing
            )

            # Only keep the control points that are part of the requested segment, if any
            if control_points_slice:
                control_points = control_points[control_points_slice]

            # Compute the perimeter as the sum of distances between each point along the contour and the previous one
            return sum(np.linalg.norm((p1 - p0) * voxelspacing) for p0, p1 in itertools.pairwise(control_points))

        # Compute the longitudinal length of the LV for each frame in the sequence
        lv_longitudinal_lengths = np.array([_lv_longitudinal_length(frame) for frame in segmentation])

        if correct_drift:
            # Measure the drift between the last and first frame
            drift = lv_longitudinal_lengths[-1] - lv_longitudinal_lengths[0]

            # Linearly interpolate between 0 and drift to determine the amount by which to correct in each frame
            drift_correction = np.linspace(0, drift, num=len(segmentation))

            # Correct the drift in the measure longitudinal lengths
            lv_longitudinal_lengths -= drift_correction

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
            ([N]), Distance between the left and right markers at the base of the left ventricle (in cm).
        """
        voxelspacing = np.array(voxelspacing)

        # Identify the base of the left ventricle
        left_corner, right_corner = EchoMeasure._endo_base(segmentation, lv_labels, myo_labels)

        # Compute the distance between the points at the base
        width = np.linalg.norm((left_corner - right_corner) * voxelspacing)
        width *= 1e-1  # Convert from mm to cm
        return width

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
            ([N]), Length of the left ventricle (in cm).
        """
        voxelspacing = np.array(voxelspacing)

        # Identify major landmarks of the left ventricle (i.e. base corners, base's midpoint and apex)
        left_corner, right_corner = EchoMeasure._endo_base(segmentation, lv_labels, myo_labels)
        base_mid = (left_corner + right_corner) / 2
        apex = EchoMeasure._extract_landmarks_from_polar_contour(
            segmentation, lv_labels, polar_smoothing_factor=5e-2, base=False
        )[0]

        # Compute the distance between the apex and the base's midpoint
        length = np.linalg.norm((apex - base_mid) * voxelspacing)
        length *= 1e-1  # Convert from mm to cm
        return length

    @staticmethod
    @auto_cast_data
    @batch_function(item_ndim=2)
    def myo_thickness(
        segmentation: T,
        lv_labels: SemanticStructureId,
        myo_labels: SemanticStructureId,
        num_control_points: int = 31,
        control_points_slice: slice = None,
        voxelspacing: Tuple[float, float] = (1, 1),
        debug_plots: bool = False,
    ) -> T:
        """Measures the average endo-epi distance orthogonal to the centerline over a given number of control points.

        Args:
            segmentation: ([N], H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the myocardium.
            num_control_points: Number of control points to sample along the contour of the endocardium/epicardium. The
                number of control points should be odd to be divisible evenly between the base -> apex and apex -> base
                segments.
            control_points_slice: Slice of control points to consider when computing the curvature. This is useful to
                compute the thickness over a subset of the control points, e.g. over the basal septum in A4C.
            voxelspacing: Size of the segmentation's voxels along each (height, width) dimension (in mm).
            debug_plots: Whether to plot the thickness at each centerline point. This is done by plotting the
                value of the thic at each control point as a color-coded scatter plot on top of the segmentation.
                This should only be used for debugging the expected value of the curvature.

        Returns:
            ([N,] `control_points_slice`), Thickness of the myocardium over the requested segmentm (in cm).
        """
        # Extract control points along the myocardium's centerline
        centerline_pix = EchoMeasure.control_points(
            segmentation, lv_labels, myo_labels, "myo", num_control_points, voxelspacing=voxelspacing
        )
        voxelspacing = np.array(voxelspacing)

        # Compute the direction of the centerline at each control point as the vector between the previous and next
        # points. However, for the first and last points, we simply use the vector between the point itself and its next
        # or previous point, respectively (in practice, this is implemented by edge-padding the centerline points)
        centerline_pad = np.pad(centerline_pix, ((1, 1), (0, 0)), mode="edge")
        centerline_vecs = centerline_pad[2:] - centerline_pad[:-2]  # (num_control_points, 2)

        # Compute the vectors orthogonal to the centerline at each control point, defined as the cross product between
        # the centerline's direction and an arbitrary vector out of the plane of the image (i.e. the z-axis)
        r90_matrix = np.array([[0, -1], [1, 0]])
        orth_vecs = centerline_vecs @ r90_matrix  # (num_control_points, 2)
        # Normalize the orthogonal vectors to be 1mm in length
        unit_orth_vecs = (orth_vecs * voxelspacing) / np.linalg.norm((orth_vecs * voxelspacing), axis=1, keepdims=True)

        # Sample points every mm (up to 1.2cm) along the orthogonal vectors towards both the endo and epi contours
        centerline = centerline_pix * voxelspacing  # (num_control_points, 2)
        sample_offsets = unit_orth_vecs[:, None, :] * np.arange(1, 13)[None, :, None]  # (num_control_points, 12, 2)
        centerline_orth_segments = np.concatenate(
            [
                centerline[:, None] - sample_offsets,  # (num_control_points, 12, 2)
                centerline[:, None],  # (num_control_points, 1, 2)
                centerline[:, None] + sample_offsets,  # (num_control_points, 12, 2)
            ],
            axis=1,
        )  # (num_control_points, 25, 2)

        # Extract the endo and epi contours, in both pixel and physical coordinates
        endo_contour_pix, epi_contour_pix = [
            find_contours(np.isin(segmentation, struct_labels), level=0.9)[0]
            for struct_labels in [lv_labels, [lv_labels, myo_labels]]
        ]
        endo_contour, epi_contour = endo_contour_pix * voxelspacing, epi_contour_pix * voxelspacing

        # For each centerline point, find the nearest point on the endo and epi contours to the points sampled along the
        # orthogonal segment
        endo_closest_orth, epi_closest_orth = [], []
        for orth_segment in centerline_orth_segments:
            endo_dist = distance.cdist(orth_segment, endo_contour)  # (25, `len(endo_contour)`)
            closest_endo_idx = np.unravel_index(np.argmin(endo_dist), endo_dist.shape)[1]
            endo_closest_orth.append(endo_contour_pix[closest_endo_idx])

            epi_dist = distance.cdist(orth_segment, epi_contour)  # (25, `len(epi_contour)`)
            closest_epi_idx = np.unravel_index(np.argmin(epi_dist), epi_dist.shape)[1]
            epi_closest_orth.append(epi_contour_pix[closest_epi_idx])

        endo_closest_orth, epi_closest_orth = np.array(endo_closest_orth), np.array(epi_closest_orth)

        # Compute the thickness as the distance between the points on the endo and epi contours that are closest to the
        # points sampled along the orthogonal vector
        thickness = np.linalg.norm((endo_closest_orth - epi_closest_orth) * voxelspacing, axis=1)
        thickness *= 1e-1  # Convert from mm to cm

        # Only keep the control points that are part of the requested segment, if any
        if control_points_slice:
            thickness = thickness[control_points_slice]

        if debug_plots:
            from matplotlib import pyplot as plt

            if control_points_slice:
                centerline_pix = centerline_pix[control_points_slice]
                endo_closest_orth = endo_closest_orth[control_points_slice]
                epi_closest_orth = epi_closest_orth[control_points_slice]

            plt.imshow(segmentation)
            for points in [centerline_pix, endo_closest_orth, epi_closest_orth]:
                plt.scatter(points[:, 1], points[:, 0], c=thickness, cmap="magma", marker="o", s=3)

            # Annotate the centerline points with their respective thickness value
            for c_coord, c_thickness in zip(centerline_pix, thickness):
                plt.annotate(
                    f"{c_thickness:.1f}",
                    (c_coord[1], c_coord[0]),
                    xytext=(2, 0),
                    textcoords="offset points",
                    fontsize="small",
                )

            plt.show()

        # Average the metric over the control points
        return thickness

    @staticmethod
    @auto_cast_data
    @batch_function(item_ndim=2)
    def curvature(
        segmentation: T,
        lv_labels: SemanticStructureId,
        myo_labels: SemanticStructureId,
        structure: Literal["endo", "epi"],
        num_control_points: int = 31,
        control_points_slice: slice = None,
        voxelspacing: Tuple[float, float] = (1, 1),
        debug_plots: bool = False,
    ) -> T:
        """Measures the average curvature of the endocardium/epicardium over a given number of control points.

        References:
            - Uses the specific definition of curvature proposed by Marciniak et al. (2021) in
              https://doi.org/10.1097/HJH.0000000000002813

        Args:
            segmentation: ([N], H, W), Segmentation map.
            lv_labels: Labels of the classes that are part of the left ventricle.
            myo_labels: Labels of the classes that are part of the myocardium.
            num_control_points: Number of control points to sample along the contour of the structure. The number of
                control points should be odd to be divisible evenly between the base -> apex and apex -> base segments.
            control_points_slice: Slice of control points to consider when computing the curvature. This is useful to
                compute the curvature over a subset of the control points, e.g. over the basal septum in A4C.
            voxelspacing: Size of the segmentation's voxels along each (height, width) dimension (in mm).
            debug_plots: Whether to plot the value of the curvature at each control point. This is done by plotting the
                value of the curvature at each control point as a color-coded scatter plot on top of the segmentation.
                This should only be used for debugging the expected value of the curvature.

        Returns:
            ([N,] `control_points_slice`), Curvature of the endocardium over the requested segment (in dm^-1).
        """
        # Extract control points along the structure's contour
        control_points_pix = EchoMeasure.control_points(
            segmentation, lv_labels, myo_labels, structure, num_control_points, voxelspacing=voxelspacing
        )  # (num_control_points, 2)
        # Convert pixel coordinates to physical coordinates
        control_points = control_points_pix * np.array(voxelspacing)

        # Re-organize the control points into arrays of x and y coordinates
        y_coords, x_coords = control_points.T

        # Compute the curvature at each control point
        # 1. Compute the first and second derivatives of the x and y coordinates
        dx, dy = np.gradient(x_coords, edge_order=2), np.gradient(y_coords, edge_order=2)
        dx2, dy2 = np.gradient(dx), np.gradient(dy)

        # 2. Compute the curvature using Eq. 1 from the paper by Marciniak et al.
        k = (dx2 * dy - dx * dy2) / ((dx**2 + dy**2) ** (3 / 2))
        k *= 1e2  # Convert from mm^-1 to dm^-1

        # Only keep the control points that are part of the requested segment, if any
        if control_points_slice:
            k = k[control_points_slice]

        if debug_plots:
            import matplotlib.colors as colors
            from matplotlib import pyplot as plt

            selected_c_points = control_points_pix
            if control_points_slice:
                selected_c_points = selected_c_points[control_points_slice]

            plt.imshow(segmentation)
            plt.scatter(
                selected_c_points[:, 1],
                selected_c_points[:, 0],
                c=k,
                cmap="seismic",
                norm=colors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10),
                marker="o",
                s=3,
            )
            # Annotate the control points with their respective curvature value
            for c_coord, c_k in zip(selected_c_points, k):
                plt.annotate(
                    f"{c_k:.1f}", (c_coord[1], c_coord[0]), xytext=(2, 0), textcoords="offset points", fontsize="small"
                )

            plt.show()

        return k
