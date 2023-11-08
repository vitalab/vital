def main():
    """Run the test script."""
    import argparse
    from math import ceil
    from unittest.mock import patch

    import numpy as np
    from skimage.morphology import disk

    from vital.utils.image.us.measure import EchoMeasure

    parser = argparse.ArgumentParser(
        description="Compute the thickness of the band between an inner and an outer circle of known radii."
    )
    parser.add_argument("--in_radius", type=float, default=80, help="Radius of the inner circle")
    parser.add_argument("--out_radius", type=float, default=100, help="Radius of the outer circle")
    parser.add_argument(
        "--num_points", type=int, default=40, help="Number of points to sample in the center of the band"
    )
    parser.add_argument("--voxel_size", type=float, default=0.5, help="Voxel size (in mm)")
    parser.add_argument("--debug_plot", action="store_true", help="Whether to plot the curvature")
    args = parser.parse_args()

    radius_diff = args.out_radius - args.in_radius

    def sample_circle(radius: float, num_points: int, pixel_coords: bool = False) -> np.ndarray:
        """Sample points along the circumference of a circle of known radius."""
        # Generate points in clockwise order, because that's what the function for computing the curvature expects
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        coords = np.vstack([y, x]).T
        if pixel_coords:
            # Convert from exact floating-point coordinates to integer coordinates
            # + offset from origin so that the circle falls between the inner and outer circles in the positive quadrant
            # + additional offset of 1 to account for the padding of the mask
            coords = coords.astype(int) + ceil(radius) + (radius_diff // 2) + 1
        return coords

    center_samples = sample_circle(
        (args.out_radius + args.in_radius) // 2, args.num_points, pixel_coords=args.debug_plot
    )
    # Create masks of the inner/outer circles
    # + pad the inner circle by the difference in radii so that its center matches the outer circle's
    outer_mask = disk(args.out_radius)
    inner_mask = np.pad(disk(args.in_radius), ((radius_diff, radius_diff), (radius_diff, radius_diff)), mode="constant")

    # Combine the inner and outer and assign them the appropriate labels
    mask = outer_mask * 2
    mask[inner_mask.astype(bool)] = 1
    mask = np.pad(mask, ((1, 1), (1, 1)), mode="constant")

    # Mock the measure object to return a known set of control points that the domain-specific code could not extract
    # from the circle
    with patch.object(EchoMeasure, "control_points", return_value=center_samples):
        # Compute the curvature of the circle
        thickness = EchoMeasure.myo_thickness(
            mask, 1, 2, num_control_points=60, voxelspacing=args.voxel_size, debug_plots=args.debug_plot
        )
        print(f"Computed thickness (in cm): {thickness}")


if __name__ == "__main__":
    main()
