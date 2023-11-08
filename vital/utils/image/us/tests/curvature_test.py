def main():
    """Run the test script."""
    import argparse
    from math import ceil
    from unittest.mock import patch

    import numpy as np

    from vital.utils.image.us.measure import EchoMeasure

    parser = argparse.ArgumentParser(description="Compute the curvature of a circle of known radius.")
    parser.add_argument("--radius", type=float, default=50, help="Radius of the circle")
    parser.add_argument("--num_points", type=int, default=40, help="Number of points to sample along the circle")
    parser.add_argument("--voxel_size", type=float, default=0.2, help="Voxel size (in mm)")
    parser.add_argument("--debug_plot", action="store_true", help="Whether to plot the curvature")
    args = parser.parse_args()

    def sample_circle(radius: float, num_points: int, pixel_coords: bool = False) -> np.ndarray:
        """Sample points along the circumference of a circle of known radius."""
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        coords = np.vstack([y, x]).T
        if pixel_coords:
            # Convert from exact floating-point coordinates to integer coordinates
            # + offset from origin so that the circle falls in the positive quadrant
            coords = coords.astype(int) + ceil(radius)
        return coords

    circle_samples = sample_circle(args.radius, args.num_points, pixel_coords=args.debug_plot)
    mask = np.zeros(np.ceil(circle_samples.max(axis=0)).astype(int) + 1)

    # Mock the measure object to return a known set of control points that the domain-specific code could not extract
    # from the circle
    with patch.object(EchoMeasure, "control_points", return_value=circle_samples):
        # Compute the curvature of the circle
        curvature = EchoMeasure.curvature(
            mask, None, None, None, num_control_points=60, voxelspacing=args.voxel_size, debug_plots=args.debug_plot
        )
        print(f"Computed curvature (in dm^-1): {curvature}")


if __name__ == "__main__":
    main()
