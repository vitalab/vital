import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from pathos.multiprocessing import Pool
from PIL import Image
from scipy.interpolate import griddata
from tqdm import tqdm

from vital.utils.image.coord import pol2cart
from vital.utils.logging import configure_logging


@dataclass
class _BMode:
    """Data structure that represents a B-mode ultrasound image along with the pixels' coordinate system."""

    # Physical constants
    c = 1540
    fc = int(2.5e6)

    # Standard imaging values
    data_min = 0
    data_max = 255

    @dataclass
    class Grid:
        """Data structure that represents a coordinate system for pixels."""

        z: np.ndarray  # z component of each grid pixel's cartesian coordinate
        x: np.ndarray  # x component of each grid pixel's cartesian coordinate

    data: np.ndarray  # Grayscale US data
    grid: Grid  # Cartesian coordinates of the pixels in `data`

    @property
    def lambda_(self) -> float:
        """Returns the lambda for the image, given its acquisition parameters."""
        return self.c / self.fc

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the shape of the underlying US image."""
        return self.data.shape

    @classmethod
    def compute_grid(cls, **kwargs) -> Grid:
        """Computes the cartesian coordinates of the pixels, depending on the image's format and parameters.

        Args:
            **kwargs: Keyword arguments depending on the image's format.

        Returns:
            Cartesian coordinates of the image's pixels.
        """
        raise NotImplementedError

    def _to_coordinates(self, grid: "_BMode.Grid", progress_bar: bool = False) -> np.ndarray:
        """Interpolates `self.data` to be expressed in the `grid` coordinates rather than `self.grid`.

        Args:
            grid: New coordinate grid to use.
            progress_bar: If ``True``, enables progress bars detailing the progress of the conversion of each frame in
                the sequence.

        Returns:
            `self.data` expressed in the `grid` coordinates.
        """
        # If the image is 2D, add a new axis to match the general case of a sequence of frames
        data = self.data
        if is_data_2d := data.ndim == 2:
            data = data[:, ...]

        # Pre-format the large array of source points to avoid redoing it on each thread
        src_points = np.stack([self.grid.z.ravel(), self.grid.x.ravel()], axis=-1)

        def griddata_wrapper(frame_data: np.ndarray) -> np.ndarray:
            # When interpolating, use `fill_value=self.data_min` to fill outside the sector with black
            return griddata(src_points, frame_data.ravel(), (grid.z, grid.x), fill_value=self.data_min)

        # Interpolate frames in parallel because processing each frame can take up to 3.5 seconds
        with Pool() as pool:
            interpolation_jobs = pool.imap(griddata_wrapper, data)
            if progress_bar:
                interpolation_jobs = tqdm(
                    interpolation_jobs,
                    desc="Converting sequence of ultrasound images in polar coordinates to cartesian coordinates",
                    total=len(data),
                    unit="frame",
                    leave=False,
                )
            interpolated_data = np.array([interpolated_frame for interpolated_frame in interpolation_jobs])

        # If the source data was 2D, make get rid of the frame dimension
        if is_data_2d:
            interpolated_data = interpolated_data[0]

        # Clip interpolated values to get rid of possible (slight) artefacts created by the interpolation
        interpolated_data = np.clip(interpolated_data, self.data_min, self.data_max)
        return interpolated_data


@dataclass
class PolarBMode(_BMode):
    """Data structure that represents a B-mode ultrasound image using a polar coordinate system."""

    def __init__(self, data: np.ndarray, depth_start: float, depth_end: float, width: float, tilt: float):
        """Initializes class instance.

        Args:
            data: ([time,]rho,theta), Grayscale US data.
            depth_start: Depth, relative to the US probe, at which the image begins (in meters).
            depth_end: Depth, relative to the US probe, at which the image ends (in meters).
            width: Width of the US sector (in radians).
            tilt: Angle between the center of the acquired sector and the probe's orientation (in degrees).
        """
        super().__init__(data=data, grid=self.compute_grid(data.shape[-2:], depth_start, depth_end, width, tilt))

    @classmethod
    def compute_grid(
        cls, zx_shape: Tuple[int, int], depth_start: float, depth_end: float, width: float, tilt: float
    ) -> _BMode.Grid:
        """Computes the cartesian coordinates of the polar image's pixels.

        Args:
            zx_shape: Z and X dimensions of the polar image (in pixels).
            depth_start: Depth, relative to the US probe, at which the image begins (in meters).
            depth_end: Depth, relative to the US probe, at which the image ends (in meters).
            width: Width of the US sector (in radians).
            tilt: Angle between the center of the acquired sector and the probe's orientation (in degrees).

        Returns:
            Cartesian coordinates of the pixels in the polar image.
        """
        # Compute the step size in both dimensions
        dtheta = width / zx_shape[1]  # polar angular step
        drho = (depth_end - depth_start) / zx_shape[0]  # radial step

        # Compute the 1-D arrays representing the coordinates of the grid in each dimension
        theta = (-(width - dtheta) * np.linspace(-0.5, 0.5, zx_shape[1])) - tilt
        rho = np.linspace(depth_start + drho / 2, depth_end - drho / 2, zx_shape[0])

        # Build the grid of polar coordinates from both 1-D arrays of coordinates
        theta_grid, rho_grid = np.meshgrid(theta, rho)

        # Convert the polar coordinates to cartesian coordinates
        grid = cls.Grid(*pol2cart(theta_grid, rho_grid))

        return grid

    @classmethod
    def from_ge_hdf5(cls, h5_file: h5py.File) -> "PolarBMode":
        """Builds an instance of a polar B-mode image from a HDF5 file following GE's standard, proprietary format.

        Args:
            h5_file: HDF5 file following GE's standard, proprietary format.

        Returns:
            Instance of a polar B-mode image, corresponding to the content of the HDF5 file using GE's format.
        """
        # Extract B-mode data from HDF5
        bmode_ds = h5_file["Tissue"]

        # Extract grayscale image data, swapping the last two axes so that the overall order is ([time,]z,x)
        data = bmode_ds[()]
        data = np.swapaxes(data, data.ndim - 2, data.ndim - 1)

        # Build polar B-mode image instance from image data and metadata
        return cls(
            data=data,
            depth_start=bmode_ds.attrs["DepthStart"].item(),
            depth_end=bmode_ds.attrs["DepthEnd"].item(),
            width=bmode_ds.attrs["Width"].item(),
            tilt=bmode_ds.attrs["Tilt"].item(),
        )


@dataclass
class CartesianBMode(_BMode):
    """Data structure that represents a B-mode ultrasound image using a cartesian coordinate system."""

    # Image metadata
    voxelspacing: Tuple[float, float]

    def __init__(self, data: np.ndarray, grid: _BMode.Grid, voxelspacing: Tuple[float, float]):
        """Initializes class instance.

        Args:
            data: ([time,]Z,X), Grayscale US data.
            grid: Cartesian coordinates of the pixels in `data`.
            voxelspacing: Size (in mm) of the pixels along each (Z, X) dimension.
        """
        super().__init__(data=data, grid=grid)
        self.voxelspacing = voxelspacing

    @classmethod
    def compute_grid(
        cls, z_bounds: Tuple[float, float], z_res: float, x_bounds: Tuple[float, float], x_res: float
    ) -> _BMode.Grid:
        """Computes the cartesian coordinates of the cartesian image's pixels.

        Args:
            z_bounds: Min and max coordinates, respectively, along the Z dimension in the image.
            z_res: Size of the pixels along the Z dimension (in mm).
            x_bounds: Min and max coordinates, respectively, along the X dimension in the image.
            x_res: Size of the pixels along the X dimension (in mm).

        Returns:
            Cartesian coordinates of the pixels in the cartesian image.
        """
        z = np.arange(*z_bounds, z_res)
        x = np.arange(*x_bounds, x_res)
        x_grid, z_grid = np.meshgrid(x, z)
        return cls.Grid(z_grid, x_grid)

    @classmethod
    def from_polar(cls, polar_bmode: PolarBMode, progress_bar: bool = False) -> "CartesianBMode":
        """Builds an instance of a cartesian B-mode image from the image expressed in polar coordinates.

        Args:
            polar_bmode: B-mode image expressed in polar coordinates.
            progress_bar: If ``True``, enables progress bars detailing the progress of the conversion of each frame in
                the sequence.

        Returns:
            Instance of a cartesian B-mode image, corresponding to the input polar B-mode image.
        """
        # Compute a grid of cartesian coordinates including the polar image and
        # corresponding to a fixed image resolution
        z_bounds = polar_bmode.grid.z.min(), polar_bmode.grid.z.max()
        x_bounds = polar_bmode.grid.x.min(), polar_bmode.grid.x.max()
        res = polar_bmode.lambda_ / 2
        cart_grid = cls.compute_grid(z_bounds, res, x_bounds, res)

        # Convert polar data to cartesian data
        cart_data = polar_bmode._to_coordinates(cart_grid, progress_bar=progress_bar)
        cart_bmode = cls(data=cart_data, grid=cart_grid, voxelspacing=(res * 1e3, res * 1e3))
        return cart_bmode


def main():
    """Run the script."""
    configure_logging(log_to_console=True, console_level=logging.INFO)

    parser = ArgumentParser(
        description="Convert polar GE HDF5 files to cartesian images and display them to interactively test B-mode "
        "formatting tools."
    )
    parser.add_argument(
        "polar_h5_file", nargs="+", type=Path, help="Path to the polar GE HDF5 file to convert to cartesian coordinates"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Root directory where to save the ultrasound in cartesian coordinates as PNG images",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Loop to convert files
    for h5_file in tqdm(
        args.polar_h5_file, desc="Converting polar GE HDF5 files to cartesian coordinates", unit="file"
    ):
        # Read h5 file as polar image and convert it to a cartesian image
        with h5py.File(h5_file) as f:
            cart_img = CartesianBMode.from_polar(PolarBMode.from_ge_hdf5(f), progress_bar=True).data
        cart_img = cart_img.astype(np.uint8)  # Convert to `np.uint8` so that PIL interprets the pixel data correctly

        # Save the resulting image
        if cart_img.ndim > 2:
            # If the input is a sequence of images, save each frame to PNG files inside a directory for the sequence
            sequence_dir = args.output_dir / h5_file.stem
            sequence_dir.mkdir(parents=True, exist_ok=True)
            for img_idx, cart_img_2d in tqdm(
                enumerate(cart_img),
                desc=f"Saving images to '{sequence_dir}'",
                total=len(cart_img),
                unit="frame",
                leave=False,
            ):
                (args.output_dir / h5_file.stem).mkdir(exist_ok=True)
                Image.fromarray(cart_img_2d).save(args.output_dir / h5_file.stem / f"{img_idx}.png")
        else:
            # Otherwise, save the 2D image as PNG file directly in the root output directory
            Image.fromarray(cart_img).save(args.output_dir / f"{h5_file.stem}.png")


if __name__ == "__main__":
    main()
