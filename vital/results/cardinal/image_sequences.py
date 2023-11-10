from argparse import ArgumentParser
from typing import Any, Dict, Literal, Sequence, Tuple

from vital.data.cardinal.config import IMG_FORMAT
from vital.data.cardinal.utils.data_struct import View
from vital.data.cardinal.utils.itertools import Views
from vital.results.processor import ResultsProcessor
from vital.utils.parsing import StoreDictKeyPair, dtype_tuple
from vital.utils.path import as_file_extension


class ImageSequences(ResultsProcessor):
    """Class that saves view`s image ana mask to image format."""

    desc = "image_sequences"
    ResultsCollection = Views

    def __init__(
        self,
        subdir_levels: Sequence[Literal["patient", "view"]] = None,
        include_tags: Sequence[str] = None,
        exclude_tags: Sequence[str] = None,
        overlay_pairs_tags: Sequence[Tuple[str, str]] = None,
        overlay_control_points: int = 0,
        target_img_tag: str = None,
        target_img_size: Tuple[int, int] = None,
        target_voxelspacing: Tuple[float, float] = None,
        img_format: str = IMG_FORMAT,
        io_backend_kwargs: Dict[str, Any] = None,
        cache_attrs: bool = True,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            subdir_levels: Levels of subdirectories to create under the root output folder.
            include_tags: Keys in `data` and `attrs` to save to disk. Mutually exclusive parameter with `exclude_tags`.
            exclude_tags: Keys in `data` and `attrs` to exclude from being saved to disk. Mutually exclusive parameter
                with `include_tags`.
            overlay_pairs_tags: Keys of image/mask pairs to save with the mask painted over the image.
            overlay_control_points: Number of control points to mark along the contours of the endocardium and
                epicardium in image/mask overlays.
            target_img_tag: Tag of the reference image to match in size before saving. Mutually exclusive parameter with
                `target_img_size` and `target_voxelspacing`.
            target_img_size: Target height and width at which to resize the image before saving. Mutually exclusive
                parameter with `target_img_tag` and `target_voxelspacing`.
            target_voxelspacing: Target height and width voxel size to obtain by resizing the image before saving.
                Mutually exclusive parameter with `target_img_tag` and `target_img_size`.
            img_format: File extension of the image format to save the data as.
            io_backend_kwargs: Arguments to pass along to the imaging backend (e.g. SimpleITK, Pillow, etc.) used to
                save the data. The backend selected will depend on the requested tags and image format.
            cache_attrs: Whether to also save (a cache of) attributes, to avoid having to compute them again when
                loading the view from disk in the future.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        formatted_img_format = as_file_extension(img_format)[1:].replace(".", "_").lower()
        super().__init__(output_name=f"{self.desc}_{formatted_img_format}", **kwargs)

        resize_kwargs = {}
        if target_img_tag:
            resize_kwargs["target_img_tag"] = target_img_tag
        if target_img_size:
            resize_kwargs["target_img_size"] = target_img_size
        if target_voxelspacing:
            resize_kwargs["target_voxelspacing"] = target_voxelspacing

        self.view_save_kwargs = {
            "subdir_levels": subdir_levels,
            "include_tags": include_tags,
            "exclude_tags": exclude_tags,
            "overlay_pairs_tags": overlay_pairs_tags,
            "overlay_control_points": overlay_control_points,
            "resize_kwargs": resize_kwargs,
            "img_format": img_format,
            "io_backend_kwargs": io_backend_kwargs,
            "cache_attrs": cache_attrs,
        }

    def process_result(self, result: View) -> None:
        """Saves specific image sequences or combinations of them to a target image format.

        Args:
            result: Data structure holding all the sequence`s data.
        """
        result.save(self.output_path, **self.view_save_kwargs)

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser for images processor.

        Returns:
            Parser object for images processor.
        """
        parser = super().build_parser()
        parser.add_argument(
            "--subdir_levels",
            type=str,
            nargs="+",
            choices=["patient", "view"],
            help="Levels of subdirectories to create under the root output folder",
        )
        tags_mutually_exclusive_parser = parser.add_mutually_exclusive_group(required=False)
        tags_mutually_exclusive_parser.add_argument(
            "--include_tags", type=str, nargs="+", help="Tags of images in the view to save to disk"
        )
        tags_mutually_exclusive_parser.add_argument(
            "--exclude_tags", type=str, nargs="+", help="Tags of images in the view to exclude from saving to disk"
        )
        parser.add_argument(
            "--overlay_pairs_tags",
            type=dtype_tuple,
            nargs="+",
            metavar="BMODE_TAG,MASK_TAG",
            help="Tags of image/mask pairs to save to disk, with the mask painted over the image",
        )
        parser.add_argument(
            "--overlay_control_points",
            type=int,
            help="Number of control points to mark along the contours of the endocardium and epicardium in image/mask "
            "overlays",
        )
        resize_mutually_exclusive_parser = parser.add_mutually_exclusive_group(required=False)
        resize_mutually_exclusive_parser.add_argument(
            "--target_img_tag", type=str, help="Tag of the reference image to match in size before saving"
        )
        resize_mutually_exclusive_parser.add_argument(
            "--target_img_size",
            type=int,
            nargs=2,
            help="Target height and width at which to resize the data before saving",
        )
        resize_mutually_exclusive_parser.add_argument(
            "--target_voxelspacing",
            type=float,
            nargs=2,
            help="Target height and width voxel size to obtain by resizing the data before saving",
        )
        parser.add_argument(
            "--img_format", type=str, default=IMG_FORMAT, help="File extension of the image format to save the views as"
        )
        parser.add_argument(
            "--io_backend_kwargs",
            type=StoreDictKeyPair,
            metavar="ARG1=VAL1,ARG2=VAL2...",
            help="Arguments to pass along to the imaging backend (e.g. SimpleITK, Pillow, etc.) used to save the "
            "sequences. The backend selected will depend on the requested tags and image format.",
        )
        parser.add_argument(
            "--no_cache_attrs",
            dest="cache_attrs",
            action="store_false",
            help="Whether to also save (a cache of) image attributes, to avoid having to compute them again when "
            "loading the view from disk in the future",
        )
        return parser


def main():
    """Run the script."""
    ImageSequences.main()


if __name__ == "__main__":
    main()
