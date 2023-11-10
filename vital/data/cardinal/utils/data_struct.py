import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, NamedTuple, Optional, Sequence, Tuple, Union

import matplotlib.colors as mcolors
import numpy as np
import yaml
from PIL import Image
from PIL.Image import Resampling
from scipy import ndimage
from skimage import color
from skimage.morphology import disk

from vital.data.cardinal.config import (
    ATTRS_CACHE_FORMAT,
    ATTRS_FILENAME_PATTERN,
    IMG_FILENAME_PATTERN,
    IMG_FORMAT,
    TABULAR_ATTRS_FORMAT,
    CardinalTag,
    Label,
    TabularAttribute,
    TimeSeriesAttribute,
)
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.utils.attributes import compute_mask_time_series_attributes
from vital.utils.data_struct import LazyDict
from vital.utils.image.io import sitk_load, sitk_save
from vital.utils.image.transform import resize_image, resize_image_to_voxelspacing
from vital.utils.image.us.measure import EchoMeasure
from vital.utils.path import as_file_extension, remove_suffixes

logger = logging.getLogger(__name__)

PIL_SEQUENCE_FORMATS = [".gif"]


def load_attributes(
    patient_id: str,
    data_roots: Sequence[Path],
    view: ViewEnum = None,
    handle_errors: Optional[Literal["warning", "error"]] = None,
) -> Dict[TabularAttribute, Any]:
    """Loads attributes related to a patient or to a specific view from a YAML metadata file.

    Args:
        patient_id: ID of the patient for whom to collect the attributes.
        data_roots: Folders in which to look for the patient's attributes' file. We look for files recursively inside
            the directories, so as long as the file respects the `ATTRS_FILENAME_PATTERN` pattern, it will be collected.
        view: View for which to collect attributes.
        handle_errors: How to handle missing or duplicate attributes' file/entries in the file.

    Returns:
        Attributes of a patient, or a specific view if `view` is specified.
    """
    match handle_errors:
        case "warning":
            err_handler = logger.warning
        case "error":

            def err_handler(msg: str) -> None:
                raise RuntimeError(msg)

        case None:

            def err_handler(msg: str) -> None:  # noqa: F811
                pass

        case _:
            raise ValueError(
                f"Unexpected value for 'handle_errors': {handle_errors}. Use one of: [None, 'warning', 'error']."
            )

    attrs_file_pattern = ATTRS_FILENAME_PATTERN.format(
        patient_id=patient_id, ext=as_file_extension(TABULAR_ATTRS_FORMAT)
    )
    attrs_files = []
    for data_root in data_roots:
        # Search recursively inside the provided directory
        attrs_files.extend(data_root.rglob(attrs_file_pattern))

    if not attrs_files:
        err_handler(f"Patient '{patient_id}' had no attributes' data.")
        return {}

    if len(attrs_files) > 1:
        err_handler(
            f"Found more than one attributes file for patient '{patient_id}' at {attrs_files}. We will use the first "
            f"one in the previous list by default, but you should check why multiple files exist for this patient."
        )
    attrs = yaml.safe_load(attrs_files[0].read_text())
    if attrs is None:
        attrs = {}

    if view:
        # If a view is requested, extract only the metadata specific to the requested view
        attrs = attrs.get(view, {})
        if not attrs:
            err_handler(
                f"Patient '{patient_id}' had no attributes' data for view '{str(view)}' in attributes' file "
                f"'{attrs_files[0]}'."
            )
    else:
        # Discard the metadata specific to each view and only keep the patient-wide metadata
        for view in ViewEnum:
            attrs.pop(view, None)

    # After the extraction of the requested view or deletion of the unwanted views,
    # convert the string keys to their corresponding enum values
    # NOTE: The string keys are escaped to convert them to legal Python variable name +
    # to follow our enum naming convention
    attrs = {TabularAttribute[k.lower().replace("/", "_")]: v for k, v in attrs.items()}

    return attrs


@dataclass
class Patient:
    """Data structure that bundles image data and metadata for one patient."""

    Id = str

    id: Id
    views: Dict[ViewEnum, "View"]
    attrs: Dict[TabularAttribute, Any]

    def get_mask_attributes(self, mask_tag: str) -> Dict[ViewEnum, Dict[str, np.ndarray]]:
        """Returns the attributes values w.r.t. time for a given mask, for each view.

        Args:
            mask_tag: Tag of the mask to get the attributes for.

        Returns:
            Dictionary of attributes and their values for the given mask, for each view.
        """
        return {view_enum: view.get_mask_attributes(mask_tag) for view_enum, view in self.views.items()}

    def get_patient_attributes(self) -> Dict[str, Union[int, float]]:
        """Returns the patient's global attributes.

        Returns:
            Dictionary of attributes and their values.
        """
        return self.attrs

    @classmethod
    def from_dir(
        cls,
        patient_id: str,
        data_roots: Sequence[Path],
        views: Sequence[ViewEnum] = None,
        handle_attrs_errors: Optional[Literal["warning", "error"]] = None,
        **kwargs,
    ) -> "Patient":
        """Loads and merges image data and metadata files, from multiple sources, related to one patient.

        Args:
            patient_id: ID of the patient for whom to collect data.
            data_roots: Folders in which to look for files related to the patient. We look for files recursively inside
                the directories, so as long as the files respect the `IMG_FILENAME_PATTERN` pattern, they will be
                collected.
            views: Specific views for which to collect data, in case not all views available should be collected. If not
                specified, then all available views will be collected by default.
            handle_attrs_errors: How to handle missing or duplicate attributes' file/entries in the file.
            kwargs: Parameters to pass along to each of the views' ``from_dir`` method.

        Returns:
            `Patient` instance, built using data related to the patient from the provided folders.
        """
        from vital.data.cardinal.utils.itertools import views_avail_by_patient

        if not views:
            # If no specific views are requested, consider all possible views
            views = list(ViewEnum)

        # Load image data
        avail_views = views_avail_by_patient(data_roots, patient_id)
        views_data = {}
        for view in views:
            if view not in avail_views:
                logging.warning(f"Patient '{patient_id}' had no data for the requested '{view}' view.")
                continue
            views_data[view] = View.from_dir(patient_id, view, data_roots, **kwargs)

        return cls(
            id=patient_id,
            views=views_data,
            attrs=load_attributes(patient_id, data_roots, handle_errors=handle_attrs_errors),
        )

    def save(
        self,
        save_dir: Path,
        subdir_levels: Sequence[Literal["patient", "view"]] = None,
        save_tabular_attrs: bool = False,
        **kwargs,
    ) -> None:
        """Saves the patient's data and metadata as multiple files on disk.

        Args:
            save_dir: Directory where to save the data.
            subdir_levels: Levels of subdirectories to create under `save_dir`.
            save_tabular_attrs: Whether to also save the patient's tabular attributes in YAML config files.
            kwargs: Parameters to pass along to each of the views' ``save`` method.
        """
        for view in self.views.values():
            view.save(save_dir=save_dir, subdir_levels=subdir_levels, **kwargs)

        if save_tabular_attrs:
            # Collect the tabular attributes, converting enum keys to strings to have a cleaner serialized output
            # + sort the attributes in the order they appear in the config enum
            tab_attrs = {str(attr_key): self.attrs[attr_key] for attr_key in TabularAttribute if attr_key in self.attrs}

            # Determine the path where to save the attributes
            if subdir_levels:
                if subdir_levels[0] == "patient":
                    save_dir /= self.id
                else:
                    raise ValueError(
                        f"You specified the following subfolder levels to save patients' data: {subdir_levels}, "
                        f"meaning the data for one patient would be split across different folders under different "
                        f"roots. This is incompatible with saving a patient's tabular attributes, which expects all "
                        f"the related to one patient to be saved under a single root folder."
                    )
            filename = ATTRS_FILENAME_PATTERN.format(patient_id=self.id, ext=as_file_extension(TABULAR_ATTRS_FORMAT))

            # Save the tabular attributes to disk
            (save_dir / filename).write_text(yaml.dump(tab_attrs, sort_keys=False))


@dataclass
class View:
    """Data structure that bundles image data and metadata for one view from a patient."""

    class Id(NamedTuple):
        """Data structure representing a view ID."""

        patient: Patient.Id
        view: ViewEnum

    id: Id
    data: Dict[str, np.ndarray] = field(default_factory=LazyDict)
    attrs: Dict[str, Dict[str, Any]] = field(default_factory=LazyDict)

    # Add additional private field to store paths of image files, to enable lazy loading
    _data_paths: Dict[str, Path] = field(init=False, repr=False, default_factory=dict)

    def _is_mask_tag(self, data_tag: str) -> bool:
        """Determines whether the provided `data_tag` corresponds to a segmentation mask or not."""
        return CardinalTag.mask in data_tag

    def _init_data_and_attrs(self, data_tag: str, data_file: Path, overwrite_attrs_cache: bool = False) -> None:
        """Stores the path to the image, and sets-up the callbacks for loading the data at a later time.

        Args:
            data_tag: Tag under which to load the image's data, at a later time.
            data_file: Path to the image.
            overwrite_attrs_cache: Whether to discard the current cache of attributes and compute them again. Has no
                effect when no attributes cache exists.
        """
        # Store the path of the data, for when we will need to load it
        self._data_paths[data_tag] = data_file
        # Store callback for loading the data where the data itself should have been stored
        # Because the dictionaries are lazy, these callbacks will be executed when fetching the data for the first time
        # NOTE: The callbacks load all the data related to `data_tag` as a side effect for optimal performance, but they
        # must also return the values to be assigned to the left operand to work with the `LazyDict` API
        self.data[data_tag] = lambda: self._load_data_and_attrs(data_tag, overwrite_attrs_cache=overwrite_attrs_cache)[
            0
        ]
        self.attrs[data_tag] = lambda: self._load_attrs(data_tag, overwrite_attrs_cache=overwrite_attrs_cache)

    def _load_data_and_attrs(
        self, data_tag: str, overwrite_attrs_cache: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Loads image data for `data_tag`, and loads related attributes from cache or computes them from the image.

        Args:
            data_tag: Tag of the image to load.
            overwrite_attrs_cache: Whether to discard the current cache of attributes and compute them again. Has no
                effect when no attributes cache exists.

        Returns:
            A tuple of i) the image and ii) the attributes related to the image.
        """
        im_array, im_metadata = sitk_load(self._data_paths[data_tag])
        voxelspacing = im_metadata["spacing"][:2][::-1]

        # If a cache of attributes already exists (and it should not be overwritten), load the attributes from there to
        # avoid having to compute them over again from the image
        attrs_cache_path = remove_suffixes(self._data_paths[data_tag]).with_suffix(
            as_file_extension(ATTRS_CACHE_FORMAT)
        )
        cached_attrs = None
        if attrs_cache_path.exists() and not overwrite_attrs_cache:
            cached_attrs = np.load(attrs_cache_path)

        self.add_image(data_tag, im_array, voxelspacing=voxelspacing, precomputed_attrs=cached_attrs)

        # If no cache of the attributes exists or it should be overwritten, create it
        if (not attrs_cache_path.exists()) or overwrite_attrs_cache:
            np.savez(attrs_cache_path, **self.attrs[data_tag])

        return self.data[data_tag], self.attrs[data_tag]

    def _load_attrs(self, data_tag: str, overwrite_attrs_cache: bool = False) -> Dict[str, Any]:
        """Ensures `data_tag` attributes are available, either by loading them from cache or computing them.

        Args:
            data_tag: Tag of the image for which to load attributes.
            overwrite_attrs_cache: Whether to discard the current cache of attributes and compute them again. Has no
                effect when no attributes cache exists.

        Returns:
            Attributes related to the `data_tag` image.
        """
        attrs_cache_path = remove_suffixes(self._data_paths[data_tag]).with_suffix(
            as_file_extension(ATTRS_CACHE_FORMAT)
        )
        if attrs_cache_path.exists() and not overwrite_attrs_cache:
            self.attrs[data_tag] = np.load(attrs_cache_path)
        else:
            # If no cache of attributes is available (or it should be overwritten), manually load the data again to
            # compute and cache the attributes. This can cause the image to be read again even if it was already loaded,
            # but since most of the cost of `_load_data_and_attrs` comes from computing the attributes, the cost of
            # reading the image itself is negligible
            _ = self._load_data_and_attrs(data_tag, overwrite_attrs_cache=True)

        return self.attrs[data_tag]

    def add_image(
        self,
        tag: str,
        img: np.ndarray,
        voxelspacing: Tuple[float, float] = None,
        reference_img_tag: str = None,
        precomputed_attrs: Dict[str, Any] = None,
    ) -> None:
        """Adds an image to the view, automatically computing relevant attributes (if not provided).

        Args:
            tag: Tag under which to add the image.
            img: Image array to add.
            voxelspacing: Size of the image's voxels along each (height, width) dimension (in mm). Mutually exclusive
                parameter with `reference_mask_tag`.
            reference_img_tag: Tag of an image to use as a reference for computing the new image's voxelspacing, under
                the assumption that the new image is some resized version of the reference image. Mutually exclusive
                parameter with `voxelspacing`.
            precomputed_attrs: Precomputed values for attributes to use instead of computing them from the image.
        """
        if not (bool(voxelspacing) ^ bool(reference_img_tag)):
            raise ValueError(
                "You must provide exactly one parameter from which to derive the image's voxelspacing: `voxelspacing` "
                "or `reference_img_tag`."
            )
        if reference_img_tag and reference_img_tag not in self.data:
            raise ValueError(f"'{reference_img_tag}' data not found in view '{self.id}'.")

        if reference_img_tag:
            # Compute the voxelspacing of the new image based on the voxelspacing of the reference image
            ref_img = self.data[reference_img_tag]
            ref_voxelspacing = self.attrs[reference_img_tag][CardinalTag.voxelspacing]
            # For each dimension, the voxelspacing equals to `shape * voxelspacing / resized_shape`
            voxelspacing = tuple(np.array(ref_img.shape[1:]) * np.array(ref_voxelspacing) / np.array(img.shape[1:]))

        self.data[tag] = img

        if precomputed_attrs is not None:
            # If precomputed attributes are available, use them directly
            self.attrs[tag] = precomputed_attrs
        else:
            # Otherwise, measure/compute the attributes directly from the image/metadata
            self.attrs[tag] = {CardinalTag.voxelspacing: voxelspacing}
            if self._is_mask_tag(tag):
                self.attrs[tag].update(compute_mask_time_series_attributes(img, voxelspacing))

    def resize_image(
        self,
        source_img_tag: str,
        target_img_tag: str = None,
        target_img_size: Tuple[int, int] = None,
        target_voxelspacing: Tuple[float, float] = None,
        return_tag: bool = False,
    ) -> Union[Tuple[np.ndarray, Dict[str, Any]], Tuple[np.ndarray, Dict[str, Any], str]]:
        """Computes a resized version of an existing image in the view.

        Args:
            source_img_tag: Tag of the source image to resize.
            target_img_tag: Tag of the reference image to match in size. Mutually exclusive parameter with
                `target_img_size` and `target_voxelspacing`.
            target_img_size: Target height and width at which to resize the image. Mutually exclusive parameter with
                `target_img_tag` and `target_voxelspacing`.
            target_voxelspacing: Target height and width voxel size to obtain by resizing the image. Mutually exclusive
                parameter with `target_img_tag` and `target_img_size`.
            return_tag: Whether to return a suggestion for a tag under which to store the new image and its related
                data.

        Returns:
            Resized image, its attributes, and (optionally) a suggested tag under which to store it in the view.
        """
        if (bool(target_img_tag) + bool(target_img_size) + bool(target_voxelspacing)) != 1:
            raise ValueError(
                "You must provide exactly one parameter from which to derive the image's new size: `target_img_tag`, "
                "`target_img_size` or `target_voxelspacing`."
            )

        resample = Resampling.NEAREST if self._is_mask_tag(source_img_tag) else Resampling.BILINEAR
        if target_img_size:
            tag = f"{source_img_tag}_{target_img_size[0]}x{target_img_size[1]}"
            resized_img = resize_image(self.data[source_img_tag], size=target_img_size[::-1], resample=resample)
            source_voxelspacing = self.attrs[source_img_tag][CardinalTag.voxelspacing]
            # For each dimension, the voxelspacing equals to `shape * voxelspacing / resized_shape`
            target_voxelspacing = tuple(
                np.array(self.data[source_img_tag].shape[1:])
                * np.array(source_voxelspacing)
                / np.array(target_img_size)
            )
        elif target_voxelspacing:
            tag = f"{source_img_tag}_voxelspacing_{target_voxelspacing[0]}x{target_voxelspacing[1]}"
            resized_img = resize_image_to_voxelspacing(
                self.data[source_img_tag],
                self.attrs[source_img_tag][CardinalTag.voxelspacing],
                target_voxelspacing,
                resample=resample,
            )
        elif target_img_tag:
            tag = f"{source_img_tag}_resized_{target_img_tag}"
            resized_img = resize_image(
                self.data[source_img_tag], self.data[target_img_tag].shape[1:][::-1], resample=resample
            )
            target_voxelspacing = self.attrs[target_img_tag][CardinalTag.voxelspacing]
        else:
            raise AssertionError(
                "Either `target_img_tag`, `target_img_size` or `target_voxelspacing` should have been provided."
            )

        # Compute the attributes on the resized image
        attrs = {CardinalTag.voxelspacing: target_voxelspacing}
        if self._is_mask_tag(tag):
            # If the data is a segmentation mask, automatically compute attributes on the mask's structures
            attrs.update(compute_mask_time_series_attributes(resized_img, target_voxelspacing))

        output = (resized_img, attrs)
        if return_tag:
            output += (tag,)
        return output

    def get_mask_attributes(self, mask_tag: str) -> Dict[str, np.ndarray]:
        """Returns the attributes values w.r.t. time for a given mask.

        Args:
            mask_tag: Tag of the mask to get the attributes for.

        Returns:
            Dictionary of attributes and their values for the given mask.
        """
        return {attr: self.attrs[mask_tag][attr] for attr in TimeSeriesAttribute}

    @classmethod
    def from_dir(
        cls,
        patient_id: str,
        view: ViewEnum,
        data_roots: Sequence[Path],
        img_format: str = IMG_FORMAT,
        eager_loading: bool = False,
        overwrite_attrs_cache: bool = False,
    ) -> "View":
        """Loads and merges image data and metadata files, from multiple sources, related to a view from one patient.

        Args:
            patient_id: ID of the patient for whom to collect data.
            view: View for which to collect data.
            data_roots: Folders in which to look for files related to the patient's view. We look for files recursively
                inside the directories, so as long as the files respect the `IMG_FILENAME_PATTERN` pattern, they will be
                collected.
            img_format: File extension of the image files to load the data from.
            eager_loading: By default, `View` objects use lazy loading and only load images/compute attributes upon the
                first time they are accessed. Enabling eager loading will by-pass this behavior and force the `View` to
                load all the images and compute their attributes directly upon instantiation.
            overwrite_attrs_cache: Whether to discard the current cache of attributes and compute them again. Has no
                effect when no attributes cache exists.

        Returns:
            `View` instance, built using data related to the patient's view from the provided folders.
        """
        view_files_pattern = IMG_FILENAME_PATTERN.format(
            patient_id=patient_id, view=view, tag="*", ext=as_file_extension(img_format)
        )
        view_files = {}
        for data_root in data_roots:
            # Search recursively inside the provided directory
            for view_file in data_root.rglob(view_files_pattern):
                data_tag = remove_suffixes(view_file).name.split("_", 2)[-1]

                # Warn if the data provided by a file conflicts with the data provided by another file
                if previous_file := view_files.get(data_tag):
                    msg = (
                        f"'{data_tag}' data loaded from file '{view_file}' conflicts with the data provided by "
                        f"'{previous_file}'. "
                    )
                    if previous_file.suffixes != view_file.suffixes:
                        # If the conflict comes from files with same stems but different file extensions
                        msg += (
                            "The file extensions are not taken into account to identify the type of data, so if the "
                            "files contain different data, their names should differ by more than their suffixes."
                        )
                    else:
                        # The conflict comes from files with same names but under different roots
                        msg += (
                            "It seems you have copies of the same file under different folders. To avoid this warning, "
                            "you should delete one of the copies of the file (by default, we use the one from the root "
                            "that appears last in `data_root`)."
                        )
                    logger.warning(msg)

                view_files[data_tag] = view_file

        view_object = cls(id=cls.Id(patient_id, view))
        for data_tag, view_file in view_files.items():
            view_object._init_data_and_attrs(data_tag, view_file, overwrite_attrs_cache=overwrite_attrs_cache)
            if eager_loading:
                # Access the data to trigger data and attributes to be loaded
                _ = view_object.data[data_tag]
        return view_object

    def save(
        self,
        save_dir: Path,
        subdir_levels: Sequence[Literal["patient", "view"]] = None,
        include_tags: Sequence[str] = None,
        exclude_tags: Sequence[str] = None,
        overlay_pairs_tags: Sequence[Tuple[str, str]] = None,
        overlay_control_points: int = 0,
        resize_kwargs: Dict[str, Any] = None,
        img_format: str = IMG_FORMAT,
        io_backend_kwargs: Dict[str, Any] = None,
        cache_attrs: bool = True,
    ) -> None:
        """Saves the view's data and metadata as multiple files on disk.

        Args:
            save_dir: Directory where to save the data.
            subdir_levels: Levels of subdirectories to create under `save_dir`.
            include_tags: Keys in `data` and `attrs` to save to disk. Mutually exclusive parameter with `exclude_tags`.
                If not provided, will default to include all the available tags in the view, unless `overlay_pairs_tags`
                is provided (in which case no single tag will be included by default).
            exclude_tags: Keys in `data` and `attrs` to exclude from being saved to disk. Mutually exclusive parameter
                with `include_tags`.
            overlay_pairs_tags: Keys of image/mask pairs to save with the mask painted over the image. This option is
                only supported when using image formats that are supported by the Pillow backend.
            overlay_control_points: Number of control points to mark along the contours of the endocardium and
                epicardium in image/mask overlays.
            resize_kwargs: Parameters to pass along to `View.resize_image` to configure how to resize the images before
                saving them.
            img_format: File extension of the image format to save the data as.
            io_backend_kwargs: Arguments to pass along to the imaging backend (e.g. SimpleITK, Pillow, etc.) used to
                save the data. The backend selected will depend on the requested tags and image format.
            cache_attrs: Whether to also save (a cache of) attributes, to avoid having to compute them again when
                loading the view from disk in the future.
        """
        if subdir_levels:
            for subdir_level in subdir_levels:
                save_dir /= getattr(self.id, subdir_level)

        if include_tags is not None and exclude_tags is not None:
            raise ValueError(
                "`include_tags` and `exclude_tags` are mutually exclusive. Only one of them should be specified at a "
                "time."
            )
        if include_tags is None and overlay_pairs_tags:
            include_tags = []

        img_format = as_file_extension(img_format)
        use_pil_backend = img_format.lower() in PIL_SEQUENCE_FORMATS
        if overlay_pairs_tags and not use_pil_backend:
            raise ValueError(
                f"Masks overlaid on grayscale images can only be saved using Pillow, which supports the following "
                f"formats for sequences: {PIL_SEQUENCE_FORMATS}. Please explicitly set `img_format` to one of those "
                f"formats to use masks overlay."
            )
        if not overlay_pairs_tags and use_pil_backend:
            logger.warning(
                "Pillow does not handle natively categorical images such as segmentation masks, so to correctly "
                "display segmentation masks you should display them on top of their corresponding grayscale images "
                "using `overlay_pair_tags` with `<image_tag>,<mask_tag>`. You should only use `include_tags` or "
                "`exclude_tags` when working with grayscale images or when using image formats handled by the "
                "SimpleITK backend."
            )
        if io_backend_kwargs is None:
            io_backend_kwargs = {}

        # Determine which data to include/exclude from saving
        tags_to_save = [
            tag
            for tag in self.data
            if (include_tags is None or tag in include_tags)  # Only include tags from the inclusion list, if it exists
            and (exclude_tags is None or tag not in exclude_tags)  # Exclude tags from the exclusion list, if it exists
        ]

        # Create a view containing only the data to save
        data_to_save, attrs_to_save = {}, {}
        for tag in tags_to_save:
            if resize_kwargs:
                im, attrs, tag = self.resize_image(tag, **resize_kwargs, return_tag=True)
            else:
                im, attrs = self.data[tag], self.attrs[tag]
            data_to_save[tag], attrs_to_save[tag] = im, attrs

        if overlay_pairs_tags:
            # Create image/mask pairs with the mask painted over the image
            for im_tag, mask_tag in overlay_pairs_tags:
                if resize_kwargs:
                    seq_im, _, new_im_tag = self.resize_image(im_tag, **resize_kwargs, return_tag=True)
                    # Resize the mask using the image's shape directly rather the provided resize parameters to avoid
                    # floating point imprecision that can cause the image/mask shapes to be off by one pixel
                    seq_mask, seq_mask_attrs = self.resize_image(mask_tag, target_img_size=seq_im.shape[1:])
                    # So that the image/mask tags match, build the new mask tag from the one suggested for the image
                    mask_tag += new_im_tag.removeprefix(im_tag)
                    im_tag = new_im_tag
                else:
                    seq_im, _ = self.data[im_tag], self.attrs[im_tag]
                    seq_mask, seq_mask_attrs = self.data[mask_tag], self.attrs[mask_tag]

                rgb_overlay = np.array(
                    [color.label2rgb(mask, image=im, bg_label=Label.BG) for mask, im in zip(seq_mask, seq_im)]
                )

                if overlay_control_points:
                    # Overlay control points as white and yellow points, for the endo and epi respectively
                    struct_colors = {
                        "endo": mcolors.BASE_COLORS["w"],
                        "epi": mcolors.BASE_COLORS["y"],
                        "myo": mcolors.BASE_COLORS["m"],
                    }

                    # Create 2D dilation mask, then make it 3D (to vectorize dilation) by adding new frames filled with
                    # zeros
                    points_struct = disk(3)
                    points_struct = np.stack(
                        [np.zeros_like(points_struct), points_struct, np.zeros_like(points_struct)]
                    )

                    # For each of the structure for which to overlay control points
                    for struct_name, points_color in struct_colors.items():
                        # Identify the control points
                        control_points = EchoMeasure.control_points(
                            seq_mask,
                            Label.LV,
                            Label.MYO,
                            struct_name,
                            overlay_control_points,
                            voxelspacing=seq_mask_attrs[CardinalTag.voxelspacing],
                        )

                        # Create a boolean mask of the pixels corresponding to control points
                        control_points_mask = np.zeros_like(seq_mask)
                        for frame_idx, frame_control_points in enumerate(control_points):
                            control_points_mask[frame_idx][frame_control_points[:, 0], frame_control_points[:, 1]] = 1

                        # Dilate around the control points' pixels to make them more visible
                        control_points_mask = ndimage.binary_dilation(control_points_mask, structure=points_struct)

                        rgb_overlay[control_points_mask] = points_color  # Update the pixel value of control points

                # Because Pillow does not handle RGB images in format [0,1] (the output of `color.label2rgb`),
                # convert back to [0,255]
                # Reference: https://stackoverflow.com/a/55319979/12345114
                rgb_overlay = (rgb_overlay * 255).astype(np.uint8)
                data_to_save[f"{im_tag}_{mask_tag}"] = rgb_overlay

        # Determine the name to use for the files based on the view's metadata
        patient_id, view = self.id
        partial_filename = functools.partial(
            IMG_FILENAME_PATTERN.format, patient_id=patient_id, view=view, ext=img_format
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        if use_pil_backend:
            # Save images as animated GIFs using Pillow backend
            for tag, im_array in data_to_save.items():
                ims = [Image.fromarray(im_frame) for im_frame in im_array]
                save_kwargs = {"loop": 0, "duration": 40}
                save_kwargs.update(io_backend_kwargs)
                ims[0].save(save_dir / partial_filename(tag=tag), save_all=True, append_images=ims[1:], **save_kwargs)

        else:
            # Save images using SimpleITK backend
            for tag, im_array in data_to_save.items():
                sitk_save(
                    im_array.round(),  # Explicitly round since casting from float to int only floors the float values
                    save_dir / partial_filename(tag=tag),
                    spacing=(*attrs_to_save[tag][CardinalTag.voxelspacing][::-1], 1),
                    dtype=np.uint8,  # Use uint8 to save space, since both mask and grayscale values are within [0,255]
                )

        if cache_attrs:
            # Save a cache of attributes as Numpy zipped archive
            partial_filename = functools.partial(
                IMG_FILENAME_PATTERN.format,
                patient_id=patient_id,
                view=view,
                ext=as_file_extension(ATTRS_CACHE_FORMAT),
            )
            for tag in tags_to_save:
                np.savez(save_dir / partial_filename(tag=tag), **self.attrs[tag])
