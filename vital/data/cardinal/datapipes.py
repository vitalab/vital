import functools
import itertools
from typing import Any, Callable, Dict, Literal, Sequence, Tuple, TypeVar, Union

import numpy as np
import torchdata.datapipes as dp
from torch import Tensor
from torch.utils.data import MapDataPipe
from torchvision.transforms import transforms

from vital.data.cardinal.config import CardinalTag, ClinicalAttribute, ImageAttribute
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.utils.attributes import CLINICAL_CAT_ATTR_LABELS
from vital.data.cardinal.utils.data_struct import Patient
from vital.data.cardinal.utils.itertools import Patients

T = TypeVar("T", np.ndarray, Tensor)
PatientData = Dict[
    Union[str, ClinicalAttribute, ViewEnum], Union[Patient.Id, T, Dict[Union[ImageAttribute, CardinalTag], T]]
]
AttributeTransform = Callable[[np.ndarray], np.ndarray]

MISSING_NUM_ATTR = np.nan
MISSING_CAT_ATTR = -1


def build_datapipes(
    patients: Patients,
    process_patient_kwargs: Dict[str, Any] = None,
    transform_patient_kwargs: Dict[str, Any] = None,
    cache: bool = True,
) -> MapDataPipe[PatientData]:
    """Builds a pipeline of datapipes for the Cardinal dataset.

    Args:
        patients: Collection of patients to process in the pipeline of datapipes.
        process_patient_kwargs: Parameter to forward to the `process_patient` function.
        transform_patient_kwargs: Parameter to forward to the `transform_patient_data` function.
        cache: Whether to store elements at the end of the pipeline in memory (to avoid repetitive I/O and processing).

    Returns:
        Pipeline of datapipes for the Cardinal dataset.
    """
    if process_patient_kwargs is None:
        process_patient_kwargs = {}
    if transform_patient_kwargs is None:
        transform_patient_kwargs = {}

    datapipe = dp.map.SequenceWrapper(list(patients.keys()))
    datapipe = datapipe.map(patients.__getitem__)  # Load patient from disk
    datapipe = datapipe.map(functools.partial(process_patient, **process_patient_kwargs))
    if cache:
        # Since the Cardinal is relatively small, it can be cached in memory to avoid repetitive I/O
        datapipe = datapipe.in_memory_cache()
    datapipe = datapipe.map(functools.partial(transform_patient_data, **transform_patient_kwargs))
    return datapipe


def process_patient(
    patient: Patient,
    clinical_attributes: Sequence[ClinicalAttribute] = None,
    image_attributes: Sequence[ImageAttribute] = None,
    mask_tag: str = CardinalTag.mask,
    bmode_tag: str = None,
) -> PatientData:
    """Processes a patient to extract and format a subset of relevant fields from all the patient's data.

    Args:
        patient: Patient to process.
        clinical_attributes: Clinical attributes to extract from the patient.
        image_attributes: Image attributes to extract from the patient.
        mask_tag: Tag of the segmentation mask to use for computing the image attributes, and to include in the item.
        bmode_tag: Tag of the B-mode image to which the segmentation mask applies. Providing it causes the returned item
            to include image data.

    Returns:
        Each item returned contains two types of attributes: image attributes and clinical attributes.
        - Image attributes are 1D vectors w.r.t. time (one value per frame per view).
        - Clinical attributes are scalars (one value overall for the patient).
        If a `bmode_tag` value was provided, the items will also contain the B-mode and mask sequences from the patient.
    """
    if clinical_attributes is None:
        clinical_attributes = ClinicalAttribute
    if image_attributes is None:
        image_attributes = ImageAttribute
    img_data_tags = [bmode_tag, mask_tag] if bmode_tag else []

    if clinical_attributes:
        # Extract the requested clinical attributes from all the ones available for the patient,
        clinical_attrs_data = {attr_tag: patient.attrs.get(attr_tag) for attr_tag in clinical_attributes}

        for attr_tag, attr in clinical_attrs_data.items():
            if attr_tag in ClinicalAttribute.numerical_attrs():
                # Convert numerical attributes to numpy arrays of dtype `np.float32`
                clinical_attrs_data[attr_tag] = np.array(
                    attr if attr is not None else MISSING_NUM_ATTR, dtype=np.float32
                )
            else:
                # Convert categorical attributes to numerical labels inside numpy arrays of dtype `np.int64`
                clinical_attrs_data[attr_tag] = np.array(
                    CLINICAL_CAT_ATTR_LABELS[attr_tag].index(attr) if attr is not None else MISSING_CAT_ATTR,
                    dtype=np.int64,
                )
    else:
        clinical_attrs_data = {}

    if image_attributes:
        # Prepare attributes computed from the images
        mask_attrs_data = patient.get_mask_attributes(mask_tag)
        # Make sure the attributes array are of dtype `np.float32`, so that they'll be converted to dtype `torch.float`
        mask_attrs_data = {
            view_enum: {
                attr_tag: attr.astype(np.float32)
                for attr_tag, attr in view_data.items()
                if attr_tag in image_attributes
            }
            for view_enum, view_data in mask_attrs_data.items()
        }
    else:
        mask_attrs_data = {}

    item = {"id": patient.id, **clinical_attrs_data, **mask_attrs_data}

    # Add the image data from the types of data listed in `self.img_data_tags`
    for img_data_tag, (view_enum, view) in itertools.product(img_data_tags, patient.views.items()):
        # In case we have images for views for which we did not have attributes, make sure to create a
        # dictionary for the view's data
        item.setdefault(view_enum, {})[img_data_tag] = view.data[img_data_tag]

    return item


def transform_patient_data(
    patient_data: PatientData,
    clinical_attributes_transforms: Dict[Union[Literal["any"], ClinicalAttribute], AttributeTransform] = None,
    image_attributes_transforms: Dict[Union[Literal["any"], ImageAttribute], AttributeTransform] = None,
) -> PatientData:
    """Applies transformations to patient, with generic funcs applied to modalities or attribute-specific funcs.

    Args:
        patient_data: Patient data.
        clinical_attributes_transforms: Mapping between clinical attributes and the transformations to apply to them.
            Transformations under the `any` key will be applied to all the attributes.
        image_attributes_transforms: Mapping between image attributes and the transformations to apply to them.
            Transformations under the `any` key will be applied to all the attributes.

    Returns:
        Patient data, where the requested clinical and image attributes where transformed.
    """
    if clinical_attributes_transforms is None:
        clinical_attributes_transforms = {}
    if image_attributes_transforms is None:
        image_attributes_transforms = {}

    clinical_attrs = [key for key in patient_data if key in list(ClinicalAttribute)]
    views = [key for key in patient_data if key in list(ViewEnum)]

    # Apply clinical attributes transforms
    for clinical_attr in clinical_attrs:
        # Store the original dtype of the attribute
        attr_dtype = patient_data[clinical_attr].dtype

        # If a generic image attribute transform is provided, apply it first
        if transform := clinical_attributes_transforms.get("any"):
            patient_data[clinical_attr] = transform(patient_data[clinical_attr])

        # If a transformation targets the current attribute specifically, apply it
        if transform := clinical_attributes_transforms.get(clinical_attr):
            patient_data[clinical_attr] = transform(patient_data[clinical_attr])

        # Make sure that the dtype of the attribute was not modified by the transforms
        patient_data[clinical_attr] = patient_data[clinical_attr].astype(attr_dtype)

    # Apply image attributes transforms
    for view_data in (patient_data[view] for view in views):
        for img_attr in view_data:
            # Store the original dtype of the attribute
            attr_dtype = view_data[img_attr].dtype

            # If a generic image attribute transform is provided, apply it first
            if transform := image_attributes_transforms.get("any"):
                view_data[img_attr] = transform(view_data[img_attr])

            # If a transformation targets the current attribute specifically, apply it
            if transform := image_attributes_transforms.get(img_attr):
                view_data[img_attr] = transform(view_data[img_attr])

            # Make sure that the dtype of the attribute was not modified by the transforms
            view_data[img_attr] = view_data[img_attr].astype(attr_dtype)

    return patient_data


def filter_image_attributes(
    item_or_batch: PatientData,
    views: Sequence[ViewEnum] = tuple(ViewEnum),
    attributes: Sequence[ImageAttribute] = tuple(ImageAttribute),
) -> Dict[Tuple[ViewEnum, ImageAttribute], T]:
    """Filters an item/batch of data from a `PatientProcessor` to only keep requested image attributes from some views.

    Args:
        item_or_batch: Item/batch of data from a `PatientProcessor`.
        views: Views for which to keep image attributes data.
        attributes: Attributes for which to keep data.

    Returns:
        Requested image attributes from the requested views in the item/batch of data.
    """
    return {
        (view_enum, view_data_tag): data
        for view_enum in views
        for view_data_tag, data in item_or_batch.get(view_enum, {}).items()
        if view_data_tag in attributes
    }


if __name__ == "__main__":
    from argparse import ArgumentParser

    import matplotlib.pyplot as plt
    from PIL import Image
    from skimage import color

    from vital.data.cardinal.config import Label
    from vital.data.cardinal.utils.attributes import build_attributes_dataframe, plot_attributes_wrt_time
    from vital.utils.importlib import import_from_module
    from vital.utils.parsing import yaml_flow_collection

    parser = ArgumentParser()
    parser = Patients.add_args(parser)
    parser.add_argument(
        "--mask_tag",
        type=str,
        default=CardinalTag.mask,
        help="Tag of the segmentation mask to use for computing the image attributes, and to include in the item if "
        "`include_img_data=True`",
    )
    parser.add_argument(
        "--bmode_tag",
        type=str,
        help="Tag of the B-mode image to which the segmentation mask applies, only used if image data is to be "
        "included (`include_img_data=True`)",
    )
    parser.add_argument(
        "--image_attributes_transforms",
        type=yaml_flow_collection,
        default={},
        metavar="{XFORM1_CLASSPATH:{ARG1:VAL1,...},...}",
        help="Transformations to apply to the attributes extracted from the images",
    )
    parser.add_argument(
        "--debug_transforms",
        action="store_true",
        help="Whether to plot the original data alongside the transformed data to visualize the transformations' "
        "impact",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    mask_tag, bmode_tag, image_attributes_transforms, debug_transforms = (
        kwargs.pop("mask_tag"),
        kwargs.pop("bmode_tag"),
        kwargs.pop("image_attributes_transforms"),
        kwargs.pop("debug_transforms"),
    )

    # Automatically build transforms from module paths
    image_attributes_transforms = {
        "any": transforms.Compose(
            [
                import_from_module(transform_classpath)(**transform_kwargs)
                for transform_classpath, transform_kwargs in image_attributes_transforms.items()
            ]
        )
    }

    # Create the patients' datapipe
    patients = Patients(**kwargs)
    process_patient_kwargs = {"mask_tag": mask_tag, "bmode_tag": bmode_tag}
    datapipe = build_datapipes(
        patients,
        process_patient_kwargs=process_patient_kwargs,
        transform_patient_kwargs={"image_attributes_transforms": image_attributes_transforms},
    )

    # Draw the first patient from the datapipe
    first_item = datapipe[0]
    patient_id = list(patients.keys())[0]

    if debug_transforms:
        # Create another datapipe that doesn't apply transforms to gather untransformed data
        datapipe_no_xform = build_datapipes(patients, process_patient_kwargs=process_patient_kwargs)
        first_item_no_xform = datapipe_no_xform[0]

    for view, data in ((view, first_item[view]) for view in ViewEnum if view in first_item):
        # Plot image data
        if bmode_tag:
            overlaid_img = np.array(
                [
                    color.label2rgb(mask, image=im, bg_label=Label.BG)
                    for mask, im in zip(data[args.mask_tag], data[args.bmode_tag])
                ]
            )
            overlaid_img = (overlaid_img * 255).astype(np.uint8)
            overlaid_frames = [Image.fromarray(overlaid_frame) for overlaid_frame in overlaid_img]
            # TODO Display image data interactively as an animated GIF
            # overlaid_frames[0].show(f"{patient_id}/{view_enum} {args.bmode_tag}/{args.mask_tag}")

        attrs = {}

        # Start by collecting untransformed data so that it will be plotted first in the graph (more natural plot)
        if debug_transforms:
            # If debugging transforms, also extract image attributes from the original item
            no_xform_view_data = first_item_no_xform[view]
            attrs["og"] = {
                attr_tag: attr for attr_tag, attr in no_xform_view_data.items() if attr_tag in list(ImageAttribute)
            }

        # Extract the image attributes from the item
        item_tag = mask_tag if not debug_transforms else "xform"
        attrs[item_tag] = {attr_tag: attr for attr_tag, attr in data.items() if attr_tag in list(ImageAttribute)}

        # Plot the curves for each attribute w.r.t. time
        for _ in plot_attributes_wrt_time(
            build_attributes_dataframe(attrs, normalize_time=True), plot_title_root=f"{patient_id}/{view}"
        ):
            plt.show()
            plt.close()
