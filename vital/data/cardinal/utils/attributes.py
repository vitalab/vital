import itertools
from typing import Dict, Hashable, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from vital.data.cardinal.config import ClinicalAttribute, ImageAttribute, Label
from vital.data.cardinal.config import View as ViewEnum
from vital.metrics.evaluate.clinical.heart_us import compute_left_ventricle_volumes
from vital.utils.image.measure import T
from vital.utils.image.us.measure import EchoMeasure

IMAGE_ATTR_LABELS = {
    **dict.fromkeys([ImageAttribute.gls], "strain (in %)"),
    **dict.fromkeys([ImageAttribute.lv_area, ImageAttribute.myo_area], "area (in mm²)"),
    **dict.fromkeys([ImageAttribute.lv_length, ImageAttribute.lv_base_width], "length (in mm)"),
    **dict.fromkeys([ImageAttribute.lv_orientation], "angle (in degrees)"),
    **dict.fromkeys([ImageAttribute.epi_center_x, ImageAttribute.epi_center_y], "position (in pixels)"),
}


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


def compute_clinical_attributes(
    a4c_mask: np.ndarray,
    a4c_voxelspacing: Tuple[float, float],
    a2c_mask: np.ndarray,
    a2c_voxelspacing: Tuple[float, float],
    a4c_ed_frame: int = None,
    a4c_es_frame: int = None,
    a2c_ed_frame: int = None,
    a2c_es_frame: int = None,
) -> Dict[ClinicalAttribute, Union[int, float]]:
    """Measures a variety of clinical attributes based on masks from orthogonal views, i.e. A4C and A2C.

    Args:
        a4c_mask: (N1, H1, W1), Mask of the A4C view.
        a4c_voxelspacing: Size of the A4C mask's voxels along each (height, width) dimension (in mm).
        a2c_mask: (N2, H2, W2), Mask of the A2C view.
        a2c_voxelspacing: Size of the A2C mask's voxels along each (height, width) dimension (in mm).
        a4c_ed_frame: If the clinical attribute are computed on predictions rather than on reference, this is used to
            specify the index of the ED frame in the reference segmentation of the A4C view.
        a4c_es_frame: If the clinical attribute are computed on predictions rather than on reference, this is used to
            specify the index of the ES frame in the reference segmentation of the A4C view.
        a2c_ed_frame: If the clinical attribute are computed on predictions rather than on reference, this is used to
            specify the index of the ED frame in the reference segmentation of the A2C view.
        a2c_es_frame: If the clinical attribute are computed on predictions rather than on reference, this is used to
            specify the index of the ES frame in the reference segmentation of the A2C view.

    Returns:
        Mapping between the clinical attributes and their scalar values.
    """
    # Extract the relevant frames from the sequences (i.e. ED and ES) to compute clinical attributes
    lv_volumes_fn_kwargs = {}
    for view, lv_mask, voxelspacing, ed_frame, es_frame in [
        (ViewEnum.A4C, np.isin(a4c_mask, Label.LV), a4c_voxelspacing, a4c_ed_frame, a4c_es_frame),
        (ViewEnum.A2C, np.isin(a2c_mask, Label.LV), a2c_voxelspacing, a2c_ed_frame, a2c_es_frame),
    ]:
        voxelarea = voxelspacing[0] * voxelspacing[1]

        # Identify the ES frame in a view as the frame where the LV is the smallest (in 2D)
        if ed_frame is None:
            ed_frame = 0
        if es_frame is None:
            es_frame = np.argmin(EchoMeasure.structure_area(lv_mask, voxelarea=voxelarea))
        view_prefix = view.lower() + "_"
        lv_volumes_fn_kwargs.update(
            {
                view_prefix + "ed": lv_mask[ed_frame],
                view_prefix + "es": lv_mask[es_frame],
                view_prefix + "voxelspacing": voxelspacing,
            }
        )

    # Compute the clinical attributes
    edv, esv = compute_left_ventricle_volumes(**lv_volumes_fn_kwargs)
    ef = int(round(100 * (edv - esv) / edv))

    return {ClinicalAttribute.ef: ef, ClinicalAttribute.edv: edv, ClinicalAttribute.esv: esv}


def build_attributes_dataframe(
    attrs: Dict[Hashable, Dict[str | Sequence[str], np.ndarray]],
    outer_name: str = "data",
    inner_name: str = "attr",
    value_name: str = "val",
    time_name: str = "time",
    normalize_time: bool = True,
) -> pd.DataFrame:
    """Builds a dataframe of attributes data in long format.

    Args:
        attrs: Nested dictionaries of i) data from which the attributes were extracted, and ii) attributes themselves.
        outer_name: Name to give to the dataframe column representing the outer key in `attrs`.
        inner_name: Name to give to the dataframe column representing the inner key in `attrs`, i.e. the tag associated
            with each individual array of attributes.
        value_name: Name to give to the dataframe column representing the individual values in the attributes' data.
        time_name: Name to give to the dataframe column representing the time-index of the attributes' data.
        normalize_time: Whether to normalize the values in `time_name` between 0 and 1. By default, these values are
            between 0 and the count of data points in the attributes' data.

    Returns:
        Dataframe of attributes data in long format.
    """
    df_by_data = {}
    for data_tag, data_attrs in attrs.items():
        df_by_attrs = []
        # Convert each attributes' values to a dataframe, which will be concatenated afterwards
        for attr_tag, attr in data_attrs.items():
            if isinstance(attr_tag, tuple):
                # If the tag identifying an attribute is a multi-value key represented by a tuple,
                # merge the multiple values in one unique key, since pandas can't broadcast the tag otherwise
                attr_tag = "/".join(attr_tag)
            df_by_attrs.append(pd.DataFrame({inner_name: attr_tag, value_name: attr}))
        df = pd.concat(df_by_attrs)
        df[time_name] = df.groupby(inner_name).cumcount()
        if normalize_time:
            # Normalize the time, which by default is the index of the value in the data points, between 0 and 1
            df[time_name] /= df.groupby(inner_name)[time_name].transform("count") - 1
        df_by_data[data_tag] = df
    attrs_df = pd.concat(df_by_data).rename_axis([outer_name, None]).reset_index(0).reset_index(drop=True)
    return attrs_df


def plot_attributes_wrt_time(
    attrs: pd.DataFrame,
    plot_title_root: str = None,
    data_name: str = "data",
    attr_name: str = "attr",
    value_name: str = "val",
    time_name: str = "time",
    hue: Optional[str] = "attr",
    style: Optional[str] = "data",
    display_title_in_plot: bool = True,
) -> Iterator[Tuple[str, Axes]]:
    """Produces individual plots for each group of attributes in the data.

    Args:
        attrs: Dataframe of attributes data in long format.
        plot_title_root: Common base of the plots' titles, to append based on each plot's group of attributes.
        data_name: Name of the column in `attrs` representing the data the attributes come from.
        attr_name: Name of the column in `attrs` representing the name of the attributes.
        value_name: Name of the column in `attrs` representing the individual values in the attributes' data.
        time_name: Name of the column in `attrs` representing the time-index of the attributes' data.
        hue: Field of the attributes' data to use to assign the curves' hues.
        style: Field of the attributes' data to use to assign the curves' styles.
        display_title_in_plot: Whether to display the title generated for the plot in the plot itself.

    Returns:
        An iterator over pairs of plots for each group of attributes in the data and their titles.
    """
    group_labels = IMAGE_ATTR_LABELS  # By default, we have labels for each individual attribute
    if attr_name in (hue, style):
        # Identify groups of attributes with the same labels/units
        attr_units = {attr: unit.split(" ")[0] for attr, unit in IMAGE_ATTR_LABELS.items()}
        attrs_replaced_by_units = attrs.replace({attr_name: attr_units})  # Replace attrs by their units
        # Create masks for each unique label/unit (which can group more than one attribute)
        group_masks = {
            unit: attrs_replaced_by_units[attr_name].str.contains(unit)
            for unit in set(attr_units.values())
            if unit in attrs_replaced_by_units[attr_name].values  # Discard empty groups
        }

        # Add individual attributes that are not part of any group of attributes
        grouped_attr_mask = np.logical_or.reduce(list(group_masks.values()))
        group_masks.update({attr: attrs[attr_name] == attr for attr in attrs[~grouped_attr_mask][attr_name].unique()})

        # For each group of attributes, map the short label to the full version
        group_labels.update({label.split(" ")[0]: label for label in IMAGE_ATTR_LABELS.values()})
    else:
        group_masks = {attr: attrs[attr_name] == attr for attr in attrs[attr_name].unique()}

    if data_name in (hue, style):
        data_masks = {None: [True] * len(attrs)}
    else:
        data_masks = {data: attrs[data_name] == data for data in attrs[data_name].unique()}

    plot_title_root = [plot_title_root] if plot_title_root else []
    for (group_tag, group_mask), (data_tag, data_mask) in itertools.product(group_masks.items(), data_masks.items()):
        plot_title_parts = plot_title_root.copy()
        if group_tag:
            plot_title_parts.append(group_tag)
        if data_tag:
            plot_title_parts.append(data_tag)
        plot_title = "/".join(plot_title_parts)
        with sns.axes_style("darkgrid"):
            lineplot = sns.lineplot(data=attrs[group_mask & data_mask], x=time_name, y=value_name, hue=hue, style=style)
        lineplot_set_kwargs = {"ylabel": group_labels.get(group_tag, value_name)}
        if display_title_in_plot:
            lineplot_set_kwargs["title"] = plot_title
        lineplot.set(**lineplot_set_kwargs)
        # Escape invalid filename characters in the returned title, in case the caller wants to use the title as part of
        # the filename
        yield plot_title.replace("/", "_"), lineplot
