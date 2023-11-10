import itertools
from typing import Any, Callable, Dict, Hashable, Iterator, Optional, Sequence, Tuple, Union

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
    **dict.fromkeys([ImageAttribute.gls, ImageAttribute.ls_left, ImageAttribute.ls_right], "strain (in %)"),
    **dict.fromkeys([ImageAttribute.lv_area], "area (in cm²)"),
    **dict.fromkeys([ImageAttribute.lv_length], "length (in cm)"),
    **dict.fromkeys(
        [ImageAttribute.myo_thickness_left, ImageAttribute.myo_thickness_right],
        "thickness (in cm)",
    ),
}
CLINICAL_CAT_ATTR_LABELS = {
    ClinicalAttribute.sex: ["M", "W"],
    ClinicalAttribute.hf: [False, True],
    ClinicalAttribute.cad: [False, True],
    ClinicalAttribute.pad: [False, True],
    ClinicalAttribute.stroke: [False, True],
    ClinicalAttribute.tobacco: ["none", "ceased", "active"],
    ClinicalAttribute.diabetes: [False, True],
    ClinicalAttribute.dyslipidemia: [False, True],
    ClinicalAttribute.etiology: ["essential", "secondary", "pa"],
    ClinicalAttribute.bradycardic: [False, True],
    ClinicalAttribute.ace_inhibitor: [False, True],
    ClinicalAttribute.arb: [False, True],
    ClinicalAttribute.tz_diuretic: [False, True],
    ClinicalAttribute.central_acting: [False, True],
    ClinicalAttribute.beta_blocker: [False, True],
    ClinicalAttribute.spironolactone: [False, True],
    ClinicalAttribute.alpha_blocker: [False, True],
    ClinicalAttribute.ccb: [False, True],
    ClinicalAttribute.ht_severity: ["wht", "controlled", "uncontrolled"],
    ClinicalAttribute.ht_grade: ["0", "1", "2", "3"],
    ClinicalAttribute.nt_probnp_group: ["neutral", "end_organ_damage", "mortality_rate"],
    ClinicalAttribute.reduced_e_prime: [False, True],
    ClinicalAttribute.dilated_la: [False, True],
    ClinicalAttribute.d_dysfunction_e_e_prime_ratio: [False, True],
    ClinicalAttribute.ph_vmax_tr: [False, True],
    ClinicalAttribute.lvh: [False, True],
    ClinicalAttribute.diastolic_dysfunction_param_sum: ["0", "1", "2", "3", "4"],
    ClinicalAttribute.diastolic_dysfunction: ["none", "uncertain", "certain"],
    ClinicalAttribute.ht_cm: ["none", "uncertain", "certain"],
}
CLINICAL_ATTR_UNITS = {
    **dict.fromkeys([ClinicalAttribute.ef], ("(in %)", int)),
    **dict.fromkeys([ClinicalAttribute.edv, ClinicalAttribute.esv], ("(in ml)", int)),
    **dict.fromkeys(
        [
            ClinicalAttribute.a4c_ed_sc_min,
            ClinicalAttribute.a4c_ed_sc_max,
            ClinicalAttribute.a4c_ed_lc_min,
            ClinicalAttribute.a4c_ed_lc_max,
            ClinicalAttribute.a2c_ed_ic_min,
            ClinicalAttribute.a2c_ed_ic_max,
            ClinicalAttribute.a2c_ed_ac_min,
            ClinicalAttribute.a2c_ed_ac_max,
        ],
        ("(in dm^-1)", float),
    ),
    ClinicalAttribute.age: ("(in years)", int),
    ClinicalAttribute.height: ("(in cm)", int),
    ClinicalAttribute.weight: ("(in kg)", float),
    ClinicalAttribute.bmi: ("(in kg/m²)", float),
    ClinicalAttribute.ddd: ("", float),
    **dict.fromkeys(
        [
            ClinicalAttribute.sbp_24,
            ClinicalAttribute.dbp_24,
            ClinicalAttribute.pp_24,
            ClinicalAttribute.sbp_day,
            ClinicalAttribute.dbp_day,
            ClinicalAttribute.pp_day,
            ClinicalAttribute.sbp_night,
            ClinicalAttribute.dbp_night,
            ClinicalAttribute.pp_night,
            ClinicalAttribute.sbp_tte,
            ClinicalAttribute.dbp_tte,
            ClinicalAttribute.pp_tte,
        ],
        ("(in mmHg)", int),
    ),
    ClinicalAttribute.hr_tte: ("(in bpm)", int),
    ClinicalAttribute.creat: ("(in µmol/L)", int),
    ClinicalAttribute.gfr: ("(in ml/min/1,73m²)", float),
    ClinicalAttribute.nt_probnp: ("(in pg/ml)", int),
    **dict.fromkeys([ClinicalAttribute.e_velocity, ClinicalAttribute.a_velocity], ("(in m/s)", float)),
    ClinicalAttribute.mv_dt: ("(in ms)", int),
    **dict.fromkeys([ClinicalAttribute.lateral_e_prime, ClinicalAttribute.septal_e_prime], ("(in cm/s)", int)),
    ClinicalAttribute.mean_e_prime: ("(in cm/s)", float),
    ClinicalAttribute.e_e_prime_ratio: ("", float),
    ClinicalAttribute.lvm_ind: ("(in g/m²)", int),
    ClinicalAttribute.la_volume: ("(in ml/m²)", float),
    ClinicalAttribute.la_area: ("(in cm²)", float),
    ClinicalAttribute.vmax_tr: ("(in m/s)", float),
    **dict.fromkeys([ClinicalAttribute.ivs_d, ClinicalAttribute.lvid_d, ClinicalAttribute.pw_d], ("(in cm)", float)),
    ClinicalAttribute.tapse: ("(in cm)", float),
    ClinicalAttribute.s_prime: ("(in cm/s)", float),
    **{
        cat_attr: (f"({'/'.join(str(category) for category in categories)})", Any)
        for cat_attr, categories in CLINICAL_CAT_ATTR_LABELS.items()
    },
}
CLINICAL_ATTR_GROUPS = {
    "info": [
        ClinicalAttribute.age,
        ClinicalAttribute.sex,
        ClinicalAttribute.height,
        ClinicalAttribute.weight,
        ClinicalAttribute.bmi,
    ],
    "medical_history": [
        ClinicalAttribute.hf,
        ClinicalAttribute.cad,
        ClinicalAttribute.pad,
        ClinicalAttribute.stroke,
        ClinicalAttribute.tobacco,
        ClinicalAttribute.diabetes,
        ClinicalAttribute.dyslipidemia,
    ],
    "holter": [
        ClinicalAttribute.sbp_24,
        ClinicalAttribute.dbp_24,
        ClinicalAttribute.pp_24,
        ClinicalAttribute.sbp_day,
        ClinicalAttribute.dbp_day,
        ClinicalAttribute.pp_day,
        ClinicalAttribute.sbp_night,
        ClinicalAttribute.dbp_night,
        ClinicalAttribute.pp_night,
    ],
    "biological_exam": [
        ClinicalAttribute.creat,
        ClinicalAttribute.gfr,
        ClinicalAttribute.nt_probnp,
    ],
    "hypertension": [
        ClinicalAttribute.etiology,
        ClinicalAttribute.nt_probnp_group,
        ClinicalAttribute.ht_grade,
        ClinicalAttribute.ht_severity,
        ClinicalAttribute.diastolic_dysfunction_param_sum,
        ClinicalAttribute.diastolic_dysfunction,
        ClinicalAttribute.ht_cm,
    ],
    "treatment": [
        ClinicalAttribute.ddd,
        ClinicalAttribute.bradycardic,
        ClinicalAttribute.ace_inhibitor,
        ClinicalAttribute.arb,
        ClinicalAttribute.tz_diuretic,
        ClinicalAttribute.central_acting,
        ClinicalAttribute.beta_blocker,
        ClinicalAttribute.spironolactone,
        ClinicalAttribute.alpha_blocker,
        ClinicalAttribute.ccb,
    ],
    "tte": [
        ClinicalAttribute.sbp_tte,
        ClinicalAttribute.dbp_tte,
        ClinicalAttribute.pp_tte,
        ClinicalAttribute.hr_tte,
        ClinicalAttribute.ef,
        ClinicalAttribute.edv,
        ClinicalAttribute.esv,
        ClinicalAttribute.a4c_ed_sc_min,
        ClinicalAttribute.a4c_ed_sc_max,
        ClinicalAttribute.a4c_ed_lc_min,
        ClinicalAttribute.a4c_ed_lc_max,
        ClinicalAttribute.a2c_ed_ic_min,
        ClinicalAttribute.a2c_ed_ic_max,
        ClinicalAttribute.a2c_ed_ac_min,
        ClinicalAttribute.a2c_ed_ac_max,
        ClinicalAttribute.e_velocity,
        ClinicalAttribute.a_velocity,
        ClinicalAttribute.mv_dt,
        ClinicalAttribute.lateral_e_prime,
        ClinicalAttribute.septal_e_prime,
        ClinicalAttribute.mean_e_prime,
        ClinicalAttribute.reduced_e_prime,
        ClinicalAttribute.e_e_prime_ratio,
        ClinicalAttribute.d_dysfunction_e_e_prime_ratio,
        ClinicalAttribute.la_volume,
        ClinicalAttribute.dilated_la,
        ClinicalAttribute.la_area,
        ClinicalAttribute.vmax_tr,
        ClinicalAttribute.ph_vmax_tr,
        ClinicalAttribute.lvm_ind,
        ClinicalAttribute.lvh,
        ClinicalAttribute.ivs_d,
        ClinicalAttribute.lvid_d,
        ClinicalAttribute.pw_d,
        ClinicalAttribute.tapse,
        ClinicalAttribute.s_prime,
    ],
}


def get_attr_sort_key(attr: ClinicalAttribute, label: str = None) -> int:
    """Sorts the attributes/labels in the order they appear in their respective enumerations.

    Args:
        attr: Name of the attribute to sort.
        label: Name of the label to sort, if the attribute is a categorical attribute.

    Returns:
        Integer key to sort the attributes and labels, in the order the attributes appear in `ClinicalAttribute` first,
        and then in the order the labels appear in `CLINICAL_CAT_ATTR_LABELS`.
    """
    # Multiply the sorting key for the attribute by 10 to give it priority over the label
    sort_attr = list(ClinicalAttribute).index(attr) * 10
    # Check if label is `str` (rather than not being None) because nan (pandas NA marker) is not None
    # but is also considered being not defined
    sort_label = CLINICAL_CAT_ATTR_LABELS[attr].index(label) if isinstance(label, str) else 0
    return sort_attr + sort_label


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
        ImageAttribute.gls: EchoMeasure.longitudinal_strain(mask, Label.LV, Label.MYO, voxelspacing=voxelspacing),
        ImageAttribute.ls_left: EchoMeasure.longitudinal_strain(
            mask, Label.LV, Label.MYO, num_control_points=31, control_points_slice=slice(11), voxelspacing=voxelspacing
        ),
        ImageAttribute.ls_right: EchoMeasure.longitudinal_strain(
            mask,
            Label.LV,
            Label.MYO,
            num_control_points=31,
            control_points_slice=slice(-11, None),
            voxelspacing=voxelspacing,
        ),
        # For the area, we have to convert from mm² to cm² (since this API is generic and does not return values in
        # units commonly used in echocardiography)
        ImageAttribute.lv_area: EchoMeasure.structure_area(mask, labels=Label.LV, voxelarea=voxelarea) * 1e-2,
        ImageAttribute.lv_length: EchoMeasure.lv_length(mask, Label.LV, Label.MYO, voxelspacing=voxelspacing),
        ImageAttribute.myo_thickness_left: EchoMeasure.myo_thickness(
            mask,
            Label.LV,
            Label.MYO,
            num_control_points=31,
            control_points_slice=slice(1, 11),
            voxelspacing=voxelspacing,
        ).mean(axis=-1),
        ImageAttribute.myo_thickness_right: EchoMeasure.myo_thickness(
            mask,
            Label.LV,
            Label.MYO,
            num_control_points=31,
            control_points_slice=slice(-11, -1),
            voxelspacing=voxelspacing,
        ).mean(axis=-1),
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

    # Compute the volumes and ejection fraction, rounding to the nearest integer
    edv, esv = compute_left_ventricle_volumes(**lv_volumes_fn_kwargs)
    edv, esv = int(round(edv, 0)), int(round(esv, 0))
    ef = int(round(100 * (edv - esv) / edv, 0))
    attrs = {ClinicalAttribute.ef: ef, ClinicalAttribute.edv: edv, ClinicalAttribute.esv: esv}

    # Compute the curvatures of the left/right walls in the A4C and A2C views
    def _curvature(
        segmentation: np.ndarray,
        voxelspacing: Tuple[float, float],
        reduce: Callable[[np.ndarray], float],
        control_points_slice: slice = None,
    ) -> float:
        k = EchoMeasure.curvature(
            segmentation,
            Label.LV,
            Label.MYO,
            structure="endo",
            num_control_points=31,
            control_points_slice=control_points_slice,
            voxelspacing=voxelspacing,
        )
        # Reduce the curvature along the control points to a single value + round to the nearest mm^-1
        reduced_k = round(reduce(k), 2)
        if isinstance(reduced_k, np.float_):
            # Convert from np.float to native float to avoid issues with YAML serialization when saving attributes
            reduced_k = reduced_k.item()
        return reduced_k

    a4c_ed_mask = a4c_mask[a4c_ed_frame if a4c_ed_frame else 0]
    a2c_ed_mask = a2c_mask[a2c_ed_frame if a2c_ed_frame else 0]
    # Skip the first 2 points near the edges of the endo (i.e. near the base), since we're using second derivatives
    # and therefore their estimations might be less accurate
    attrs.update(
        {
            ClinicalAttribute.a4c_ed_sc_min: _curvature(
                a4c_ed_mask, a4c_voxelspacing, min, control_points_slice=slice(2, 11)
            ),
            ClinicalAttribute.a4c_ed_sc_max: _curvature(
                a4c_ed_mask, a4c_voxelspacing, max, control_points_slice=slice(2, 11)
            ),
            ClinicalAttribute.a4c_ed_lc_min: _curvature(
                a4c_ed_mask, a4c_voxelspacing, min, control_points_slice=slice(-11, -2)
            ),
            ClinicalAttribute.a4c_ed_lc_max: _curvature(
                a4c_ed_mask, a4c_voxelspacing, max, control_points_slice=slice(-11, -2)
            ),
            ClinicalAttribute.a2c_ed_ic_min: _curvature(
                a2c_ed_mask, a2c_voxelspacing, min, control_points_slice=slice(2, 11)
            ),
            ClinicalAttribute.a2c_ed_ic_max: _curvature(
                a2c_ed_mask, a2c_voxelspacing, max, control_points_slice=slice(2, 11)
            ),
            ClinicalAttribute.a2c_ed_ac_min: _curvature(
                a2c_ed_mask, a2c_voxelspacing, min, control_points_slice=slice(-11, -2)
            ),
            ClinicalAttribute.a2c_ed_ac_max: _curvature(
                a2c_ed_mask, a2c_voxelspacing, max, control_points_slice=slice(-11, -2)
            ),
        }
    )

    return attrs


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
