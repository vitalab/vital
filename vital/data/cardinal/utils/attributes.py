import itertools
from typing import Any, Callable, Dict, Hashable, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from vital.data.cardinal.config import Label, TabularAttribute, TimeSeriesAttribute
from vital.data.cardinal.config import View as ViewEnum
from vital.metrics.evaluate.clinical.heart_us import compute_left_ventricle_volumes
from vital.utils.image.measure import T
from vital.utils.image.us.measure import EchoMeasure

TIME_SERIES_ATTR_LABELS = {
    **dict.fromkeys(
        [TimeSeriesAttribute.gls, TimeSeriesAttribute.ls_left, TimeSeriesAttribute.ls_right], "strain (in %)"
    ),
    **dict.fromkeys([TimeSeriesAttribute.lv_area], "area (in cm²)"),
    **dict.fromkeys([TimeSeriesAttribute.lv_length], "length (in cm)"),
    **dict.fromkeys(
        [TimeSeriesAttribute.myo_thickness_left, TimeSeriesAttribute.myo_thickness_right],
        "thickness (in cm)",
    ),
}
TABULAR_CAT_ATTR_LABELS = {
    TabularAttribute.sex: ["M", "W"],
    TabularAttribute.hf: [False, True],
    TabularAttribute.cad: [False, True],
    TabularAttribute.pad: [False, True],
    TabularAttribute.stroke: [False, True],
    TabularAttribute.tobacco: ["none", "ceased", "active"],
    TabularAttribute.diabetes: [False, True],
    TabularAttribute.dyslipidemia: [False, True],
    TabularAttribute.etiology: ["essential", "secondary", "pa"],
    TabularAttribute.bradycardic: [False, True],
    TabularAttribute.ace_inhibitor: [False, True],
    TabularAttribute.arb: [False, True],
    TabularAttribute.tz_diuretic: [False, True],
    TabularAttribute.central_acting: [False, True],
    TabularAttribute.beta_blocker: [False, True],
    TabularAttribute.spironolactone: [False, True],
    TabularAttribute.alpha_blocker: [False, True],
    TabularAttribute.ccb: [False, True],
    TabularAttribute.ht_severity: ["wht", "controlled", "uncontrolled"],
    TabularAttribute.ht_grade: ["0", "1", "2", "3"],
    TabularAttribute.nt_probnp_group: ["neutral", "end_organ_damage", "mortality_rate"],
    TabularAttribute.reduced_e_prime: [False, True],
    TabularAttribute.dilated_la: [False, True],
    TabularAttribute.d_dysfunction_e_e_prime_ratio: [False, True],
    TabularAttribute.ph_vmax_tr: [False, True],
    TabularAttribute.lvh: [False, True],
    TabularAttribute.diastolic_dysfunction_param_sum: ["0", "1", "2", "3", "4"],
    TabularAttribute.diastolic_dysfunction: ["none", "uncertain", "certain"],
    TabularAttribute.ht_cm: ["none", "uncertain", "certain"],
}
TABULAR_ATTR_UNITS = {
    **dict.fromkeys([TabularAttribute.ef], ("(in %)", int)),
    **dict.fromkeys([TabularAttribute.edv, TabularAttribute.esv], ("(in ml)", int)),
    **dict.fromkeys(
        [
            TabularAttribute.a4c_ed_sc_min,
            TabularAttribute.a4c_ed_sc_max,
            TabularAttribute.a4c_ed_lc_min,
            TabularAttribute.a4c_ed_lc_max,
            TabularAttribute.a2c_ed_ic_min,
            TabularAttribute.a2c_ed_ic_max,
            TabularAttribute.a2c_ed_ac_min,
            TabularAttribute.a2c_ed_ac_max,
        ],
        ("(in dm^-1)", float),
    ),
    TabularAttribute.age: ("(in years)", int),
    TabularAttribute.height: ("(in cm)", int),
    TabularAttribute.weight: ("(in kg)", float),
    TabularAttribute.bmi: ("(in kg/m²)", float),
    TabularAttribute.ddd: ("", float),
    **dict.fromkeys(
        [
            TabularAttribute.sbp_24,
            TabularAttribute.dbp_24,
            TabularAttribute.pp_24,
            TabularAttribute.sbp_day,
            TabularAttribute.dbp_day,
            TabularAttribute.pp_day,
            TabularAttribute.sbp_night,
            TabularAttribute.dbp_night,
            TabularAttribute.pp_night,
            TabularAttribute.sbp_tte,
            TabularAttribute.dbp_tte,
            TabularAttribute.pp_tte,
        ],
        ("(in mmHg)", int),
    ),
    TabularAttribute.hr_tte: ("(in bpm)", int),
    TabularAttribute.creat: ("(in µmol/L)", int),
    TabularAttribute.gfr: ("(in ml/min/1,73m²)", float),
    TabularAttribute.nt_probnp: ("(in pg/ml)", int),
    **dict.fromkeys([TabularAttribute.e_velocity, TabularAttribute.a_velocity], ("(in m/s)", float)),
    TabularAttribute.mv_dt: ("(in ms)", int),
    **dict.fromkeys([TabularAttribute.lateral_e_prime, TabularAttribute.septal_e_prime], ("(in cm/s)", int)),
    TabularAttribute.mean_e_prime: ("(in cm/s)", float),
    TabularAttribute.e_e_prime_ratio: ("", float),
    TabularAttribute.lvm_ind: ("(in g/m²)", int),
    TabularAttribute.la_volume: ("(in ml/m²)", float),
    TabularAttribute.la_area: ("(in cm²)", float),
    TabularAttribute.vmax_tr: ("(in m/s)", float),
    **dict.fromkeys([TabularAttribute.ivs_d, TabularAttribute.lvid_d, TabularAttribute.pw_d], ("(in cm)", float)),
    TabularAttribute.tapse: ("(in cm)", float),
    TabularAttribute.s_prime: ("(in cm/s)", float),
    **{
        cat_attr: (f"({'/'.join(str(category) for category in categories)})", Any)
        for cat_attr, categories in TABULAR_CAT_ATTR_LABELS.items()
    },
}
TABULAR_ATTR_GROUPS = {
    "info": [
        TabularAttribute.age,
        TabularAttribute.sex,
        TabularAttribute.height,
        TabularAttribute.weight,
        TabularAttribute.bmi,
    ],
    "medical_history": [
        TabularAttribute.hf,
        TabularAttribute.cad,
        TabularAttribute.pad,
        TabularAttribute.stroke,
        TabularAttribute.tobacco,
        TabularAttribute.diabetes,
        TabularAttribute.dyslipidemia,
    ],
    "holter": [
        TabularAttribute.sbp_24,
        TabularAttribute.dbp_24,
        TabularAttribute.pp_24,
        TabularAttribute.sbp_day,
        TabularAttribute.dbp_day,
        TabularAttribute.pp_day,
        TabularAttribute.sbp_night,
        TabularAttribute.dbp_night,
        TabularAttribute.pp_night,
    ],
    "biological_exam": [
        TabularAttribute.creat,
        TabularAttribute.gfr,
        TabularAttribute.nt_probnp,
    ],
    "hypertension": [
        TabularAttribute.etiology,
        TabularAttribute.nt_probnp_group,
        TabularAttribute.ht_grade,
        TabularAttribute.ht_severity,
        TabularAttribute.diastolic_dysfunction_param_sum,
        TabularAttribute.diastolic_dysfunction,
        TabularAttribute.ht_cm,
    ],
    "treatment": [
        TabularAttribute.ddd,
        TabularAttribute.bradycardic,
        TabularAttribute.ace_inhibitor,
        TabularAttribute.arb,
        TabularAttribute.tz_diuretic,
        TabularAttribute.central_acting,
        TabularAttribute.beta_blocker,
        TabularAttribute.spironolactone,
        TabularAttribute.alpha_blocker,
        TabularAttribute.ccb,
    ],
    "tte": [
        TabularAttribute.sbp_tte,
        TabularAttribute.dbp_tte,
        TabularAttribute.pp_tte,
        TabularAttribute.hr_tte,
        TabularAttribute.ef,
        TabularAttribute.edv,
        TabularAttribute.esv,
        TabularAttribute.a4c_ed_sc_min,
        TabularAttribute.a4c_ed_sc_max,
        TabularAttribute.a4c_ed_lc_min,
        TabularAttribute.a4c_ed_lc_max,
        TabularAttribute.a2c_ed_ic_min,
        TabularAttribute.a2c_ed_ic_max,
        TabularAttribute.a2c_ed_ac_min,
        TabularAttribute.a2c_ed_ac_max,
        TabularAttribute.e_velocity,
        TabularAttribute.a_velocity,
        TabularAttribute.mv_dt,
        TabularAttribute.lateral_e_prime,
        TabularAttribute.septal_e_prime,
        TabularAttribute.mean_e_prime,
        TabularAttribute.reduced_e_prime,
        TabularAttribute.e_e_prime_ratio,
        TabularAttribute.d_dysfunction_e_e_prime_ratio,
        TabularAttribute.la_volume,
        TabularAttribute.dilated_la,
        TabularAttribute.la_area,
        TabularAttribute.vmax_tr,
        TabularAttribute.ph_vmax_tr,
        TabularAttribute.lvm_ind,
        TabularAttribute.lvh,
        TabularAttribute.ivs_d,
        TabularAttribute.lvid_d,
        TabularAttribute.pw_d,
        TabularAttribute.tapse,
        TabularAttribute.s_prime,
    ],
}


def get_attr_sort_key(attr: TabularAttribute, label: str = None) -> int:
    """Sorts the attributes/labels in the order they appear in their respective enumerations.

    Args:
        attr: Name of the attribute to sort.
        label: Name of the label to sort, if the attribute is a categorical attribute.

    Returns:
        Integer key to sort the attributes and labels, in the order the attributes appear in `TabularAttribute` first,
        and then in the order the labels appear in `TABULAR_CAT_ATTR_LABELS`.
    """
    # Multiply the sorting key for the attribute by 10 to give it priority over the label
    sort_attr = list(TabularAttribute).index(attr) * 10
    # Check if label is `str` (rather than not being None) because nan (pandas NA marker) is not None
    # but is also considered being not defined
    sort_label = TABULAR_CAT_ATTR_LABELS[attr].index(label) if isinstance(label, str) else 0
    return sort_attr + sort_label


def compute_mask_time_series_attributes(mask: T, voxelspacing: Tuple[float, float]) -> Dict[TimeSeriesAttribute, T]:
    """Measures a variety of attributes on a (batch of) mask(s).

    Args:
        mask: ([N], H, W), Mask(s) on which to compute the attributes.
        voxelspacing: Size of the mask's voxels along each (height, width) dimension (in mm).

    Returns:
        Mapping between the attributes and ([N], 1) arrays of their values for each mask in the batch.
    """
    voxelarea = voxelspacing[0] * voxelspacing[1]
    return {
        TimeSeriesAttribute.gls: EchoMeasure.longitudinal_strain(mask, Label.LV, Label.MYO, voxelspacing=voxelspacing),
        TimeSeriesAttribute.ls_left: EchoMeasure.longitudinal_strain(
            mask, Label.LV, Label.MYO, num_control_points=31, control_points_slice=slice(11), voxelspacing=voxelspacing
        ),
        TimeSeriesAttribute.ls_right: EchoMeasure.longitudinal_strain(
            mask,
            Label.LV,
            Label.MYO,
            num_control_points=31,
            control_points_slice=slice(-11, None),
            voxelspacing=voxelspacing,
        ),
        # For the area, we have to convert from mm² to cm² (since this API is generic and does not return values in
        # units commonly used in echocardiography)
        TimeSeriesAttribute.lv_area: EchoMeasure.structure_area(mask, labels=Label.LV, voxelarea=voxelarea) * 1e-2,
        TimeSeriesAttribute.lv_length: EchoMeasure.lv_length(mask, Label.LV, Label.MYO, voxelspacing=voxelspacing),
        TimeSeriesAttribute.myo_thickness_left: EchoMeasure.myo_thickness(
            mask,
            Label.LV,
            Label.MYO,
            num_control_points=31,
            control_points_slice=slice(1, 11),
            voxelspacing=voxelspacing,
        ).mean(axis=-1),
        TimeSeriesAttribute.myo_thickness_right: EchoMeasure.myo_thickness(
            mask,
            Label.LV,
            Label.MYO,
            num_control_points=31,
            control_points_slice=slice(-11, -1),
            voxelspacing=voxelspacing,
        ).mean(axis=-1),
    }


def compute_mask_tabular_attributes(
    a4c_mask: np.ndarray,
    a4c_voxelspacing: Tuple[float, float],
    a2c_mask: np.ndarray,
    a2c_voxelspacing: Tuple[float, float],
    a4c_ed_frame: int = None,
    a4c_es_frame: int = None,
    a2c_ed_frame: int = None,
    a2c_es_frame: int = None,
) -> Dict[TabularAttribute, Union[int, float]]:
    """Measures tabular and clinically relevant attributes based on masks from orthogonal views, i.e. A4C and A2C.

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
    attrs = {TabularAttribute.ef: ef, TabularAttribute.edv: edv, TabularAttribute.esv: esv}

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
            TabularAttribute.a4c_ed_sc_min: _curvature(
                a4c_ed_mask, a4c_voxelspacing, min, control_points_slice=slice(2, 11)
            ),
            TabularAttribute.a4c_ed_sc_max: _curvature(
                a4c_ed_mask, a4c_voxelspacing, max, control_points_slice=slice(2, 11)
            ),
            TabularAttribute.a4c_ed_lc_min: _curvature(
                a4c_ed_mask, a4c_voxelspacing, min, control_points_slice=slice(-11, -2)
            ),
            TabularAttribute.a4c_ed_lc_max: _curvature(
                a4c_ed_mask, a4c_voxelspacing, max, control_points_slice=slice(-11, -2)
            ),
            TabularAttribute.a2c_ed_ic_min: _curvature(
                a2c_ed_mask, a2c_voxelspacing, min, control_points_slice=slice(2, 11)
            ),
            TabularAttribute.a2c_ed_ic_max: _curvature(
                a2c_ed_mask, a2c_voxelspacing, max, control_points_slice=slice(2, 11)
            ),
            TabularAttribute.a2c_ed_ac_min: _curvature(
                a2c_ed_mask, a2c_voxelspacing, min, control_points_slice=slice(-11, -2)
            ),
            TabularAttribute.a2c_ed_ac_max: _curvature(
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
    group_labels = TIME_SERIES_ATTR_LABELS  # By default, we have labels for each individual attribute
    if attr_name in (hue, style):
        # Identify groups of attributes with the same labels/units
        attr_units = {attr: unit.split(" ")[0] for attr, unit in TIME_SERIES_ATTR_LABELS.items()}
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
        group_labels.update({label.split(" ")[0]: label for label in TIME_SERIES_ATTR_LABELS.values()})
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
