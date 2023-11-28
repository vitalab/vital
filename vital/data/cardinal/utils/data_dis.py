import itertools
import logging
from typing import Dict, Iterator, List, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from seaborn import PairGrid
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from vital.data.cardinal.config import CardinalTag, TabularAttribute, TimeSeriesAttribute
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.utils.attributes import (
    TABULAR_ATTR_UNITS,
    TABULAR_CAT_ATTR_LABELS,
    TIME_SERIES_ATTR_LABELS,
    build_attributes_dataframe,
)
from vital.data.cardinal.utils.data_struct import Patient
from vital.data.cardinal.utils.itertools import Patients
from vital.data.transforms import Interp1d

logger = logging.getLogger(__name__)


def check_subsets(patients: Sequence[Patient.Id], subsets: Dict[str, Sequence[Patient.Id]]) -> None:
    """Checks the lists of patients overall and in each subset to ensure each patient belongs to one and only subset.

    Args:
        patients: Collection of patients.
        subsets: Lists of patients making up subsets.
    """
    for (subset1, subset1_patients), (subset2, subset2_patients) in itertools.combinations(subsets.items(), 2):
        if intersect := set(subset1_patients) & set(subset2_patients):
            raise RuntimeError(
                f"All provided subsets should be disjoint from each other, but subsets {subset1}' and '{subset2}' "
                f"have the following patients in common: {sorted(intersect)}."
            )

    patient_ids_in_subsets = set().union(*subsets.values())
    if unassigned_patients := set(patients) - patient_ids_in_subsets:
        raise RuntimeError(
            f"All patients should be part of one of the subset. However, the following patients are not included in "
            f"any subset: {sorted(unassigned_patients)}."
        )


def plot_patients_distribution(
    patients: Patients,
    plot_attributes: Sequence[TabularAttribute],
    subsets: Dict[str, Sequence[Patient.Id]] = None,
    progress_bar: bool = False,
) -> PairGrid:
    """Plots the pairwise relationships between tabular attributes for a collection (or multiple subsets) of patients.

    Args:
        patients: Collection of patients.
        plot_attributes: Patients' tabular attributes whose distributions to compare pairwise.
        subsets: Lists of patients making up each subset, to plot with different hues. The subsets should be disjoint
            from one another.
        progress_bar: If ``True``, enables progress bars detailing the progress of the collecting data from patients.

    Returns:
        PairGrid representing the pairwise relationships between tabular attributes for the patients.
    """
    if subsets is not None:
        check_subsets(list(patients), subsets)

    patients = patients.values()
    msg = "Collecting plotting data from patients"
    if progress_bar:
        patients = tqdm(patients, desc=msg, unit="patient")
    else:
        logger.info(msg + "...")

    # Collect the data of the attributes to plot from the patients
    patients_data = {patient.id: {attr: patient.attrs[attr] for attr in plot_attributes} for patient in patients}

    # Format the patients' data as a dataframe
    patients_data = pd.DataFrame.from_dict(patients_data, orient="index")
    # Add units to attributes names (to be used as labels in the plot)
    patients_data = patients_data.rename(
        columns={attr: " ".join([attr, attr_unit]) for attr, (attr_unit, _) in TABULAR_ATTR_UNITS.items()}
    )

    # Add additional subset information to the dataframe, if provided
    plot_kwargs = {}
    if subsets:
        plot_kwargs.update({"hue": "subset", "hue_order": list(subsets)})
        patients_data["subset"] = pd.Series(dtype=str)
        for subset, patient_ids_in_subset in subsets.items():
            patients_data.loc[list(set(patient_ids_in_subset) & set(patients_data.index)), "subset"] = subset

    # Plot pairwise relationships between the selected attributes in the dataset
    g = sns.pairplot(patients_data, **plot_kwargs)
    g.map_lower(sns.kdeplot)
    return g


def plot_tabular_attrs_wrt_group(
    patients: Patients,
    groups: Mapping[Patient.Id, int | str],
    tabular_attrs: Sequence[TabularAttribute] = None,
    groups_desc: str = "group",
    cat_plot_kwargs: dict = None,
    num_plot_kwargs: dict = None,
) -> Iterator[Tuple[str, Axes]]:
    """Plots the variability of tabular attributes across groups of patients.

    Args:
        patients: Collection of patients data from which to extract the attributes.
        groups: Groups of patients, represented as a mapping between patient IDs and group labels.
        tabular_attrs: Subset of tabular attributes for which to plot the distribution by group. If not provided, will
            default to all available attributes.
        groups_desc: Description of the groups of patients over which the tabular attributes are aggregated.
        cat_plot_kwargs: Parameters to forward to the call to `seaborn.barplot` for categorical attributes.
        num_plot_kwargs: Parameters to forward to the call to `seaborn.boxplot` for numerical attributes.

    Returns:
        Iterator over figures (and their corresponding titles) plotting the variability of tabular attributes over
        groups of patients.
    """
    if cat_plot_kwargs is None:
        cat_plot_kwargs = {}
    if num_plot_kwargs is None:
        num_plot_kwargs = {}

    if groups_desc in TabularAttribute.categorical_attrs():
        # If the group corresponds to a categorical attribute, use the labels of the attribute as the group labels
        # so that the order of the labels respects the hard-coded order in the config
        group_labels = TABULAR_CAT_ATTR_LABELS[groups_desc]
    else:
        group_labels = sorted(set(groups.values()))

    # Gather the tabular data of the patients, and add the group labels
    groups_data = patients.to_dataframe(tabular_attrs=tabular_attrs)
    groups_data[groups_desc] = pd.Series(groups)
    groups_data = groups_data.set_index(groups_desc, append=True)

    # Ignore `matplotlib.category` logger 'INFO' level logs to avoid repeated logs about categorical units parsable
    # as floats
    logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

    # For each attribute, plot the variability of the attribute w.r.t. groups
    for attr in groups_data.columns:
        title = f"{attr}_wrt_{groups_desc}"
        attr_data = groups_data[attr]

        # Based on whether the attribute is categorical or numerical, define different types of plots
        if attr in TabularAttribute.categorical_attrs():
            # Compute the occurrence of each category for each group (including NA)
            attr_stats = attr_data.groupby([groups_desc]).value_counts(normalize=True, dropna=False) * 100
            # After the NA values have been taken into account for the count, drop them
            attr_stats = attr_stats.dropna()

            # For unknown reasons, this plot is unable to pickup variables in the multi-index. As a workaround, we
            # reset the index and to make the index levels into columns available to the plot
            attr_stats = attr_stats.reset_index()

            # For boolean attributes, convert the values to string so that seaborn can properly pick up label names
            # Avoids the following error: 'bool' object has no attribute 'startswith'
            # At the same time, assign relevant labels/hues/etc. for either boolean or categorical attributes
            if attr in TabularAttribute.boolean_attrs():
                attr_stats = attr_stats.astype({attr: str})
                ylabel = "(% true)"
                hue_order = [str(val) for val in TABULAR_CAT_ATTR_LABELS[attr]]
            else:
                ylabel = "(% by label)"
                hue_order = TABULAR_CAT_ATTR_LABELS[attr]

            # Use dodged barplots for categorical attributes
            with sns.axes_style("darkgrid"):
                plot = sns.barplot(
                    data=attr_stats,
                    x=groups_desc,
                    y="proportion",
                    hue=attr,
                    order=group_labels,
                    hue_order=hue_order,
                    estimator="median",
                    errorbar=lambda data: (np.quantile(data, 0.25), np.quantile(data, 0.75)),
                    **cat_plot_kwargs,
                )

            plot.set(title=title, ylabel=ylabel)

        else:  # attr in TabularAttribute.numerical_attrs()
            # Use boxplots for numerical attributes
            with sns.axes_style("darkgrid"):
                # Reset index on the data to make the index levels available as values to plot
                plot = sns.boxplot(
                    data=attr_data.reset_index(), x=groups_desc, y=attr, order=group_labels, **num_plot_kwargs
                )

            plot.set(title=title, ylabel=TABULAR_ATTR_UNITS[attr][0])

        yield title, plot


def plot_time_series_attrs_wrt_group(
    patients: Patients,
    groups: Mapping[Patient.Id, int | str],
    time_series_attrs: Sequence[Tuple[ViewEnum, TimeSeriesAttribute]],
    mask_tag: str = CardinalTag.mask,
    groups_desc: str = "group",
    plot_kwargs: dict = None,
) -> Iterator[Tuple[str, Axes]]:
    """Plots the variability of time-series attributes by aggregating them over groups of patients.

    Args:
        patients: Collection of patients data from which to extract the attributes.
        groups: Groups of patients, represented as a mapping between patient IDs and group labels.
        time_series_attrs: Subset of time-series attributes derived from segmentations (identified by view/attribute
            pairs) for which to plot the average curves across groups of patients.
        mask_tag: Tag of the segmentation mask for which to extract the time-series attributes.
        groups_desc: Description of the groups of patients over which the time-series attributes are aggregated.
        plot_kwargs: Parameters to forward to the call to `seaborn.lineplot` for time-series attributes.

    Returns:
        Iterator over figures (and their corresponding titles) plotting the variability of time-series attributes over
        groups of patients.
    """
    if groups_desc in TabularAttribute.categorical_attrs():
        # If the group corresponds to a categorical attribute, use the labels of the attribute as the group labels
        # so that the order of the labels respects the hard-coded order in the config
        group_labels = TABULAR_CAT_ATTR_LABELS[groups_desc]
    else:
        group_labels = sorted(set(groups.values()))

    # Convert groups from mapping between patient IDs and group labels to lists of patient IDs by group
    groups = {
        group_label: sorted(patient_id for patient_id, patient_group in groups.items() if patient_group == group_label)
        for group_label in group_labels
    }

    # Fetch the data of the patients in each group
    patients_by_group = {
        group_label: [patients[patient_id] for patient_id in patient_ids] for group_label, patient_ids in groups.items()
    }

    # For each time-series attr
    for time_series_attr in time_series_attrs:
        # Extract the data of only the current time-series attribute for each patient
        # In the process, resample the time-series attributes to a common number of points to allow for easy
        # aggregation by seaborn
        attr_view, attr = time_series_attr
        resampling_fn = Interp1d(64)
        time_series_attr_by_group = {
            group_label: {
                patient.id: resampling_fn(patient.get_mask_attributes(mask_tag)[attr_view][attr])
                for patient in patients
            }
            for group_label, patients in patients_by_group.items()
        }

        # Build the dataframe of the time-series for each patient of each group
        time_series_attr_data = build_attributes_dataframe(
            time_series_attr_by_group, outer_name=groups_desc, inner_name="patient"
        )

        # Plot the curves for each group
        with sns.axes_style("darkgrid"):
            plot = sns.lineplot(
                data=time_series_attr_data, x="time", y="val", hue=groups_desc, hue_order=group_labels, **plot_kwargs
            )
        title = f"{'/'.join(time_series_attr)}_wrt_{groups_desc}"
        plot.set(
            title=title, xlabel="(normalized) cardiac cycle phase", ylabel=TIME_SERIES_ATTR_LABELS[time_series_attr[1]]
        )

        yield title, plot


def generate_patients_splits(
    patients: Patients,
    stratify: TabularAttribute,
    bins: int = 5,
    test_size: Union[int, float] = None,
    train_size: Union[int, float] = None,
    seed: int = None,
    progress_bar: bool = False,
) -> Tuple[List[Patient.Id], List[Patient.Id]]:
    """Splits patients into train and test subsets, preserving the distribution of `stratify` variable across subsets.

    Notes:
        - Wrapper around `sklearn.model_selection.train_test_split` that performs binning on continuous variables, since
          out-of-the-box `sklearn`'s `train_test_split` only works with categorical `stratify` variables.

    Args:
        patients: Collection of patients to split.
        stratify: Name of the tabular attribute whose distribution in each of the subset should be similar. Contrary to
            `sklearn.model_selection.train_test_split`, this attribute can be continuous.
        bins: If `stratify` is a continuous attribute, number of bins into which to categorize the values, to ensure
            each bin is distributed representatively in the split.
        test_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the
            test split. If int, represents the absolute number of test samples. If None, the value is set to the
            complement of the train size.
        train_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in
            the train split. If int, represents the absolute number of train samples. If None, the value is
            automatically set to the complement of the test size.
        seed: Seed to control the shuffling applied to the data before applying the split.
        progress_bar: If ``True``, enables progress bars detailing the progress of the collecting data from patients.

    Returns:
        Lists of patients in the train and tests subsets, respectively.
    """
    patients = patients.values()
    msg = "Collecting patients' data"
    if progress_bar:
        patients = tqdm(patients, desc=msg, unit="patient")
    else:
        logger.info(msg + "...")

    # Collect the data of the attribute by which to stratify the split from the patient
    patients_stratify = {patient.id: patient.attrs[stratify] for patient in patients}

    if stratify in TabularAttribute.numerical_attrs():
        # Compute categorical stratify variable from scalar attribute
        stratify_vals = list(patients_stratify.values())
        stratify_bins = np.linspace(min(stratify_vals), max(stratify_vals), num=bins + 1)
        stratify_bins[-1] += 1e-6  # Add epsilon to the last bin's upper bound since it's excluded by `np.digitize`
        stratify_labels = np.digitize(stratify_vals, stratify_bins) - 1  # Subtract 1 because bin indexing starts at 1
    else:
        stratify_labels = list(patients_stratify.values())

    logger.info("Generating splits...")
    patient_ids_train, patient_ids_test = train_test_split(
        list(patients_stratify), test_size=test_size, train_size=train_size, random_state=seed, stratify=stratify_labels
    )
    return sorted(patient_ids_train), sorted(patient_ids_test)
