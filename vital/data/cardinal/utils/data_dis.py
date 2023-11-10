import itertools
import logging
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import PairGrid
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from vital.data.cardinal.config import TabularAttribute
from vital.data.cardinal.utils.attributes import TABULAR_ATTR_UNITS
from vital.data.cardinal.utils.data_struct import Patient
from vital.data.cardinal.utils.itertools import Patients

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
