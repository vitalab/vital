import itertools
from pathlib import Path

import pandas as pd
import seaborn.objects as so
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from tqdm import tqdm

from vital.data.cardinal.config import CardinalTag, TabularAttribute, TimeSeriesAttribute
from vital.data.cardinal.utils.data_dis import plot_tabular_attrs_wrt_group, plot_time_series_attrs_wrt_group
from vital.data.cardinal.utils.itertools import Patients


def main():
    """Run the script."""
    from argparse import ArgumentParser

    from vital.utils.parsing import yaml_flow_collection

    parser = ArgumentParser()
    groups = parser.add_mutually_exclusive_group(required=True)
    groups.add_argument(
        "--groups_txt",
        nargs="+",
        type=Path,
        help="Path to text files listing the IDs of the patients in each group",
    )
    groups.add_argument(
        "--groups_csv",
        type=Path,
        help="Path to a single CSV file mapping a patient IDs (`patient` column) to group labels (`group` column)",
    )
    parser = Patients.add_args(parser)
    parser.add_argument(
        "--tabular_attrs",
        type=TabularAttribute,
        nargs="*",
        choices=list(TabularAttribute),
        help="Subset of tabular attributes on which to compile the results. If not provided, will default to all "
        "available attributes",
    )
    parser.add_argument(
        "--time_series_attrs",
        type=TimeSeriesAttribute,
        choices=list(TimeSeriesAttribute),
        nargs="*",
        default=list(TimeSeriesAttribute),
        help="Subset of time-series attributes derived from segmentations for which to plot the intra/inter-cluster "
        "variability",
    )
    parser.add_argument(
        "--mask_tag",
        type=str,
        default=CardinalTag.mask,
        help="Tag of the segmentation mask for which to extract the time-series attributes",
    )
    parser.add_argument(
        "--tabular_cat_plot_kwargs",
        type=yaml_flow_collection,
        metavar="{ARG1:VAL1,ARG2:VAL2,...}",
        help="Parameters to forward to the call to `seaborn.barplot` for categorical tabular attributes figures",
    )
    parser.add_argument(
        "--tabular_num_plot_kwargs",
        type=yaml_flow_collection,
        metavar="{ARG1:VAL1,ARG2:VAL2,...}",
        help="Parameters to forward to the call to `seaborn.boxplot` for numerical tabular attributes figures",
    )
    parser.add_argument(
        "--time_series_plot_kwargs",
        type=yaml_flow_collection,
        metavar="{ARG1:VAL1,ARG2:VAL2,...}",
        help="Parameters to forward to the call to `seaborn.lineplot` for time-series attributes figures",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("patient_groups_description"),
        help="Root directory under which to save the figures plotting the variability of the attributes w.r.t. groups",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    (
        groups_txt,
        groups_csv,
        tabular_attrs,
        time_series_attrs,
        mask_tag,
        cat_plot_kwargs,
        num_plot_kwargs,
        time_series_plot_kwargs,
        output_dir,
    ) = list(
        map(
            kwargs.pop,
            [
                "groups_txt",
                "groups_csv",
                "tabular_attrs",
                "time_series_attrs",
                "mask_tag",
                "tabular_cat_plot_kwargs",
                "tabular_num_plot_kwargs",
                "time_series_plot_kwargs",
                "output_dir",
            ],
        )
    )
    time_series_attrs_keys = [
        (view, time_series_attr) for view, time_series_attr in itertools.product(args.views, time_series_attrs)
    ]

    # Load the dataset
    patients = Patients(**kwargs)

    # Load and interpret the clustering instances
    if groups_csv:
        groups = pd.read_csv(groups_csv, index_col=0, dtype={"patient": str, "group": str})["group"].to_dict()
        groups_desc = groups_csv.stem
    else:  # groups_txt
        # Designate the name of the text files as the name of the groups
        # If the names are positive integers, then interpret them as integer IDs of the groups
        groups_txt = {
            # HACK: Use YAML parser to cast dtypes group IDs (e.g. bool, int, etc.)
            yaml_flow_collection(group_file.stem): group_file
            for group_file in groups_txt
        }
        groups_desc = list(groups_txt.values())[0].parent.stem
        # Load the groups from the text files
        groups = {
            patient_id: group_id
            for group_id, group_file in groups_txt.items()
            for patient_id in group_file.read_text().split()
        }

    # Ensure that matplotlib is using 'agg' backend in non-interactive case
    plt.switch_backend("agg")

    # Plot the variability of the tabular and time-series attributes w.r.t. the groups
    tabular_attrs_plots = plot_tabular_attrs_wrt_group(
        patients,
        groups,
        tabular_attrs=tabular_attrs,
        groups_desc=groups_desc,
        cat_plot_kwargs=cat_plot_kwargs,
        num_plot_kwargs=num_plot_kwargs,
    )
    time_series_attrs_plots = plot_time_series_attrs_wrt_group(
        patients,
        groups,
        time_series_attrs_keys,
        mask_tag=mask_tag,
        groups_desc=groups_desc,
        plot_kwargs=time_series_plot_kwargs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)  # Prepare the output folder for the method
    tabular_attrs = tabular_attrs if tabular_attrs is not None else list(TabularAttribute)
    n_plots = len(tabular_attrs) + len(time_series_attrs_keys)
    if groups_desc in tabular_attrs:
        # If the groups correspond to one of the tabular attrs, then no plots will be generated for it
        n_plots -= 1
    for title, plot in tqdm(
        itertools.chain(tabular_attrs_plots, time_series_attrs_plots),
        desc=f"Plotting the variability of the attributes w.r.t. {groups_desc}",
        unit="attr",
        total=n_plots,
    ):
        title_pathified = title.lower().replace("/", "_").replace(" ", "_")
        filepath = output_dir / f"{title_pathified}.svg"

        if isinstance(plot, so.Plot):
            plot.save(filepath, bbox_inches="tight")
        elif isinstance(plot, Axes):
            plt.savefig(filepath)
            plt.close()  # Close the figure to avoid contamination between plots
        else:
            raise ValueError(f"Unable to save the figure for plot type: {type(plot)}.")


if __name__ == "__main__":
    main()
