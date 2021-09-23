import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scripts.comet_grouped_plots import get_experiments_data, get_workspace_experiment_keys
from vital.utils.logging import configure_logging
import pandas as pd


def main():
    """Run the script."""
    configure_logging(log_to_console=True, console_level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_key",
        type=str,
        nargs="+",
        help="Key of the experiment to plot. If no experiment is provided, defaults to plotting for all the "
             "experiments in the workspace. NOTE: Manually specifying the experiment keys disables all other filters "
             "(e.g. `--include_tag`, `--exclude_experiment`, etc.)",
    )
    parser.add_argument(
        "--include_tag",
        type=str,
        nargs="+",
        help="Tag that experiments should have to be included in the plots",
    )
    parser.add_argument(
        "--exclude_tag",
        type=str,
        nargs="+",
        help="Tag that experiments should NOT have to be included in the plots",
    )
    parser.add_argument(
        "--exclude_experiment",
        type=str,
        nargs="+",
        help="Key of the experiment to plot, or path to a file listing key of experiments to exclude from the plots",
    )

    parser.add_argument("--group_by", type=str, help="Hyperparameter by which to group experiments", required=True)
    # parser.add_argument("--out_dir", type=Path, help="Output directory where to save the figures", required=True)
    args = parser.parse_args()

    metrics = ['test_dice', 'val_dice',
               'nb_edge_pixels', 'filtered_edge_error_pixels',
               'total_pixels', 'annotated_pixels',
               'consulted_images', 'corrected_images', 'validated_images']

    # Cast to path excluded experiments arguments that are valid file paths
    excluded_experiments = (
        [
            Path(exclude_item) if os.path.isfile(exclude_item) else exclude_item
            for exclude_item in args.exclude_experiment
        ]
        if args.exclude_experiment
        else None
    )

    # Determine the experiments to include in the plots
    experiment_keys = args.experiment_key
    if not experiment_keys:
        experiment_keys = get_workspace_experiment_keys(
            include_tags=args.include_tag, exclude_tags=args.exclude_tag, to_exclude=excluded_experiments
        )

    if not (num_experiments := len(experiment_keys)) > 1:
        raise ValueError(
            f"Cannot generate plots for only one experiment. Please provide at least "
            f"{2 - num_experiments} other experiment(s) for which to plot curves."
        )

    # args.out_dir.mkdir(parents=True, exist_ok=True)

    experiments_data = get_experiments_data(experiment_keys, metrics)
    experiments_data.to_csv('metrics.csv')

    #
    # experiments_data = pd.read_csv('metrics.csv')
    # print(experiments_data)

    def pop_std(x):
        return x.std(ddof=0)

    experiments_data = experiments_data.groupby([args.group_by, 'step', 'metricName'], as_index=False)
    experiments_data = experiments_data.agg({'metricValue': ['mean', 'std'],
                                             args.group_by: 'first',
                                             'step': 'first',
                                             'metricName': 'first'})
    print(experiments_data)

    experiments_data = experiments_data.reset_index(drop=True)

    experiments_data.columns = ['metricValue', 'std', args.group_by, 'step', 'MetricName']

    print(experiments_data)

    exps = set(experiments_data[args.group_by])
    print(exps)

    res = {}

    for exp in exps:
        exp_data = experiments_data[experiments_data[args.group_by] == exp]

        val_dice = np.array(exp_data[(exp_data['MetricName'] == 'val_dice')]['metricValue'])
        test_dice = np.array(exp_data[exp_data['MetricName'] == 'test_dice']['metricValue'])

        edge_error_pixels = np.array(exp_data[exp_data['MetricName'] == 'filtered_edge_error_pixels']['metricValue'])
        nb_edge_pixels = np.array(exp_data[exp_data['MetricName'] == 'nb_edge_pixels']['metricValue'])

        total_pixels = np.array(exp_data[exp_data['MetricName'] == 'total_pixels']['metricValue'])
        annotated_pixels = np.array(exp_data[exp_data['MetricName'] == 'annotated_pixels']['metricValue'])

        validated_images = np.array(exp_data[exp_data['MetricName'] == 'validated_images']['metricValue'])
        consulted_images = np.array(exp_data[exp_data['MetricName'] == 'consulted_images']['metricValue'])
        corrected_images = np.array(exp_data[exp_data['MetricName'] == 'corrected_images']['metricValue'])

        dice = test_dice[np.argmax(val_dice)]
        contour_error = edge_error_pixels[-1] / nb_edge_pixels[-1]
        consulted = consulted_images[-1] / validated_images[-1]
        pixel_error = annotated_pixels[-1] / total_pixels[-1]

        initial_dice = np.array(exp_data[exp_data['MetricName'] == 'test_dice']['metricValue'])[0]
        initial_dice_std = np.array(exp_data[exp_data['MetricName'] == 'test_dice']['std'])[0]

        print(f"{exp} initial dice {initial_dice:.4f} \pm {initial_dice_std}")

        exp_res = {'test_dice': dice,
                   'contour_error': contour_error,
                   'annotated_pixels': pixel_error,
                   'consulted': consulted}

        res[exp] = exp_res

    res = pd.DataFrame(res).T
    # res = res.reindex(["bald", "MCP", "bald-flipped", "MCP-flipped", "random", "human-0.15", "human-0.3", "human-0.45"])
    res = res.reindex(["bald", "MCP", "bald-flipped", "MCP-flipped", "random", "human"])

    res.index.name = 'Method'

    def bold_extreme_values(data, data_bold=-1):
        if data == data_bold:
            return "\\textbf{%.4f}" % data
        else:
            return f"{data:.4f}"

    exp_bold = {'test_dice': 'max',
                'contour_error': 'min',
                'annotated_pixels': 'min',
                'consulted': 'min'}

    for col in res.columns.get_level_values(0).unique():
        data_bold = res[col].max() if exp_bold[col] == 'max' else res[col].min()
        res[col] = res[col].apply(lambda data: bold_extreme_values(data, data_bold=data_bold))

    print(res)
    print(res.to_latex(escape=False))


if __name__ == "__main__":
    main()
