import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scripts.comet_grouped_plots import get_experiments_data, get_workspace_experiment_keys
from vital.utils.logging import configure_logging


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

    metrics = ['nb_edge_pixels', 'test_dice', 'val_dice', 'filtered_edge_error_pixels', 'validated_images',
               'consulted_images']

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

    max_test_dice(experiments_data, args.group_by)
    val_test_dice(experiments_data, args.group_by)
    contour(experiments_data, args.group_by)


def max_test_dice(experiments_data, group_by):
    test_dice = experiments_data.loc[experiments_data.metricName == 'test_dice']
    for group in set(test_dice[group_by]):
        try:
            seed_data = []
            for seed in set(test_dice['seed']):
                dice = np.array(
                    test_dice.loc[(test_dice.seed == seed) & (test_dice[group_by] == group)]['metricValue']).max()
                seed_data.append(dice)
            seed_data = np.array(seed_data)
            print(f"Max Test dice {group}: {seed_data.mean()} +- {seed_data.std()}")
        except Exception as e:
            print(f"Max Test dice {group}: Failed, {e}")


def val_test_dice(experiments_data, group_by):
    test_dice = experiments_data.loc[experiments_data.metricName == 'test_dice']
    val_dice = experiments_data.loc[experiments_data.metricName == 'val_dice']
    for group in set(test_dice[group_by]):
        try:
            seed_data = []
            for seed in set(test_dice['seed']):
                run_test_dice = np.array(
                    test_dice.loc[(test_dice.seed == seed) & (test_dice[group_by] == group)]['metricValue'])
                run_val_dice = np.array(
                    val_dice.loc[(val_dice.seed == seed) & (val_dice[group_by] == group)]['metricValue'])

                best_val_idx = np.argmax(run_val_dice)

                seed_data.append(run_test_dice[best_val_idx])
            seed_data = np.array(seed_data)
            print(f"Max val Test dice {group}: {seed_data.mean()} +- {seed_data.std()}")
        except Exception as e:
            print(f"Max val Test dice {group}: Failed, {e}")


def contour(experiments_data, group_by):
    edge_error = experiments_data.loc[experiments_data.metricName == 'filtered_edge_error_pixels']
    edge = experiments_data.loc[experiments_data.metricName == 'nb_edge_pixels']
    for group in set(edge_error[group_by]):
        try:
            seed_data = []
            for seed in set(edge_error['seed']):
                contour_error = \
                    np.array(edge_error.loc[(edge_error.seed == seed) & (edge_error[group_by] == group)]['metricValue'])[
                        -1]
                total_pixels = np.array(edge.loc[(edge.seed == seed) & (edge[group_by] == group)]['metricValue'])[-1]
                seed_data.append(contour_error / total_pixels)
            seed_data = np.array(seed_data)
            print(f"Contour {group}: {seed_data.mean()} +- {seed_data.std()}")
        except Exception as e:
            print(f"Contour {group}: Failed {e}")


if __name__ == "__main__":
    main()
