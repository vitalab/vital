import logging
from argparse import ArgumentParser

import pandas as pd
from comet_ml.api import APIExperiment
from scripts.comet_grouped_plots import get_workspace_experiment_keys
from vital.utils.logging import configure_logging

logger = logging.getLogger(__name__)


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
    args = parser.parse_args()
    experiment_keys = get_workspace_experiment_keys(
        include_tags=args.include_tag
    )
    print(experiment_keys)

    for experiment_key in experiment_keys:
        # Fetch the current experiment's metadata
        exp = APIExperiment(previous_experiment=experiment_key)
        params = pd.DataFrame(exp.get_parameters_summary())
        columns = params['name'].to_list()
        params = params.T
        params.columns = columns
        params = params.iloc[1]
        heuristic = params['heuristic']
        heuristic = 'MCP' if heuristic.lower() == 'certainty' else heuristic
        if params['human'] == 'true':
            name = f"{'human-' + params['correct_thresh'] if params['human'] == 'true' else ''}"
            human_name = 'human'
        else:
            name = f"{heuristic}" \
                   f"{'-flipped' if params['flip_uncertainties'] == 'true' else ''}" \
                   # f"{'_human_' + params['correct_thresh'] if params['human'] == 'true' else ''}"
            human_name = name

        weights = params['weights'] if params['weights'] != 'None' else params['skip_0_weights']
        weights_name = f"{weights}_{name}"

        print(name)
        print(weights_name)
        print(human_name)

        exp.log_parameter('group_by', name)
        exp.log_parameter('group_by_human', human_name)
        exp.log_parameter('group_by_weights', weights_name)


if __name__ == "__main__":
    main()
