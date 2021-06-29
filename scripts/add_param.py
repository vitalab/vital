import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Sequence, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from comet_ml import get_config
from comet_ml.api import API, APIExperiment
from scripts.comet_grouped_plots import get_workspace_experiment_keys
from tqdm import tqdm

from vital.utils.format.native import filter_excluded
from vital.utils.logging import configure_logging
from vital.utils.parsing import StoreDictKeyPair

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
        print(params)#.set_index('name')
        params.columns = params.T['name'].to_list()
        params = params.iloc[1]

        name = f"{params['weights']}_" \
               f"{params['heuristic']}" \
               f"{'-flipped' if params['flip_uncertainties'] == 'true' else ''}" \
               f"{'_human_' + params['correct_thresh'] if params['human'] == 'true' else ''}"

        print(name)

if __name__ == "__main__":
    main()
