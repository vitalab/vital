import functools
import logging
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from vital.metrics.evaluate.attribute import check_temporal_consistency_errors
from vital.utils.logging import configure_logging
from vital.utils.parsing import StoreDictKeyPair
from vital.utils.signal.regression import kernel_ridge_regression
from vital.utils.signal.snake import DualLagrangianRelaxationSnake, PenaltySnake, Snake
from vital.utils.signal.tests import load_signals_from_json

AttributesStatistics = Dict[str, Tuple[float, float]]


def signal_filtering_test(
    signal_data: Path,
    attr_stats_by_domain: Dict[str, AttributesStatistics],
    filters: Dict[str, Callable[[np.ndarray, AttributesStatistics], np.ndarray]],
    output_prefix: str = None,
) -> None:
    """Processes noisy 1D signals with provided filters, saving plots of the results on disk.

    Args:
        signal_data: Path to various pairs of noisy 1D signals and target references.
        attr_stats_by_domain: Pre-computed sets of statistics to use to normalize the signals, depending on their
            domain.
        filters: Signal processing algorithms to run on the noisy 1D signals.
        output_prefix: Prefix to add to the plots' filenames.
    """
    output_dir = Path.cwd() / "signal_filtering_test_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    signals = load_signals_from_json(signal_data)

    desc = "Processing signal samples" + f" for {output_prefix} test" if output_prefix else ""
    for signal_tag, (signal, target) in tqdm(signals.items(), desc=desc, unit="sample"):
        if len(signal) != len(target):
            raise ValueError(
                "Invalid signal data provided to `snake_test`. It is expected that the signal and its target "
                "should have the same number of data points, but it is not the case for this pair: "
                f"signal: {len(signal)} data points, target: {len(target)} data points."
            )

        # Interpret the signal's tag to point to one specific set of attributes statistics
        if "pred" in signal_tag or "gt" in signal_tag:
            attr_stats = attr_stats_by_domain["gt"]
        elif "ar-vae" in signal_tag:
            attr_stats = attr_stats_by_domain["ar-vae"]
        else:
            raise RuntimeError(
                "Could not interpret the tag of the signal to determine which known set of pre-computed attribute "
                "statistics to use."
            )

        # Process the signal
        data = {"target": target, "signal": signal}
        data.update({filter_tag: filter(signal, attr_stats) for filter_tag, filter in filters.items()})

        # Build the data structure to facilitate plotting the signal processing results
        plot_data = pd.DataFrame.from_dict(data).rename_axis("time").reset_index()
        plot_data = plot_data.melt(
            id_vars=["time"], value_vars=plot_data.columns.difference(["time"]), var_name="data", value_name="val"
        )

        # Plot the signal, reference and processed signal in the same plot
        with sns.axes_style("darkgrid"):
            sns.lineplot(data=plot_data, x="time", y="val", hue="data", hue_order=list(data))
        plt.savefig(output_dir / f"{output_prefix + '_' if output_prefix else ''}{signal_tag}.png")
        plt.close()


if __name__ == "__main__":
    from argparse import ArgumentParser

    configure_logging(log_to_console=True, console_level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    parser = ArgumentParser()
    parser.add_argument(
        "attr_thresholds",
        type=Path,
        help="File containing pre-computed thresholds on the acceptable temporal consistency metrics' values for the "
        "`lv_area` signal",
    )
    parser.add_argument(
        "--attr_stats_by_domain",
        action=StoreDictKeyPair,
        help="Files containing pre-computed sets of statistics to use to normalize the signals, depending on their "
        "domain.",
    )
    args = parser.parse_args()

    # Load statistics on the attributes and their thresholds from the config files
    with open(args.attr_thresholds) as f:
        attr_thresholds = yaml.safe_load(f)
    attr_stats_by_domain = {}
    for domain, attr_stats in args.attr_stats_by_domain.items():
        with open(attr_stats) as f:
            attr_stats_by_domain[domain] = yaml.safe_load(f)

    # Setup a variety of different filter configurations
    krr_poly_filters = {
        "krr_poly gamma=1,degree=6,alpha=0.1": lambda signal, attr_stats: kernel_ridge_regression(
            signal, kernel="poly", gamma=1, alpha=0.1, degree=6, padding=0.1
        ),
        "krr_poly gamma=10,degree=6,alpha=0.1": lambda signal, attr_stats: kernel_ridge_regression(
            signal, kernel="poly", gamma=10, alpha=0.1, degree=6, padding=0.1
        ),
        "krr_poly gamma=10,degree=6,alpha=1e-2": lambda signal, attr_stats: kernel_ridge_regression(
            signal, kernel="poly", gamma=10, alpha=1e-2, degree=6, padding=0.1
        ),
        "krr_poly gamma=10,degree=5,alpha=0.1": lambda signal, attr_stats: kernel_ridge_regression(
            signal, kernel="poly", gamma=10, alpha=0.1, degree=5, padding=0.1
        ),
    }
    krr_rbf_filters = {
        "krr_rbf gamma=1,alpha=0.1": lambda signal, attr_stats: kernel_ridge_regression(
            signal, kernel="rbf", gamma=1, alpha=0.1, padding=0.1
        ),
        "krr_rbf gamma=10,alpha=0.1": lambda signal, attr_stats: kernel_ridge_regression(
            signal, kernel="rbf", gamma=10, alpha=0.1, padding=0.1
        ),
        "krr_rbf gamma=10,alpha=1e-2": lambda signal, attr_stats: kernel_ridge_regression(
            signal, kernel="rbf", gamma=10, alpha=1e-2, padding=0.1
        ),
    }

    snake_filters = {
        "snake smooth=1": lambda signal, attr_stats: Snake(grad_step=1e-2, smoothness_weight=1)(signal),
        "snake smooth=10": lambda signal, attr_stats: Snake(grad_step=1e-2, smoothness_weight=10)(signal),
        "snake smooth=50": lambda signal, attr_stats: Snake(grad_step=1e-2, smoothness_weight=50)(signal),
    }

    def _smoothness_penalty(signal: np.ndarray, attr_stats: AttributesStatistics) -> np.ndarray:
        return check_temporal_consistency_errors(attr_thresholds["lv_area"], signal, bounds=attr_stats["lv_area"])

    constrained_snake_filters = {
        "psnake smooth=0": lambda signal, attr_stats: PenaltySnake(
            grad_step=1e-3, smoothness_constraint_func=functools.partial(_smoothness_penalty, attr_stats=attr_stats)
        )(signal),
        "psnake smooth=10": lambda signal, attr_stats: PenaltySnake(
            grad_step=1e-3,
            smoothness_weight=10,
            smoothness_constraint_func=functools.partial(_smoothness_penalty, attr_stats=attr_stats),
        )(signal),
        "dlrsnake": lambda signal, attr_stats: DualLagrangianRelaxationSnake(
            grad_step=1e-3, smoothness_constraint_func=functools.partial(_smoothness_penalty, attr_stats=attr_stats)
        )(signal),
    }
    filters = {
        "krr_poly": krr_poly_filters["krr_poly gamma=10,degree=6,alpha=0.1"],
        "krr_rbf": krr_rbf_filters["krr_rbf gamma=10,alpha=0.1"],
        "snake": snake_filters["snake smooth=10"],
        "csnake": constrained_snake_filters["dlrsnake"],
    }

    # Produce plots for different combinations of filters
    signal_filtering_test(Path("signal_data.json"), attr_stats_by_domain, krr_poly_filters, output_prefix="krr_poly")
    signal_filtering_test(Path("signal_data.json"), attr_stats_by_domain, krr_rbf_filters, output_prefix="krr_rbf")
    signal_filtering_test(Path("signal_data.json"), attr_stats_by_domain, snake_filters, output_prefix="snake")
    signal_filtering_test(
        Path("signal_data.json"), attr_stats_by_domain, constrained_snake_filters, output_prefix="csnake"
    )
    signal_filtering_test(Path("signal_data.json"), attr_stats_by_domain, filters, output_prefix="all")
