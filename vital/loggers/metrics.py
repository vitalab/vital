from argparse import ArgumentParser
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from vital.loggers.logger import Logger
from vital.utils.delegate import delegate_inheritance


@delegate_inheritance()
class MetricsLogger(Logger):
    """Abstract class that computes metrics on the results and saves them to csv."""

    Log = Mapping[str, Real]
    data_choices: Sequence[str]  #: Tags of the data on which it is possible to compute the metrics

    def __init__(self, data: str, **kwargs):  # noqa: D205,D212,D415
        """
        Args:
            data: Tag of the data on which to compute metrics.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(output_name=f"{data}_{self.desc}.csv", **kwargs)
        if data not in self.data_choices:
            raise ValueError(
                f"The `data` parameter should be chosen from one of the supported values: {self.data_choices}. "
                f"You passed '{data}' as value for `data`."
            )
        self.data = data

    @classmethod
    def aggregate_logs(cls, logs: Mapping[str, Log], output_path: Path) -> None:
        """Writes the computed metrics, with the aggregated results at the top, to csv format.

        Args:
            logs: Mapping between each result in the iterable results and their metrics' values.
            output_path: Name of the metrics' csv file to be produced as output.
        """
        df_metrics = pd.DataFrame.from_dict(logs, orient="index")

        # Build a dataframe with the aggregated metrics at the top and relevant index names
        aggregated_metrics = cls._aggregate_metrics(df_metrics)
        df_full_metrics = pd.concat([aggregated_metrics, df_metrics])
        df_full_metrics.index.name = cls.IterableResultT.desc

        # Save the combination of aggregated and detailed metrics to the csv file
        pd.DataFrame(df_full_metrics).to_csv(output_path, na_rep="Nan")

    @classmethod
    def _aggregate_metrics(cls, metrics: pd.DataFrame) -> pd.DataFrame:
        """Computes global statistics on the metrics computed over each result.

        Args:
            metrics: Metrics computed over each result.

        Returns:
            Global statistics on the metrics computed over each result.
        """
        return metrics.agg(["mean", "std"])

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser with support for generic metrics and iterable logger arguments.

        Returns:
            Parser object with support for generic metrics and iterable logger arguments.
        """
        parser = super().build_parser()
        parser.add_argument(
            "--data",
            type=str,
            default=cls.data_choices[0],
            choices=cls.data_choices,
            help="Data on which to compute the metrics",
        )
        return parser
