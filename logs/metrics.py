from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Union, List

import pandas as pd

from vital.logs.logger import Logger


class MetricsLogger(Logger):
    """Abstract class that computes metrics on the results and saves them to csv."""
    log_type = Dict[str, Union[int, float]]
    data_choices: List[str]

    def __init__(self, data: str, **kwargs):
        """
        Args:
            data: name to the data on which to compute metrics.
        """
        super().__init__(**kwargs, output_name_template=f'{{}}_{data}_{self.name}.csv')
        self.data = data

    @classmethod
    def write_logs(cls, logs: Dict[str, Dict[str, Union[int, float]]], output_name: Path):
        """ Writes the computed metrics, with the aggregated results at the top, to csv format.

        Args:
            logs: mapping between each result in the iterable results and their metrics' values.
            output_name: the name of the metrics' csv file to be produced as output.
        """
        df_metrics = pd.DataFrame.from_dict(logs, orient='index')

        # Build a dataframe with the aggregated metrics at the top and relevant index names
        aggregated_metrics = cls._aggregate_metrics(df_metrics)
        df_full_metrics = pd.concat([aggregated_metrics, df_metrics])
        df_full_metrics.index.name = cls.iterable_result_cls.index_name

        # Save the combination of aggregated and detailed metrics to the csv file
        pd.DataFrame(df_full_metrics).to_csv(output_name, na_rep='Nan')

    @classmethod
    def _aggregate_metrics(cls, metrics: pd.DataFrame) -> pd.DataFrame:
        """

        Args:
            metrics:

        Returns:

        """
        return metrics.agg(['mean', 'std'])

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """ Adds support for generic metrics logger arguments to a parser.

        Args:
           parser: parser object for which to add generic metrics arguments.
        """
        parser = super().build_parser()
        super().add_data_selection_args(parser, default=cls.data_choices[0], choices=cls.data_choices)
        return parser
