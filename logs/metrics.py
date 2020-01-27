from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Union, List

import pandas as pd

from vital.logs.logger import Logger


class MetricsLogger(Logger):
    """Abstract class that computes metrics on the results and saves them to csv."""
    log_type = Dict[str, Union[int, float]]
    data_choices: List[str]  # tags of the data on which to compute metrics.

    def __init__(self, data: str, **iterable_result_params):
        """
        Args:
            data: tag of the data on which to compute metrics.
            iterable_result_params: parameters to configure the iterable over the results.
        """
        super().__init__(output_name_template=f'{{}}_{data}_{self.name}.csv', **iterable_result_params)
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
        df_full_metrics.index.name = cls.iterable_result_cls.desc

        # Save the combination of aggregated and detailed metrics to the csv file
        pd.DataFrame(df_full_metrics).to_csv(output_name, na_rep='Nan')

    @classmethod
    def _aggregate_metrics(cls, metrics: pd.DataFrame) -> pd.DataFrame:
        """ Computes global statistics on the metrics computed over each result.

        Args:
            metrics: metrics computed over each result.

        Returns:
            global statistics on the metrics computed over each result.
        """
        return metrics.agg(['mean', 'std'])

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """ Creates parser with support for generic metrics and iterable logger arguments.

        Returns:
          parser object with support for generic metrics and iterable logger arguments.
        """
        parser = super().build_parser()
        super().add_data_selection_args(parser, choices=cls.data_choices)
        return parser
