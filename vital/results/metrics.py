from argparse import ArgumentParser
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from vital.results.processor import ResultsProcessor


class Metrics(ResultsProcessor):
    """Abstract class that computes metrics on the results and saves them to csv."""

    ProcessingOutput = Mapping[str, Real]
    input_choices: Sequence[str] = None  #: Tags of the data on which it is possible to compute the metrics
    target_choices: Sequence[str] = None  #: Tags of reference data that can serve as target when computing the metrics

    def __init__(self, input: str, target: str = None, **kwargs):
        """Initializes class instance.

        Args:
            input: Tag of the data for which to compute metrics.
            target: Tag of the (optional) reference data to use as target when computing metrics.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(output_name=f"{input.replace('/', '-')}_{self.desc}.csv", **kwargs)
        if self.input_choices and input not in self.input_choices:
            raise ValueError(
                f"The `input` parameter should be chosen from one of the supported values: {self.input_choices}. "
                f"You passed '{input}' as value for `input`."
            )
        if self.target_choices and target not in self.target_choices:
            raise ValueError(
                f"The `target` parameter should be chosen from one of the supported values: {self.target_choices}. "
                f"You passed '{target}' as value for `target`."
            )

        self.input_tag = input
        self.target_tag = target

    def aggregate_outputs(self, outputs: Mapping[str, ProcessingOutput], output_path: Path) -> None:
        """Writes the computed metrics, with the aggregated results at the top, to csv format.

        Args:
            outputs: Mapping between each result in the results collection and their metrics' values.
            output_path: Name of the metrics' csv file to be produced as output.
        """
        df_metrics = pd.DataFrame.from_dict(outputs, orient="index")

        # Build a dataframe with the aggregated metrics at the top and relevant index names
        aggregated_metrics = self._aggregate_metrics(df_metrics)
        df_full_metrics = pd.concat([aggregated_metrics, df_metrics])
        df_full_metrics.index.name = self.ResultsCollection.desc

        # Save the combination of aggregated and detailed metrics to the csv file
        pd.DataFrame(df_full_metrics).to_csv(output_path, na_rep="Nan")

    def _aggregate_metrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Computes global statistics for the metrics computed over each result.

        Args:
            metrics: Metrics computed over each result.

        Returns:
            Global statistics for the metrics computed over each result.
        """
        return metrics.agg(["mean", "std"])

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser with support for generic metrics and collection arguments.

        Returns:
            Parser object with support for generic metrics and collection arguments.
        """
        parser = super().build_parser()
        input_kwargs = {}
        if cls.input_choices:
            input_kwargs.update({"default": cls.input_choices[0], "choices": cls.input_choices})
        parser.add_argument("--input", type=str, **input_kwargs, help="Data for which to compute the metrics")
        if cls.target_choices is not None:
            target_kwargs = {}
            if cls.target_choices:
                target_kwargs.update({"default": cls.target_choices[0], "choices": cls.target_choices})
            parser.add_argument(
                "--target", type=str, **target_kwargs, help="Reference data to use as target when computing metrics"
            )
        return parser
