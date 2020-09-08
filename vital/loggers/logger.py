from argparse import ArgumentParser
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Type

from pathos.multiprocessing import Pool
from tqdm import tqdm

from vital.loggers.utils.itertools import IterableResult, Result


class Logger:
    """Abstract class used for logging results during the evaluation phase."""

    IterableResultT: Type[IterableResult[Result]]  #: Iterable over which the logs are generated.
    Log: Type = None  #: Type of the data returned by logging a single result, if any.
    desc: str  #: Description of the logger. Used in e.g. progress bar, logs file name, etc.

    def __init__(
        self, output_name_template: str = None, debug: bool = False, **iterable_result_params
    ):  # noqa: D205,D212,D415
        """
        Args:
            output_name_template: Name template for the aggregated log, if the logger produces an aggregated log.
            debug: If ``True``, disables multiprocessing when collecting logs for each result; otherwise, enables
                multiprocessing.
            iterable_result_params: Parameters to configure the iterable over the results. Can be ``None`` if the logger
                will only be used to write logs (and not called).
        """
        self.output_name_template = output_name_template
        self.debug = debug
        self.iterable_result_params = iterable_result_params

    def __call__(self, results_path: Path, output_folder: Path) -> None:
        """Iterates over a set of results and logs the result of the evaluation to a logger-specifc format.

        Args:
            results_path: Root path of the results to log.
            output_folder: Path where to save the logs.
        """
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)

        results = self.IterableResultT(results_path=results_path, **self.iterable_result_params)

        if self.Log is not None:  # If the logger returns data for each result to be aggregated in a single log
            if self.debug:
                logs = dict(
                    self._log_result(result)
                    for result in tqdm(
                        results, unit=results.desc, desc=f"Collecting {results_path.stem} data for {self.desc}"
                    )
                )
            else:
                with Pool() as pool:
                    logs = dict(
                        tqdm(
                            pool.imap(self._log_result, results),
                            total=len(results),
                            unit=results.desc,
                            desc=f"Collecting {results_path.stem} data for {self.desc}",
                        )
                    )
            output_path = output_folder.joinpath(self.output_name_template.format(results_path.stem))
            tqdm.write(f"Aggregating {results_path.stem} {self.desc} in {output_path} ...")
            self.aggregate_logs(logs, output_path)
        else:  # If the logger writes the log as side-effects as it iterates over the results
            if self.debug:
                for result in tqdm(
                    results, unit=results.desc, desc=f"Logging {results_path.stem} {self.desc} to {output_folder}"
                ):
                    self._log_result(result)
            else:
                with Pool() as pool:
                    list(
                        tqdm(
                            pool.imap(self._log_result, results),
                            total=len(results),
                            unit=results.desc,
                            desc=f"Logging {results_path.stem} {self.desc} to {output_folder}",
                        )
                    )

    def _log_result(self, result: Result) -> Optional[Tuple[str, "Log"]]:
        """Generates a log (either writing to a file or computing a result to aggregate) for a single result.

        Args:
            result: Data structure holding all the information relevant to generate a log for a single result.

        Returns:
            If not ``None``:
            - Identifier of the result for which the logs where generated.
            - Generated log for the result, to aggregate with logs generated for other results.
        """
        raise NotImplementedError

    def aggregate_logs(self, logs: Mapping[str, "Log"], output_path: Path) -> None:
        """Collects the logs aggregated from all the results, and performs operations on the aggregated logs.

        Args:
            logs: Mapping between each result in the iterable results and their log.
            output_path: Path where to write the results of the operations on the aggregated logs.
        """
        if self.Log is None:
            pass
        else:
            raise NotImplementedError

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser with support for generic logger and iterable arguments.

        Returns:
           Parser object with support for generic logger and iterable arguments.
        """
        parser = ArgumentParser()
        parser.add_argument("--results_path", type=Path, required=True, help="Path to a HDF5 file of results to log")
        parser.add_argument(
            "--output_folder", type=Path, default="logs", help="Path to the directory in which to save the logs"
        )
        parser.add_argument(
            "--debug", action="store_true", help="Run logger in debug mode, which disables multiprocessing"
        )
        parser = cls.IterableResultT.add_args(parser)
        return parser

    @classmethod
    def add_data_selection_args(cls, parser: ArgumentParser, choices: Sequence[str]) -> ArgumentParser:
        """Adds data selection arguments to a parser.

        Args:
           parser: Parser object for which to add arguments handling data selection.
           choices: Tags of the data the logger can be called on. The first tag in the list is the default choice.

        Returns:
            Parser object with support for arguments handling data selection.
        """
        parser.add_argument("--data", type=str, default=choices[0], choices=choices, help="Results data to log.")
        return parser

    @classmethod
    def main(cls):
        """Generic main that handles CLI and logger calling for use in loggers that could be executable scripts."""
        parser = cls.build_parser()
        kwargs = vars(parser.parse_args())

        results_path = kwargs.pop("results_path")
        output_folder = kwargs.pop("output_folder")
        logger = cls(**kwargs)
        logger(results_path=results_path, output_folder=output_folder)
