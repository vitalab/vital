import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Mapping, Optional, Tuple, Type

from pathos.multiprocessing import Pool
from tqdm import tqdm

from vital.loggers.utils.itertools import IterableResult, Result

logger = logging.getLogger(__name__)


class Logger:
    """Abstract class used for logging results during the evaluation phase."""

    IterableResultT: Type[IterableResult[Result]]  #: Iterable over which the logs are generated.
    Log: Type = None  #: Type of the data returned by logging a single result, if any.
    desc: str  #: Description of the logger. Used in e.g. progress bar, logs file name, etc.

    def __init__(
        self, output_name: str = None, progress_bar: bool = True, multiprocessing: bool = True, **iterable_result_params
    ):
        """Initializes class instance.

        Args:
            output_name: Name for the aggregated log, if the logger produces an aggregated log.
            progress_bar: If ``True``, enables progress bars detailing the progress of the computations.
            multiprocessing: If ``True``, enables multiprocessing when collecting logs for each result.
            iterable_result_params: Parameters to pass along to result iterator's ``__init__``.

        Raises:
            ValueError: If logger that returns results to be aggregated is not provided with a name to use as part of
                the output path.
        """
        if self.Log is not None and not output_name:
            raise ValueError(
                "When using a logger that returns results to be aggregated, you must provide a non-empty `output_name` "
                "so that we can generate the path of the aggregated results."
            )
        self.output_name = output_name
        self.progress_bar = progress_bar
        self.multiprocessing = multiprocessing
        self.iterable_result_params = iterable_result_params

    def __call__(self, results_path: Path, output_folder: Path, output_prefix: str = None) -> None:
        """Iterates over a set of results and logs the result of the evaluation to a logger-specifc format.

        Args:
            results_path: Root path of the results to log.
            output_folder: Path where to save the logs.
            output_prefix: Prefix to distinguish the logging info, and aggregated saved output if any.
        """
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)

        results = self.IterableResultT(results_path=results_path, **self.iterable_result_params)
        if self.Log is not None:  # If the logger returns data for each result to be aggregated in a single log
            log_desc = f"Collecting {output_prefix} data for {self.desc}"
        else:  # If the logger writes the log as side-effects as it iterates over the results
            log_desc = f"Logging {output_prefix} {self.desc} to {output_folder}"

        if self.multiprocessing:
            pool = Pool()
            log_results_iter = pool.imap(self._log_result, results)
        else:
            log_results_iter = (self._log_result(result) for result in results)

        if self.progress_bar:
            log_results_iter = tqdm(log_results_iter, total=len(results), unit=results.desc, desc=log_desc)
        else:
            logger.info(log_desc)

        if self.Log is not None:  # If the logger returns data for each result to be aggregated in a single log
            logs = dict(log_results_iter)
            output_name = (output_prefix + "_" if output_prefix else "") + self.output_name
            output_path = output_folder / output_name
            logger.info(f"Aggregating {output_prefix} {self.desc} in {output_path}")
            self.aggregate_logs(logs, output_path)
        else:  # If the logger writes the log as side-effects as it iterates over the results
            for _ in log_results_iter:
                pass

        if self.multiprocessing:  # Ensure pool resources are freed at the end
            pool.close()
            pool.join()

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
            assert "`aggregate_logs` should not be called on logger instances where `self.Log` is None."
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
            "--output_folder",
            type=Path,
            default=Path.cwd() / "logs",
            help="Path to the directory in which to save the logs",
        )
        parser.add_argument(
            "--output_prefix", type=str, help="Prefix to distinguish the logging info and aggregated saved output"
        )
        parser.add_argument(
            "--disable_progress_bar",
            dest="progress_bar",
            action="store_false",
            help="Disables the progress bars detailing the progress of the computations",
        )
        parser.add_argument(
            "--disable_multiprocessing",
            dest="multiprocessing",
            action="store_false",
            help="Disables multiprocessing when collecting logs for each result",
        )
        parser = cls.IterableResultT.add_args(parser)
        return parser

    @classmethod
    def main(cls):
        """Generic main that handles CLI and logger calling for use in loggers that could be executable scripts."""
        parser = cls.build_parser()
        kwargs = vars(parser.parse_args())

        results_path = kwargs.pop("results_path")
        output_folder = kwargs.pop("output_folder")
        output_prefix = kwargs.pop("output_prefix")
        cls(**kwargs)(results_path=results_path, output_folder=output_folder, output_prefix=output_prefix)
