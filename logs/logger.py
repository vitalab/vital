from argparse import ArgumentParser
from pathlib import Path
from typing import List, Type, Any, Tuple, Optional, Dict

from pathos.multiprocessing import Pool
from tqdm import tqdm

from vital.utils.itertools import IterableResult


class Logger:
    """Abstract class used for logging results during the evaluation phase."""
    name: str  # Description of the logger. Used in e.g. progress bar, logs file name, etc.
    iterable_result_cls: Type[IterableResult]  # Iterable over which the logs are generated.
    log_type: Type = None  # Type of the data returned by logging a single result, if any.

    def __init__(self, output_name_template: str = None,
                 **iterable_result_params):
        """
        Args:
            output_name_template: name template for the aggregated log, if the logger produces an aggregated log.
            iterable_result_params: parameters to configure the iterable over the results.
        """
        self.iterable_result_params = iterable_result_params
        self.output_name_template = output_name_template

    def __call__(self, results_path: Path, output_folder: Path):
        """ Iterates over a set of results and logs the result of the evaluation to a format defined by the
        implementation of the abstract class.

        Args:
            results_path: root path of the results to log.
            output_folder: path where to save the logs.
        """
        output_folder.mkdir(parents=True, exist_ok=True)

        results = self.iterable_result_cls(results_path=results_path, **self.iterable_result_params)
        with Pool() as pool:
            logs = dict(tqdm(pool.imap(self._log_result, results),
                             total=len(results), unit=results.desc,
                             desc=f"Logging {results_path.stem} {self.name} to {output_folder}"))

        # If the logger returns data, for each result, to be aggregated in a single log
        if self.log_type is not None:
            self.write_logs(logs, output_folder.joinpath(self.output_name_template.format(results_path.stem)))

    def _log_result(self, result: Any) -> Optional[Tuple[str, "log_type"]]:
        """ Generates a log (either writing to a file or computing a result to aggregate) for a single result.

        Args:
            result: data structure holding all the information relevant to generate a log for a single result.

        Returns:
            - identifier of the result for which the logs where generated.
            - generated log for the result, to aggregate with logs generated for other results.
            or None
        """
        raise NotImplementedError

    @classmethod
    def write_logs(cls, logs: Dict[str, "log_type"], output_name: Path):
        """ Writes the logs aggregated from all the results, with the aggregated results at the top.

        Args:
            logs: mapping between each result in the iterable results and their log.
            output_name: path where to write the aggregated log file.
        """
        if cls.log_type is None:
            pass
        else:
            raise NotImplementedError

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """ Creates parser with support for generic logger and iterable arguments.

        Returns:
           parser object with support for generic logger and iterable arguments.
        """
        parser = ArgumentParser()
        parser.add_argument("--results_path", type=Path, required=True,
                            help="Path to a HDF5 file of results to log")
        parser.add_argument("--output_folder", type=Path, default="logs",
                            help="Path to the directory in which to save the logs")
        cls.iterable_result_cls.add_args(parser)
        return parser

    @classmethod
    def add_data_selection_args(cls, parser: ArgumentParser, choices: List[str]):
        """ Adds data selection arguments to a parser.

        Args:
           parser: parser object for which to add arguments handling data selection.
           choices: tags of the data the logger can be called on. The first tag in the list is the default choice.
        """
        parser.add_argument("--data", type=str, default=choices[0], choices=choices, help="Results data to log.")
