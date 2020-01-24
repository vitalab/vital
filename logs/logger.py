from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union, List, Type, Any, Tuple, Optional, Dict

from pathos.multiprocessing import Pool
from tqdm import tqdm

from vital.utils.itertools import IterableResult


class Logger:
    """Base class used for logs results during the evaluation phase."""
    name: str
    iterable_result_cls: Type[IterableResult]
    log_type: Type = None

    def __init__(self, output_folder: Union[str, Path],
                 output_name_template: str,
                 **iterable_result_params):
        """
        Args:
            iterable_result_params: parameters to configure the iterable over the results.
            output_folder: path where to save the logs.
        """
        self.iterable_result_params = iterable_result_params

        if isinstance(output_folder, str):
            output_folder = Path(output_folder)
        self.output_folder = output_folder
        output_folder.mkdir(parents=True, exist_ok=True)
        self.output_name_template = str(self.output_folder.joinpath(output_name_template))

    def __call__(self, results_path: Path):
        """ Iterates over a set of results and logs the result of the evaluation to a format defined by the
        implementation of the abstract class.

        Args:
            results_path: path of the HDF5 file containing the results to log.
        """
        results = self.iterable_result_cls(results_path=results_path, **self.iterable_result_params)
        with Pool() as pool:
            logs = dict(tqdm(pool.imap(self._log_result, results),
                             total=len(results), unit=results.unit,
                             desc=f"Logging {results_path.stem} {self.name} to {self.output_folder}"))

        if self.log_type is not None:
            self.write_logs(logs, Path(self.output_name_template.format(results_path.stem)))

    def _log_result(self, result: Any) -> Optional[Tuple[str, "log_type"]]:
        """

        Args:
            result:

        Returns:

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
        """ Creates parser with support for generic logger arguments to a parser.

        Returns:
           parser object for which to add generic arguments.
        """
        parser = ArgumentParser()
        parser.add_argument("--results_path", type=str, required=True,
                            help="Path to a HDF5 file of results to log")
        parser.add_argument("--output_folder", type=str, default="logs",
                            help="Path to the directory in which to save the logs")
        cls.iterable_result_cls.add_args(parser)
        return parser

    @classmethod
    def parse_args(cls, args: Namespace) -> Dict[str, Any]:
        """

        Args:
            args:

        Returns:

        """
        return vars(args)

    @classmethod
    def add_data_selection_args(cls, parser: ArgumentParser, default: str, choices: List[str]):
        """ Adds support for data selection arguments to a parser.

        Args:
           parser: parser object for which to add arguments handling data selection.
        """
        parser.add_argument("--data", type=str, default=default, choices=choices, help="Results data to log.")
