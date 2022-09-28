import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Mapping, Optional, Tuple, Type

import pytorch_lightning as pl
from pathos.multiprocessing import Pool
from pytorch_lightning import Callback
from tqdm import tqdm

from vital.utils.itertools import Collection, Item
from vital.utils.logging import configure_logging

logger = logging.getLogger(__name__)


class ResultsProcessor:
    """Abstract class used for processing inference results, e.g. compute metrics, convert format, etc."""

    ResultsCollection: Type[Collection[Item]]  # Type of the collection of results to process.
    ProcessingOutput: Type = None  # Type of the data returned by processing a single result, if any.
    desc: str  # Description of the processor. Used in progress bar, output filenames, etc.

    def __init__(
        self,
        output_name: str = None,
        progress_bar: bool = True,
        multiprocessing: bool = True,
        **results_collection_kwargs,
    ):
        """Initializes class instance.

        Args:
            output_name: Relative path where to save the results under the output folder. Depending on the whether the
                processor logs output for individual results or aggregates outputs in a file, can be either the name of
                a file or a subdirectory.
            Name for the aggregated output, if the processor produces an aggregated output.
            progress_bar: If ``True``, enables progress bars detailing the progress of the processing.
            multiprocessing: If ``True``, enables multiprocessing when processing results.
            **results_collection_kwargs: Parameters to pass along to the results' collection's ``__init__``. Be careful
                to avoid conflicts with kwargs passed along to the results' collection in `__call__`. In case of
                conflicts with kwargs passed to `__call__`, the `__call__` parameters will override those passed here.

        Raises:
            ValueError: If processor that returns results to be aggregated is not provided with a name to use as part of
                the output path.
        """
        if self.ProcessingOutput is not None and not output_name:
            raise ValueError(
                "When using a processor that returns outputs to be aggregated, you must provide a non-empty "
                "`output_name` so that we can generate the path of the aggregated output."
            )
        self.output_name = output_name
        self.progress_bar = progress_bar
        self.multiprocessing = multiprocessing
        self.results_collection_kwargs = results_collection_kwargs

    def __call__(self, output_folder: Path, **results_collection_kwargs) -> None:
        """Iterates over a set of results and logs/saves the result of the processing to a processor-specific format.

        Args:
            output_folder: Path where to save the output.
            **results_collection_kwargs: Parameters to pass along to the results' collection's ``__init__``. Be careful
                to avoid conflicts with kwargs passed along to the results' collection in `__init__`. In case of
                conflicts with kwargs passed to `__init__`, the parameters passed here will override those passed to
                `__init__`.
        """
        # Resolve the path where to save the results (whether it points to a subdirectory or a file)
        self.output_path = output_folder
        if self.output_name:
            self.output_path /= self.output_name

        # Clean up any leftover outputs from a previous run of the processor targeting the same output directory
        if self.output_path.suffix:
            # If the output is a file, delete the file if it already exists
            # to avoid problems if processors append to an existing file the result of processing each item
            self.output_path.unlink(missing_ok=True)

            # Identify the directory where the file is saved
            output_dir = self.output_path.parent
        else:
            # If the output is a directory, merge the processor's output with the directory's prior contents
            output_dir = self.output_path

        # Ensure the lowest-level output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Merge possibly conflicting options, with warnings for conflicting arguments
        for kwarg, val in results_collection_kwargs.items():
            if kwarg in self.results_collection_kwargs:
                logger.warning(
                    f"In '{self.__class__.__name__}', kwarg '{kwarg}={val}' passed to `__call__' conflicts with the "
                    f"same kwarg passed to `__init__` '{kwarg}={self.results_collection_kwargs[kwarg]}'. The value "
                    f"passed to `__call__` takes precedence."
                )
        results_collection_kwargs = {**self.results_collection_kwargs, **results_collection_kwargs}

        results = self.ResultsCollection(**results_collection_kwargs)
        num_results, results_desc = len(results), results.desc
        if isinstance(results, Mapping):
            results = results.values()

        log_msg = f"Processing results through {self.desc}"
        if self.ProcessingOutput is None:
            # If the processor only writes to logs as a side effect as it iterates over the results
            log_msg += f" and logging to '{self.output_path}'"

        if self.multiprocessing:
            pool = Pool()
            result_processing_jobs = pool.imap(self.process_result, results)
        else:
            result_processing_jobs = (self.process_result(result) for result in results)

        if self.progress_bar:
            result_processing_jobs = tqdm(result_processing_jobs, total=num_results, unit=results_desc, desc=log_msg)
        else:
            logger.info(log_msg)

        if self.ProcessingOutput:
            # If the processor outputs data for each result to be aggregated to a single output
            logger.info(f"Aggregating output processed by {self.desc} to {self.output_path}")
            self.aggregate_outputs(dict(result_processing_jobs), self.output_path)
        else:  # If the processor only writes to logs as a side effect as it iterates over the results
            for _ in result_processing_jobs:
                pass

        if self.multiprocessing:  # Ensure pool resources are freed at the end
            pool.close()
            pool.join()

    def process_result(self, result: Item) -> Optional[Tuple[str, "ProcessingOutput"]]:
        """Processes a single result (either writing to a file or computing an output to aggregate later).

        Args:
            result: Data structure holding all the information relevant to process a single result.

        Returns:
            If not ``None``:
            - Identifier of the result to process.
            - Output of the processed result, to aggregate with outputs processed from other results.
        """
        raise NotImplementedError

    def aggregate_outputs(self, outputs: Mapping[str, "ProcessingOutput"], output_path: Path) -> None:
        """Aggregates the outputs obtained by processing all the results.

        Args:
            outputs: Mapping between each result in the results collection and their output.
            output_path: Path where to write the aggregation of the outputs.
        """
        if self.ProcessingOutput is None:
            raise AssertionError(
                "`aggregate_outputs` should not be called on `ResultsProcessor` instances where "
                "`self.ProcessingOutput` is `None`."
            )
        else:
            raise NotImplementedError

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser with support for generic results processor and collection arguments.

        Returns:
           Parser object with support for generic results processor and collection arguments.
        """
        parser = ArgumentParser()
        parser.add_argument(
            "--output_folder",
            type=Path,
            default=Path.cwd() / "logs",
            help="Path to the directory in which to save the output of the processor",
        )
        parser.add_argument(
            "--disable_progress_bar",
            dest="progress_bar",
            action="store_false",
            help="Disables the progress bars detailing the progress of the processing",
        )
        parser.add_argument(
            "--disable_multiprocessing",
            dest="multiprocessing",
            action="store_false",
            help="Disables parallel processing of results",
        )
        parser = cls.ResultsCollection.add_args(parser)
        return parser

    @classmethod
    def main(cls):
        """Generic main that handles CLI and automatic calling for processors that are executable scripts."""
        configure_logging(log_to_console=True, console_level=logging.INFO)
        parser = cls.build_parser()
        kwargs = vars(parser.parse_args())

        output_folder = kwargs.pop("output_folder")
        cls(**kwargs)(output_folder)


class ResultsProcessorCallback(Callback):
    """Generic wrapper that creates a callback compatible with PL from a pre-configured `ResultsProcessor` instance."""

    def __init__(self, processor: ResultsProcessor):
        self.processor = processor

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:  # noqa: D102
        self.processor(pl_module.log_dir / "results")
