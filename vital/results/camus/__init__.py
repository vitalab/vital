from abc import ABC
from argparse import ArgumentParser

from vital.data.camus.config import Label


class CamusResultsProcessor(ABC):
    """Class that encapsulates the handling of CAMUS-specific arguments for results processors' CLI."""

    @classmethod
    def add_labels_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds arguments for handling CAMUS-specific arguments to a parser.

        Args:
           parser: Parser object for which to add arguments CAMUS-specific arguments .

        Returns:
            Parser object with support for CAMUS-specific arguments.
        """
        parser.add_argument(
            "--labels",
            type=Label.from_proto_label,
            default=tuple(Label),
            nargs="+",
            choices=tuple(Label),
            help="Labels of the classes included in the segmentations. By default, considers that all class labels "
            "provided in the dataset are included",
        )
        return parser
