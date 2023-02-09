from argparse import ArgumentParser
from typing import Sequence

import numpy as np

from vital.data.camus.config import CamusTags
from vital.results.camus.utils.data_struct import PatientResult
from vital.results.camus.utils.itertools import Patients
from vital.results.processor import ResultsProcessor
from vital.utils.image.io import sitk_save


class MhdImages(ResultsProcessor):
    """Class that saves the results' data arrays to image format."""

    desc = "mhd_images"
    ResultsCollection = Patients
    input_choices = [
        f"{CamusTags.pred}/{CamusTags.raw}",
        f"{CamusTags.pred}/{CamusTags.post}",
        f"{CamusTags.gt}/{CamusTags.raw}",
    ]  #: Tags of the data that can be saved as MHD images

    def __init__(self, inputs: Sequence[str], **kwargs):
        super().__init__(output_name="mhd", **kwargs)
        if any(input_tag not in self.input_choices for input_tag in inputs):
            raise ValueError(
                f"The `inputs` values should be chosen from one of the supported values: {self.input_choices}. "
                f"You passed '{inputs}' as values for `inputs`."
            )
        self.input_tags = inputs

    def process_result(self, result: PatientResult) -> None:
        """Saves a patient's results as images according to a template directory structure.

        Args:
            result: Data structure holding all the data arrays to save as images for a single patient.
        """
        for view in result.views.values():
            view_folder = self.output_path / view.id
            view_folder.mkdir(parents=True, exist_ok=True)

            for tag in self.input_tags:
                sitk_save(
                    view[tag].data,
                    view_folder / (tag.replace("/", "_") + ".mhd"),
                    spacing=view.voxelspacing[::-1],
                    dtype=np.uint8,
                )

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser for images processor.

        Returns:
            Parser object for images processor.
        """
        parser = super().build_parser()
        parser.add_argument(
            "--inputs",
            type=str,
            nargs="+",
            default=cls.input_choices[:1],
            choices=cls.input_choices,
            help="Tags of the data to save as MHD images",
        )
        return parser


def main():
    """Run the script."""
    MhdImages.main()


if __name__ == "__main__":
    main()
