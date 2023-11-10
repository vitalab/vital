import logging
from argparse import ArgumentParser
from typing import Literal, Sequence, Tuple

import numpy as np

from vital.data.cardinal.config import CardinalTag, Label
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.utils.attributes import compute_mask_tabular_attributes
from vital.data.cardinal.utils.data_struct import Patient
from vital.data.cardinal.utils.itertools import Patients
from vital.results.metrics import Metrics
from vital.utils.image.us.measure import EchoMeasure

logger = logging.getLogger(__name__)


class ImageTabularAttributes(Metrics):
    """Class that computes tabular attributes from the images and saves them, either aggregated or per patient."""

    desc = "image_tabular_attributes"
    ResultsCollection = Patients
    input_choices = []

    def __init__(self, save_by_patient: bool = True, subdir_levels: Sequence[Literal["patient"]] = None, **kwargs):
        """Initializes class instance.

        Args:
            save_by_patient: Whether to also save the image tabular attributes for each patient to patient-specific YAML
                files.
            subdir_levels: Levels of subdirectories to create under the root output folder.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)
        self.save_by_patient = save_by_patient
        self.patient_save_kwargs = {"subdir_levels": subdir_levels}

    def process_result(self, result: Patient) -> Tuple[str, "Metrics.ProcessingOutput"]:
        """Computes image tabular attributes on data from a patient.

        Args:
            result: Data structure holding all the relevant information to compute the requested attributes for a single
                patient.

        Returns:
            - Identifier of the result for which the attributes where computed.
            - Mapping between the attributes and their value for the patient.
        """
        a4c_view = result.views[ViewEnum.A4C]
        a2c_view = result.views[ViewEnum.A2C]

        a4c_es_frame, a2c_es_frame = None, None
        if self.target_tag:
            # Identify the ES frame based on the reference segmentation mask, not the prediction to evaluate
            a4c_es_frame = np.argmin(EchoMeasure.structure_area(np.isin(a4c_view.data[self.target_tag], Label.LV)))
            a2c_es_frame = np.argmin(EchoMeasure.structure_area(np.isin(a2c_view.data[self.target_tag], Label.LV)))

        tab_attrs = compute_mask_tabular_attributes(
            a4c_view.data[self.input_tag],
            a4c_view.attrs[self.input_tag][CardinalTag.voxelspacing],
            a2c_view.data[self.input_tag],
            a2c_view.attrs[self.input_tag][CardinalTag.voxelspacing],
            a4c_es_frame=a4c_es_frame,
            a2c_es_frame=a2c_es_frame,
        )

        if self.save_by_patient:
            result.attrs.update(tab_attrs)
            result.save(
                self.output_path.with_suffix(""),
                save_tabular_attrs=True,
                include_tags=[],
                **self.patient_save_kwargs,
            )

        return result.id, tab_attrs

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser for image tabular attributes.

        Returns:
            Parser object for image tabular attributes.
        """
        parser = super().build_parser()
        parser.add_argument(
            "--disable_save_by_patient",
            dest="save_by_patient",
            action="store_false",
            help="Disable saving the image tabular attributes for each patient to patient-specific YAML files",
        )
        parser.add_argument(
            "--subdir_levels",
            type=str,
            nargs="+",
            choices=["patient"],
            help="Levels of subdirectories to create under the root output folder",
        )
        return parser


def main():
    """Run the script."""
    ImageTabularAttributes.main()


if __name__ == "__main__":
    main()
