import argparse
import os
from functools import reduce
from numbers import Real
from operator import add
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Sequence, Tuple

import h5py
import numpy as np
from PIL.Image import LINEAR
from tqdm import tqdm

from vital.data.camus.config import CamusTags, Instant, Label, View, image_size, img_save_options, seg_save_options
from vital.data.config import Subset
from vital.utils.image.io import load_mhd
from vital.utils.image.register.camus import CamusRegisteringTransformer
from vital.utils.image.transform import remove_labels, resize_image


class CrossValidationDatasetGenerator:
    """Utility to process the raw CAMUS data (as it is when downloaded) and to generate a cross validation HDF5 file.

    The cross validation split organizes the data in the following way:
        - split data into train, validation and test sets for a selection of folds
        - save the list of patient for each set, for each fold, as metadata in the HDF5 file
        - only save a single copy of the data for each patient in the HDF5

    The processing on the data performs the following task, in order:
        - keep only a subset of the labels from the groundtruth (optional)
        - register images (optional)
        - resize images to a given size
    """

    _subset_names_in_data: Dict[Subset, Literal["training", "validation", "testing"]] = {
        Subset.TRAIN: "training",
        Subset.VALID: "validation",
        Subset.TEST: "testing",
    }

    def __call__(
        self,
        data: Path,
        output: Path,
        folds: Sequence[int] = range(1, 11),
        target_image_size: Tuple[int, int] = (image_size, image_size),
        sequence: bool = False,
        register: bool = False,
        labels: Sequence[Label] = None,
    ) -> None:
        """Organizes the raw CAMUS image data in a single HDF5 file, with the metadata of the lists of patients to use
        for cross-validation experiments.

        Args:
            data: path to the CAMUS root directory, under which the patient directories are stored.
            output: path to the HDF5 file to generate, containing all the raw image data and cross-validation metadata.
            folds: IDs of the folds for which to include metadata in the generated HDF5 file.
            sequence: whether to augment the dataset by adding the data for the full sequence between ED and ES.
            register: enable/disable registering.
            labels: labels of the segmentation classes to include.
        """
        # Save parameters useful in downstream functions inside the object
        # This is done to avoid overly long function signatures in low-level functions
        self.data = data
        self.flags = {f"{CamusTags.full_sequence}": sequence, f"{CamusTags.registered}": register}
        self.labels_to_remove = [] if labels is None else [label for label in Label if label not in labels]
        self.registering_transformer = CamusRegisteringTransformer(
            num_classes=Label.count(), crop_shape=target_image_size
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output, "w") as dataset:
            # Write which option flags were activated when generating the HDF5 file
            for flag, value in self.flags.items():
                dataset.attrs[flag] = value

            # Write the metadata for each cross-validation fold
            cross_validation_group = dataset.create_group("cross_validation")
            for fold in folds:
                fold_group = cross_validation_group.create_group(f"fold_{fold}")
                for subset, subset_name_in_data in self._subset_names_in_data.items():
                    fold_subset_patients = self.get_fold_subset_from_file(data, fold, subset_name_in_data)
                    fold_group.create_dataset(subset.value, data=np.array(fold_subset_patients, dtype="S"))

            # Get a list of all the patients in the dataset
            patient_ids = reduce(
                add,
                [self.get_fold_subset_from_file(data, fold, subset) for subset in self._subset_names_in_data.values()],
            )
            patient_ids.sort()

            # Write the raw image data for each patient in the dataset
            for patient_id in tqdm(patient_ids, unit="patient", desc=f"Writing patient data to HDF5 file: {output}"):
                self._write_patient_data(dataset.create_group(patient_id))

    @classmethod
    def get_fold_subset_from_file(
        cls, data: Path, fold: int, subset: Literal["training", "validation", "testing"]
    ) -> List[str]:
        """Reads patient ids for a subset of a cross-validation configuration.

        Args:
            data: path to the CAMUS root directory, under which the patient directories are stored.
            fold: the ID of the test set for the cross-validation configuration.
            subset: name of the subset for which to fetch patient IDs for the cross-validation configuration.

        Returns:
            IDs of the patients that are included in the subset of the fold.
        """
        list_fn = data.joinpath("listSubGroups", f"subGroup{fold}_{subset}.txt")
        # Open text file containing patient ids (one patient id by row)
        with open(str(list_fn), "r") as f:
            patient_ids = [line for line in f.read().splitlines()]
        return patient_ids

    def _write_patient_data(self, patient_group: h5py.Group) -> None:
        """Writes the raw image data of a patient to a designated HDF5 group within the HDF5 file.

        Args:
            patient_group: HDF5 patient group for which to fetch and save the data.
        """
        patient_id = os.path.basename(patient_group.name)

        for view in View:
            # The order of the instants within a view dataset is chronological: ED -> ES -> ED
            data_x, data_y, info_view, instants_with_gt = self._get_view_data(patient_id, view)

            data_y = remove_labels(data_y, [lbl.value for lbl in self.labels_to_remove], fill_label=Label.BG.value)

            if self.flags[CamusTags.registered]:
                registering_parameters, data_y_proc, data_x_proc = self.registering_transformer.register_batch(
                    data_y, data_x
                )
            else:
                data_x_proc = np.array([resize_image(x, (image_size, image_size), resample=LINEAR) for x in data_x])
                data_y_proc = np.array([resize_image(y, (image_size, image_size)) for y in data_y])

            # Write image and groundtruth data
            patient_view_group = patient_group.create_group(view.value)
            patient_view_group.create_dataset(
                name=CamusTags.img_proc, data=data_x_proc[..., np.newaxis], **img_save_options
            )
            patient_view_group.create_dataset(name=CamusTags.gt, data=data_y, **seg_save_options)
            patient_view_group.create_dataset(name=CamusTags.gt_proc, data=data_y_proc, **seg_save_options)

            # Write metadata useful for providing instants or full sequences
            patient_view_group.attrs[CamusTags.info] = info_view
            for instant, instant_idx in instants_with_gt.items():
                patient_view_group.attrs[instant.value] = instant_idx

            # Write metadata concerning the registering applied
            if self.flags[CamusTags.registered]:
                for registering_step, values in registering_parameters.items():
                    patient_view_group.attrs[registering_step] = values

    def _get_view_data(
        self, patient_id: str, view: View
    ) -> Tuple[np.ndarray, np.ndarray, List[Real], Dict[Instant, int]]:
        """Fetches the data for a specific view of a patient.

        If ``self.use_sequence`` is True, augments the dataset with sequence between the ED and ES instants.
        Otherwise, returns the view data as is.

        Args:
            patient_id: patient id formatted to match the identifiers in the mhd files' names.
            view: the view for which to fetch the patient's data.

        Returns:
            - sequence of ultrasound images acquired between ED and ES. The trend is that the first images are closer
              to ED, and the last images are closer to ES.
            - segmentations interpolated between ED and ES. The trend is that the first segmentations are closer to ED,
              and the last segmentations are closer to ES.
            - metadata concerning the sequence.
            - mapping between the instants with manually validated groundtruths and the index where they appear in the
              sequence.
        """
        view_info_fn = self.data.joinpath(patient_id, f"Info_{view.value}.cfg")

        # Determine the index of segmented instants in sequence
        instants_with_gt = {}
        with open(str(view_info_fn), "r") as view_info_file:
            view_info_lines = view_info_file.read().splitlines()
            for instant_idx, instant in enumerate(Instant):
                instants_with_gt[instant] = int(view_info_lines[instant_idx].split()[-1]) - 1

        # Get data for the whole sequence ranging from ED to ES
        sequence, sequence_gt, info = self._get_sequence_data(patient_id, view, instants_with_gt)

        if instants_with_gt[Instant.ED] > instants_with_gt[Instant.ES]:  # Ensure ED comes before ES (swap when ES->ED)
            instants_with_gt[Instant.ED], instants_with_gt[Instant.ES] = (
                instants_with_gt[Instant.ES],
                instants_with_gt[Instant.ED],
            )

        # Include all or only some instants from the input and reference data according to the parameters
        data_x, data_y = [], []
        if self.flags[CamusTags.full_sequence]:
            data_x, data_y = sequence, sequence_gt
        else:
            for instant in Instant:
                data_x.append(sequence[instants_with_gt[instant]])
                data_y.append(sequence_gt[instants_with_gt[instant]])

            # Update indices of instants with annotated segmentations in view sequences in newly sliced sequences
            instants_with_gt = {instant_key: idx for idx, instant_key in enumerate(Instant)}

        # Add channel dimension
        return np.array(data_x), np.array(data_y), info, instants_with_gt

    def _get_sequence_data(
        self, patient_id: str, view: View, instants_with_gt: Mapping[Instant, int]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Real]]:
        """Fetches additional reference segmentations, interpolated between ED and ES instants.

        Args:
            patient_id: patient id formatted to match the identifiers in the mhd files' names.
            view: the view for which to fetch the patient's data.
            instants_with_gt: mapping between instant keys and the indices of their groundtruths in the sequence.

        Returns:
            - sequence of ultrasound images acquired between ED and ES. The trend is that the first images are
              closer to ED, and the last images are closer to ES.
            - segmentations interpolated between ED and ES. The trend is that the first segmentations are closer to
              ED, and the last segmentations are closer to ES.
            - metadata concerning the sequence.
        """
        patient_folder = self.data.joinpath(patient_id)
        sequence_fn_template = f"{patient_id}_{view.value}_sequence{{}}.mhd"

        # Indicate if ED comes after ES (normal) or the opposite
        reverse_sequence = False if instants_with_gt[Instant.ED] < instants_with_gt[Instant.ES] else True

        # Open interpolated segmentations
        data_x, data_y = [], []
        sequence, info = load_mhd(patient_folder.joinpath(sequence_fn_template.format("")))
        sequence_gt, _ = load_mhd(patient_folder.joinpath(sequence_fn_template.format("_gt")))

        for image, segmentation in zip(sequence, sequence_gt):  # For every instant in the sequence
            data_x.append(image)
            data_y.append(segmentation)

        if reverse_sequence:  # Reverse order of sequence if ES comes before ED
            data_x, data_y = list(reversed(data_x)), list(reversed(data_y))

        info = [item for sublist in info for item in sublist]  # Flatten info

        return data_x, data_y, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the CAMUS root directory, under which the patient directories are stored",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="camus.h5",
        help="Path to the HDF5 file to generate, containing all the raw image data and cross-validation metadata",
    )
    parser.add_argument(
        "--image_size",
        type=Tuple[int, int],
        default=(image_size, image_size),
        help="Target height and width at which to resize the image and groundtruth",
    )
    parser.add_argument(
        "-s",
        "--sequence",
        action="store_true",
        help="Augment the dataset by adding data for the sequence between ED and ES, where the "
        "groundtruths between ED and ES are interpolated linearly from reference segmentations",
    )
    parser.add_argument("-r", "--register", action="store_true", help="Apply registering on images and groundtruths")
    parser.add_argument(
        "--labels",
        type=Label.from_name,
        default=list(Label),
        nargs="+",
        choices=list(Label),
        help="Labels of the segmentation classes to take into account (including background). "
        "If None, target all labels included in the data",
    )
    args = parser.parse_args()

    CrossValidationDatasetGenerator()(
        args.data,
        args.output,
        target_image_size=args.image_size,
        sequence=args.sequence,
        register=args.register,
        labels=args.labels,
    )


if __name__ == "__main__":
    main()
