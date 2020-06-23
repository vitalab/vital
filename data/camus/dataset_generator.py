import argparse
import os
from numbers import Real
from pathlib import Path
from typing import List, Dict, Tuple, Sequence, Mapping, Literal

import h5py
import numpy as np
from PIL.Image import LINEAR
from pathos.multiprocessing import Pool
from tqdm import tqdm

from vital.data.camus.config import image_size, View, Instant, CamusTags, Label
from vital.data.config import Subset
from vital.utils.image.io import load_mhd
from vital.utils.image.register.camus import CamusRegisteringTransformer
from vital.utils.image.transform import resize_image, resize_image, remove_labels


class CrossValidationDatasetGenerator:
    """Utility to process the raw CAMUS data (as it is when downloaded) and to generate cross validation HDF5 files.

    The cross validation split organizes the data in the following way:
        - split data into train, validation and test sets for a selection of subfolds
        - save each subfold in a separate HDF5 file for further use

    The processing on the data performs the following task, in order:
        - keep only a subset of the labels from the groundtruth (optional)
        - register images (optional)
        - resize images to a given size
    """

    def __init__(self, data: Path, output_template: Path, use_sequence: bool, register: bool,
                 labels: Sequence[Label] = None):
        """
        Args:
            data: path to the mhd data.
            output_template: path template for saving cross validation configurations to HDF5 files.
            use_sequence: whether to augment the dataset by adding sequence between ED and ES.
            register: enable/disable registering.
            labels: labels of the segmentation classes to predict.
        """
        self.data = data
        self.output_template = output_template
        self.use_sequence = use_sequence
        self.register = register
        self.labels_to_remove = [] if labels is None else [label for label in Label
                                                           if label not in labels]

    def create_subfold_dataset(self, subfold: int):
        """This function writes a cross validation configuration of the dataset, where ``subfold`` makes up the test
        set, to a HDF5 file.

        Args:
            subfold: the id of the test set for the cross-validation configuration.
        """
        # Create directory hierarchy if the output template is not only a filename
        self.output_template.parent.mkdir(exist_ok=True)
        subfold_dataset = h5py.File(str(self.output_template).format(subfold), 'w')

        for subset, subgroup_name in zip(Subset, ['training', 'validation', 'testing']):
            # read test train valid details in the correspond .txt files
            group_patients = self.get_subgroup_from_file(subfold, subgroup_name)

            # save in hdf5
            self._create_subset(subfold_dataset.create_group(subset.value), subfold, group_patients)

        # Add additional metadata to the file
        subfold_dataset.attrs[CamusTags.full_sequence] = self.use_sequence
        subfold_dataset.attrs[CamusTags.registered] = self.register

    def get_subgroup_from_file(self, subfold: int, subset: Literal['training', 'validation', 'testing']) -> List[str]:
        """Reads patient ids for a subset of a cross-validation configuration.

        Args:
            subfold: the id of the test set for the cross-validation configuration.
            subset: name of the subset for which to fetch patient ids for the cross-validation configuration.

        Returns:
            patient ids.
        """
        list_fn = self.data.joinpath('listSubGroups', f'subGroup{subfold}_{subset}.txt')
        # Open text file containing patient ids (one patient id by row)
        with open(str(list_fn), 'r') as f:
            patient_ids = [line for line in f.read().splitlines()]
        return patient_ids

    def _create_subset(self, subset_group: h5py.Group, subfold: int, patient_ids: Sequence[str]):
        """This function writes in a hdf5 the data for a given subset.

        Args:
            subset_group: group for the subset in which to write the data.
            subfold: int, the id of the test set for the cross-validation configuration.
            patient_ids: ids of the patient whose data will be written in the subset.
        """
        registering_transformer = CamusRegisteringTransformer(num_classes=Label.count(),
                                                              crop_shape=(image_size, image_size))
        for patient_id in tqdm(patient_ids, total=len(patient_ids), unit='patient',
                               desc="Creating {} group for subfold {}".format(os.path.basename(subset_group.name),
                                                                              subfold)):

            patient_group = subset_group.create_group(patient_id)

            img_save_options = {'dtype': np.float32, 'compression': "gzip", 'compression_opts': 4}
            seg_save_options = {'dtype': np.uint8, 'compression': "gzip", 'compression_opts': 4}
            for view in View:
                # The order of the instants within a view dataset is chronological: ED -> ES -> ED
                data_x, data_y, info_view, instants_with_gt = self._get_view_data(patient_id, view)

                data_y = remove_labels(data_y, [lbl.value for lbl in self.labels_to_remove],
                                       fill_label=Label.BG.value)

                if self.register:
                    registering_parameters, data_y_proc, data_x_proc = registering_transformer.register_batch(data_y,
                                                                                                              data_x)
                else:
                    data_x_proc = np.array([resize_image(x, (image_size, image_size), resample=LINEAR)
                                            for x in data_x])
                    data_y_proc = np.array([resize_image(y, (image_size, image_size))
                                            for y in data_y])

                # Write image and groundtruth data
                patient_view_group = patient_group.create_group(view.value)
                patient_view_group.create_dataset(name=CamusTags.img_proc, data=data_x_proc[..., np.newaxis],
                                                  **img_save_options)
                patient_view_group.create_dataset(name=CamusTags.gt, data=data_y[..., np.newaxis], **seg_save_options)
                patient_view_group.create_dataset(name=CamusTags.gt_proc, data=data_y_proc[..., np.newaxis],
                                                  **seg_save_options)

                # Write metadata useful for providing instants or full sequences
                patient_view_group.attrs[CamusTags.info] = info_view
                for instant, instant_idx in instants_with_gt.items():
                    patient_view_group.attrs[instant.value] = instant_idx

                # Write metadata concerning the registering applied
                if self.register:
                    for registering_step, values in registering_parameters.items():
                        patient_view_group.attrs[registering_step] = values

    def _get_view_data(self, patient_id: str, view: View) \
            -> Tuple[np.ndarray, np.ndarray, List[Real], Dict[Instant, int]]:
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
        view_info_fn = self.data.joinpath(patient_id, f'Info_{view.value}.cfg')

        # Determine the index of segmented instants in sequence
        instants_with_gt = {}
        with open(str(view_info_fn), 'r') as view_info_file:
            view_info_lines = view_info_file.read().splitlines()
            for instant_idx, instant in enumerate(Instant):
                instants_with_gt[instant] = int(view_info_lines[instant_idx].split()[-1]) - 1

        # Get data for the whole sequence ranging from ED to ES
        sequence, sequence_gt, info = self._get_sequence_data(patient_id, view, instants_with_gt)

        if instants_with_gt[Instant.ED] > instants_with_gt[Instant.ES]:  # Ensure ED comes before ES (swap when ES->ED)
            instants_with_gt[Instant.ED], instants_with_gt[Instant.ES] = \
                instants_with_gt[Instant.ES], instants_with_gt[Instant.ED]

        # Include all or only some instants from the input and reference data according to the parameters
        data_x, data_y = [], []
        if self.use_sequence:
            data_x, data_y = sequence, sequence_gt
        else:
            for instant in Instant:
                data_x.append(sequence[instants_with_gt[instant]])
                data_y.append(sequence_gt[instants_with_gt[instant]])

            # Update indices of instants with annotated segmentations in view sequences in newly sliced sequences
            instants_with_gt = {instant_key: idx for idx, instant_key in enumerate(Instant)}

        # Add channel dimension
        return np.array(data_x), np.array(data_y), info, instants_with_gt

    def _get_sequence_data(self, patient_id: str, view: View, instants_with_gt: Mapping[Instant, int]) \
            -> Tuple[List[np.ndarray], List[np.ndarray], List[Real]]:
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
        sequence_fn_template = f'{patient_id}_{view.value}_sequence{{}}.mhd'

        # Indicate if ED comes after ES (normal) or the opposite
        reverse_sequence = False if instants_with_gt[Instant.ED] < instants_with_gt[Instant.ES] else True

        # Open interpolated segmentations
        data_x, data_y = [], []
        sequence, info = load_mhd(patient_folder.joinpath(sequence_fn_template.format('')))
        sequence_gt, _ = load_mhd(patient_folder.joinpath(sequence_fn_template.format('_gt')))

        for image, segmentation in zip(sequence, sequence_gt):  # For every instant in the sequence
            data_x.append(image)
            data_y.append(segmentation)

        if reverse_sequence:  # Reverse order of sequence if ES comes before ED
            data_x, data_y = list(reversed(data_x)), list(reversed(data_y))

        info = [item for sublist in info for item in sublist]  # Flatten info

        return data_x, data_y, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True,
                        help="Root directory under which the patient directories are stored")
    parser.add_argument("--output_template", type=Path, default="camus_subfold_{}.h5",
                        help="Path template for saving cross validation configurations to HDF5 files")
    parser.add_argument("-s", "--sequence", action='store_true',
                        help="Augment the dataset by adding data for the sequence between ED and ES, where the "
                             "groundtruths between ED and ES are interpolated linearly from reference segmentations")
    parser.add_argument("-r", "--register", action='store_true', help="Apply registering on images and groundtruths")
    parser.add_argument("--labels", type=Label.from_name, default=list(Label), nargs='+', choices=list(Label),
                        help="Labels of the segmentation classes to take into account (including background). "
                             "If None, target all labels included in the data")
    args = parser.parse_args()

    dataset_generator = CrossValidationDatasetGenerator(args.data, args.output_template, args.sequence, args.register,
                                                        labels=args.labels)

    with Pool() as pool:
        pool.map(dataset_generator.create_subfold_dataset, range(1, 11))
