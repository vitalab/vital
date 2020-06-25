from argparse import ArgumentParser, Namespace
from pathlib import Path

from torch.utils.data import DataLoader

from vital.data.camus.config import image_size, in_channels, Label
from vital.data.camus.dataset import Camus, DataParameters
from vital.data.config import Subset
from vital.systems.vital_system import SystemDataManagerMixin


class CamusSystemDataManagerMixin(SystemDataManagerMixin):
    use_da: bool = False  # whether the system applies Data Augmentation (DA) by default.
    use_sequence: bool = False  # whether the system uses complete sequences by default.
    use_sequence_index: bool = False  # whether the system requires instants' normalized indices in the sequences.

    def __init__(self, hparams: Namespace):
        super().__init__(hparams)
        self.data_params = DataParameters(in_shape=(image_size, image_size, in_channels),
                                          out_shape=(image_size, image_size, len(hparams.labels)),
                                          use_sequence_index=self.use_sequence_index)
        self._labels = [str(label) for label in self.hparams.labels]

    def setup(self, stage: str):
        common_args = {'path': self.hparams.dataset_path,
                       'labels': self.hparams.labels,
                       'use_sequence': self.hparams.use_sequence,
                       'use_sequence_index': self.data_params.use_sequence_index}
        self.dataset = {
            Subset.TRAIN: Camus(image_set=Subset.TRAIN, **common_args),
            Subset.VALID: Camus(image_set=Subset.VALID, **common_args),
            Subset.TEST: Camus(image_set=Subset.TEST, predict=True, **common_args)
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset[Subset.TRAIN],
                          batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset[Subset.VALID],
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset[Subset.TEST], batch_size=None, num_workers=self.hparams.workers)

    @classmethod
    def add_data_manager_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser = super().add_data_manager_args(parser)
        parser.add_argument("dataset_path", type=Path, help="Path to the HDF5 dataset")
        parser.add_argument("--labels", type=Label.from_name, default=list(Label), nargs='+', choices=list(Label),
                            help="Labels of the segmentation classes to take into account (including background). "
                                 "If None, target all labels included in the data")

        if cls.use_da:
            parser.add_argument("no_da", dest="use_da", action='store_false',
                                help="Disable online dataset augmentation")
        else:
            parser.add_argument("--use_da", dest="use_da", action='store_true',
                                help="Enable online dataset augmentation")

        if cls.use_sequence:
            parser.add_argument("--no_sequence", dest="use_sequence", action='store_false',
                                help="Disable use of interpolated sequences")
        else:
            parser.add_argument("--use_sequence", dest="use_sequence", action='store_true',
                                help="Enable use of interpolated sequences")

        return parser
