import os
from abc import ABC
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Type, Union, List

from pytorch_lightning import Trainer

from vital.systems.vital_system import VitalSystem


class VitalTrainer(ABC):
    """Abstract trainer that runs the main training/val loop, etc... using Lightning Trainer."""

    @classmethod
    def get_trainable_systems(cls) -> Dict[str, Type[VitalSystem]]:
        """Maps Lightning modules trainable through this trainer to short, descriptive acronyms.

        Returns:
            mapping between Lightning modules trainable through this trainer and their short, descriptive acronyms..
        """
        raise NotImplementedError

    @classmethod
    def main(cls):
        """Sets-up the CLI for the Lightning modules trainable through this trainer and calls the main training/val
        loop.
        """
        trainable_systems = cls.get_trainable_systems()

        # Initialize the parser with generic trainer arguments
        parser = ArgumentParser()
        cls.add_trainer_args(parser)

        # Add subparsers for all systems available through the trainer
        system_subparsers = parser.add_subparsers(title='system', dest='system', description="System to train")
        for system_opt, system_cls in trainable_systems.items():
            system_subparsers.add_parser(system_opt, help=f'{system_opt} system',
                                         parents=[system_cls.build_parser()])

        # Parse args and run the target system
        args = parser.parse_args()
        system_cls = trainable_systems[args.system]
        VitalTrainer.run_system(system_cls, args)

    @staticmethod
    def run_system(system_cls: Type[VitalSystem], hparams: Namespace):
        """Handles the training/validation loop.

        Args:
            system_cls: Lightning module to train/test.
            hparams: namespace of arguments parsed from the CLI.
        """
        trainer = Trainer(
            default_save_path=hparams.save_dir,
            fast_dev_run=hparams.fast_dev_run,
            profiler=hparams.profiler,
            weights_summary=hparams.weights_summary,
            gpus=hparams.gpus,
            num_nodes=hparams.num_nodes,
            min_epochs=hparams.min_epochs if 'min_epochs' in hparams else 1,
            max_epochs=hparams.max_epochs
        )

        if hparams.predict:  # If we want to skip training and go straight to testing
            model = system_cls.load_from_checkpoint(hparams.pretrained)
            trainer.model = model
            trainer.test(model)
        else:  # If we want to train and then test the system
            model = system_cls(hparams)
            trainer.fit(model)
            trainer.test()

    @classmethod
    def add_trainer_args(cls, parser: ArgumentParser):
        """Adds generic Lightning trainer arguments to a parser object.

        Args:
            parser: parser object to which trainer arguments will be added.
        """
        # save/load parameters
        parser.add_argument("--save_dir", type=Path, required=True, help="Path for logs+weights+results")
        parser.add_argument("--pretrained", type=Path, help="Path to Lightning module checkpoints to restore system")

        # evaluation parameters
        parser.add_argument("--predict", action='store_true', help="Skip training and do test phase")

        # training configuration parameters
        parser.add_argument('--weights_summary', type=str, default='full', choices=['full', 'top', None])
        parser.add_argument('--gpus', type=Union[int, List[int]], default=1)
        parser.add_argument('--num_nodes', type=int, default=1)
        parser.add_argument('--workers', type=int, default=os.cpu_count() // 2)

        # Lightning configuration parameters
        parser.add_argument('--fast_dev_run', action='store_true',
                            help="Runs full iteration over everything to find bugs")
        parser.add_argument('--profiler', action='store_true',
                            help="Profile standard training events")
        return parser
