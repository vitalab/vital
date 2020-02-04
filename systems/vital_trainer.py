import argparse
from abc import ABC
from dataclasses import asdict
from typing import Dict, Type

from pytorch_lightning import Trainer

from vital.utils.parameters import TrainerParameters
from vital.systems.vital_system import VitalSystem


class VitalTrainer(ABC):
    """Abstract trainer that runs the main training/val loop, etc... using Lightning Trainer."""

    @classmethod
    def get_trainable_systems(cls) -> Dict[str, Type[VitalSystem]]:
        """ Maps Lightning modules trainable through this trainer to short, descriptive acronyms.

        Returns:
            mapping between Lightning modules trainable through this trainer and their short, descriptive acronyms..
        """
        raise NotImplementedError

    @classmethod
    def main(cls):
        """ Sets-up the CLI for the Lightning modules trainable through this trainer and calls the main training/val
        loop.
        """
        trainable_systems = cls.get_trainable_systems()

        # Build parser for all systems available through the trainer
        parser = argparse.ArgumentParser()
        system_subparsers = parser.add_subparsers(title='system', dest='system', description="System to train")
        for system_opt, system_cls in trainable_systems.items():
            system_subparsers.add_parser(system_opt, help=f'{system_opt} system',
                                         parents=[system_cls.build_parser()])
        args = parser.parse_args()

        # Parse args and run the target system
        system_cls = trainable_systems[args.system]
        VitalTrainer.run_system(system_cls, **system_cls.parse_args(args))

    @staticmethod
    def run_system(system_cls: Type[VitalSystem], trainer_params: TrainerParameters,
                   predict: bool = False, **system_params):
        """ Handles the training/val loop.

        Args:
            system_cls: Lightning module to train/test.
            trainer_params:
            predict: whether to skip and directly run the test phasing.
            **system_params: parameters to configure the system to train/test.
        """
        trainer = Trainer(**asdict(trainer_params))  # init trainer

        if predict:
            # TODO load system from checkpoints
            model: system_cls
            trainer.model = model
            pass
        else:
            model = system_cls(**system_params)  # init system
            trainer.fit(model)  # train system

        trainer.run_evaluation(test=True)  # test system
