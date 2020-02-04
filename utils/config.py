from enum import Enum
from typing import Union, Sequence

from vital.utils.parameters import parameters

SemanticStructureId = Union[int, Sequence[int]]


class ConfigurationLabel(Enum):

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)

    @classmethod
    def values(cls):
        return [e.value for e in cls]

    @classmethod
    def count(cls):
        return sum(1 for _ in cls)

    @classmethod
    def from_name(cls, name):
        try:
            return cls[name.upper()]
        except KeyError:
            return name


class Subset(ConfigurationLabel):
    """
    Args:
        train: id of the training subset.
        valid: id of the validation subset.
        test: id of the testing subset.
    """
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


@parameters
class ResultTags:
    """ Class to gather the tags referring to the generic results stored in the HDF5 result files.

    Args:
        img: name of the tag referring to images.
        gt: name of the tag referring to groundtruths, used as reference when evaluating models' scores.
        pred: name of the tag referring to original predictions.
        post_pred: name of the tag referring to post processed predictions.
    """
    img: str = 'img'
    gt: str = 'gt'
    pred: str = 'pred'
    post_pred: str = 'post_pred'
    encoding: str = 'encoding'
