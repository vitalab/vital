from enum import Enum
from typing import List, Sequence, Union

from vital.utils.parameters import parameters

SemanticStructureId = Union[int, Sequence[int]]


class DataTag(Enum):
    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)

    @classmethod
    def values(cls) -> List:
        return [e.value for e in cls]

    @classmethod
    def count(cls) -> int:
        return sum(1 for _ in cls)

    @classmethod
    def from_name(cls, name: str) -> "DataTag":
        return cls[name.upper()]


class Subset(DataTag):
    """
    Args:
        train: id of the training subset.
        valid: id of the validation subset.
        test: id of the testing subset.
    """

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


@parameters
class Tags:
    """Class to gather the tags referring to the different type of data stored.

    Args:
        img: name of the tag referring to images.
        gt: name of the tag referring to groundtruths, used as reference when evaluating models' scores.
        pred: name of the tag referring to original predictions.
        post_pred: name of the tag referring to post processed predictions.
    """

    img: str = "img"
    gt: str = "gt"
    regression: str = "reg"
    pred: str = "pred"
    post_pred: str = "post_pred"
    encoding: str = "encoding"