from enum import Enum
from typing import List, Sequence, Union

from vital.utils.parameters import parameters

SemanticStructureId = Union[int, Sequence[int]]


class DataTag(Enum):
    """Extension of Python's ``Enum`` type to provide easy conversion and display methods."""

    def __str__(self):  # noqa: D105
        return self.name.lower()

    def __repr__(self):  # noqa: D105
        return str(self)

    @classmethod
    def values(cls) -> List:
        """Lists the values for all the elements of the enumeration.

        Returns:
            Values of all the elements in the enumeration.
        """
        return [e.value for e in cls]

    @classmethod
    def count(cls) -> int:
        """Counts the number of elements in the enumeration.

        Returns:
            Count of the number of elements in the enumeration.
        """
        return sum(1 for _ in cls)

    @classmethod
    def from_name(cls, name: str) -> "DataTag":
        """Fetches an element of the enumeration based on its name.

        Args:
            name: attribute name of the element in the enumeration.

        Returns:
            Element from the enumeration corresponding to the requested name.
        """
        return cls[name.upper()]


class Subset(DataTag):
    """Enumeration to gather tags referring to commonly used subsets of a whole dataset.

    Attributes:
        TRAIN: Label of the training subset.
        VAL: Label of the validation subset.
        TEST: Label of the testing subset.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@parameters
class Tags:
    """Class to gather the tags referring to the different type of data stored.

    Args:
        id: Tag referring to an identifier (not necessarily unique) associated to the data.
        img: Tag referring to images.
        gt: Tag referring to groundtruths, used as reference when evaluating models' scores.
        aux: Tag referring to additional (auxiliary) information associated to the input.
            (e.g. timestamp in a temporal sequence, z position, etc.)
        pred: Tag referring to original predictions.
        post_pred: Tag referring to post processed predictions.
        encoding: Tag referring to an encoding of the system's input.
    """

    id: str = "id"
    img: str = "img"
    gt: str = "gt"
    aux: str = "aux"
    pred: str = "pred"
    post_pred: str = "post_pred"
    encoding: str = "encoding"
