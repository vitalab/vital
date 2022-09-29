from dataclasses import dataclass
from enum import IntEnum, auto, unique
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from strenum import LowercaseStrEnum

SemanticStructureId = Union[int, Sequence[int]]
ProtoLabel = Union[int, str, "LabelEnum"]


class LabelEnum(IntEnum):
    """Abstract base class for enumerated labels, providing more functionalities on top of the core `IntEnum`."""

    def __str__(self):  # noqa: D105
        return self.name.lower()

    @classmethod
    def from_proto_label(cls, proto_label: ProtoLabel) -> "LabelEnum":
        """Creates a label from a protobuf label.

        Args:
            proto_label: Either the integer value of a label, the (case-insensitive) name of a label, or directly a
                `LabelEnum`.

        Returns:
            A `LabelEnum` instance.
        """
        if isinstance(proto_label, int):
            label = cls(proto_label)
        elif isinstance(proto_label, str):
            label = cls[proto_label.upper()]
        elif isinstance(proto_label, cls):
            label = proto_label
        else:
            raise ValueError(
                f"Unsupported label type: {type(proto_label)} for proto-label. Should be one of: {ProtoLabel}."
            )
        return label

    @classmethod
    def from_proto_labels(cls, proto_labels: Sequence[ProtoLabel]) -> List["LabelEnum"]:
        """Creates a list of labels from a sequence of protobuf labels.

        Args:
            proto_labels: Sequence of either integer values of labels, (case-insensitive) names of labels, or directly
                `LabelEnum`s.

        Returns:
            List of `LabelEnum` instances
        """
        return [cls.from_proto_label(proto_label) for proto_label in proto_labels]


@unique
class Subset(LowercaseStrEnum):
    """Commonly used subsets of a whole dataset."""

    TRAIN = auto()
    """Training subset."""
    VAL = auto()
    """Validation subset."""
    TEST = auto()
    """Testing subset."""
    PREDICT = auto()
    """Prediction subset."""


@dataclass(frozen=True)
class Tags:
    """Class to gather the tags referring to the different type of data stored.

    Args:
        id: Tag referring to a unique identifier for the data.
        group: Tag referring to an identifier for the group the data belongs to.
        neighbors: Tag referring to an item's neighbor, provided alongside the item itself.
        img: Tag referring to images.
        gt: Tag referring to groundtruths, used as reference when evaluating models' scores.
        pred: Tag referring to original predictions.
        post_pred: Tag referring to post processed predictions.
        encoding: Tag referring to an encoding of the system's input.
    """

    id: str = "id"
    group: str = "group"
    neighbors: str = "neighbors"
    img: str = "img"
    gt: str = "gt"
    pred: str = "pred"
    post_pred: str = "post_pred"
    encoding: str = "z"


_Size = Tuple[int, ...]


class DataParameters(NamedTuple):
    """Class for defining parameters related to the nature of the data.

    Args:
        in_shape: Shape of the input data, if constant for all items (e.g. channels, height, width). It can be a dict of
            multiple shapes in the case of multi-modal data.
        out_shape: Shape of the target data, if constant for all items (e.g. classes, height, width). It can be a dict
            of multiple shapes in the case of multi-modal data.
        labels: Labels provided with the data, required when using segmentation task APIs.
    """

    in_shape: Optional[Union[_Size, Dict[str, _Size]]] = None
    out_shape: Optional[Union[_Size, Dict[str, _Size]]] = None
    labels: Optional[Tuple[LabelEnum, ...]] = None
