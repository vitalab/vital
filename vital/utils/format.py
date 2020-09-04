from typing import Any, Dict, Mapping, Sequence, TypeVar, Union

import numpy as np


def to_onehot(y: np.ndarray, num_classes: int = None, channel_axis: int = -1, dtype: str = "uint8") -> np.ndarray:
    """Converts a class vector (integers) to binary class matrix.

    Args:
        y: Class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: Total number of classes.
        channel_axis: Index of the channel axis to expand.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...).

    Returns:
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype="int")
    input_shape = list(y.shape)
    if input_shape and input_shape[channel_axis] == 1 and len(input_shape) > 1:
        input_shape.pop(channel_axis)
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape[:]
    categorical_axis = channel_axis if channel_axis != -1 else len(output_shape)
    output_shape.insert(categorical_axis, num_classes)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def to_categorical(y: np.ndarray, channel_axis: int = -1, dtype: str = "uint8") -> np.ndarray:
    """Converts a binary class matrix to class vector (integers).

    Args:
        y: Matrix to be converted into a class vector
            (pixel-wise binary vectors of length num_classes).
        channel_axis: Index of the channel axis to expand.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...).

    Returns:
        A class vector representation of the input.
    """
    return y.argmax(axis=channel_axis).astype(dtype)


def prefix(map: Mapping[str, Any], prefix: str, exclude: Union[str, Sequence[str]] = None) -> Dict[str, Any]:
    """Prepends a prefix to the keys of a mapping with string keys.

    Args:
        map: Mapping with string keys for which to add a prefix to the keys.
        prefix: Prefix to add to the current keys in the mapping.
        exclude: Keys to exclude from the prefix addition. These will remain unchanged in the new mapping.

    Returns:
        Mapping where the keys have been prepended with `prefix`.
    """
    if exclude is None:
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]

    return {f"{prefix}{k}" if k not in exclude else k: v for k, v in map.items()}


Item = TypeVar("Item")


def squeeze(seq: Sequence[Item]) -> Union[Item, Sequence[Item]]:
    """Extracts the item from a sequence if it is the only one, otherwise leaves the sequence as is.

    Args:
        seq: Sequence to process.

    Returns:
        Single item from the sequence, or sequence itself it has more than one item.
    """
    if len(seq) == 1:
        (seq,) = seq
    return seq
