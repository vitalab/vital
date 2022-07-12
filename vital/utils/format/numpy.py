from typing import Callable, Union

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


def wrap_pad(
    fn: Callable[..., np.ndarray],
    array: np.ndarray,
    *args,
    pad_mode: str,
    pad_width: Union[int, float],
    pad_axis: int = None,
    **kwargs,
) -> np.ndarray:
    """Pad array before calling the function that processes the array, and undo padding on the result.

    Args:
        fn: Function that processes an array and returns a result of the same shape.
        array: Array to pad before processing.
        *args: Additional parameters to pass along to ``fn``.
        pad_mode: Mode used to determine how to pad points at the beginning/end of the array. The options available are
            those of the ``mode`` parameter of ``numpy.pad``.
        pad_width: If it is an integer, the number of entries to repeat before/after the array. If it is a float, the
            fraction of the data's length to repeat before/after the array.
        pad_axis: Axis along which to pad. If `None`, pad along all axes.
        **kwargs: Additional parameters to pass along to ``fn``.

    Returns:
        Result of processing the padded array.
    """
    if pad_mode:
        if isinstance(pad_width, float):
            len_pad_width = int(len(array) * pad_width)
        elif isinstance(pad_width, int):
            len_pad_width = pad_width
        else:
            raise ValueError(
                f"The padding width requested is of unexpected type: {type(pad_width)}. It should be either a float "
                f"(to indicate a fraction of the signal's length to pad) or an integer (to indicate a number of points "
                f"to pad)."
            )

        if pad_axis is None:
            pad_mask = [(len_pad_width, len_pad_width)] * array.ndim
        else:
            pad_mask = [(0, 0)] * array.ndim
            pad_mask[pad_axis] = (len_pad_width, len_pad_width)

        array = np.pad(array, pad_mask, mode=pad_mode)

    res = fn(array, *args, **kwargs)

    if pad_mode:
        pad_slice = tuple(np.s_[before : dim - after] for dim, (before, after) in zip(array.shape, pad_mask))
        res = res[pad_slice]

    return res
