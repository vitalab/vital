"""Format utilities.
Derived from Keras' numpy utility: https://github.com/keras-team/keras/blob/master/keras/utils/np_utils.py"""

import numpy as np


def to_categorical(y: np.ndarray, num_classes: int = None, channel_axis: int = -1, dtype='float32') -> np.ndarray:
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    Args:
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        channel_axis: index of the channel axis to expand.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    Returns:
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
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
