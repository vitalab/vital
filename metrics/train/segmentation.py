from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F


def dice(input: Tensor, target: Tensor, label: int, reduction: Literal['mean', 'none'] = 'mean') -> Tensor:
    """ Computes the dice score for a specific class.

    Args:
        input: (N, C, H, W), raw, unnormalized scores for each class.
        target: (N, H, W), where each value is 0 <= targets[i] <= C-1.
        label: class for which to compute the dice score.
        reduction: specifies the reduction to apply to the output:
                   ``'none'``: no reduction will be applied,
                   ``'mean'``: the sum of the output will be divided by the number of elements in the output.

    Returns:
        (1,) or (N,), the dice score for the requested class, reduced or for each sample.
    """
    label = torch.tensor(label)
    if reduction == 'mean':
        reduce_axis = (0, 1)
    else:  # reduction == 'none'
        reduce_axis = 1

    input = F.softmax(input, dim=1)[:, label, ...]  # For the input, extract the probabilities of the requested label
    target = torch.eq(target, label)  # For the target, extract the boolean mask of the requested label

    # Flatten the tensors to facilitate broadcasting
    input = torch.flatten(input, start_dim=1)
    target = torch.flatten(target, start_dim=1)

    # Compute dice score
    intersect = input + target
    sum_input = torch.sum(input, 1, keepdim=True)
    sum_target = torch.sum(target, 1, keepdim=True)
    dice = torch.mean((2 * intersect + 1) / (sum_input + sum_target + 1), dim=reduce_axis)
    return dice


def mean_dice(input: Tensor, target: Tensor, reduction: Literal['mean', 'none'] = 'mean') -> Tensor:
    """ Computes the mean dice score for all classes present in the target.

    Args:
        input: (N, C, H, W), raw, unnormalized scores for each class.
        target: (N, H, W), where each value is 0 <= targets[i] <= C-1.
        reduction: specifies the reduction to apply to the output:
                   ``'none'``: no reduction will be applied,
                   ``'mean'``: the sum of the output will be divided by the number of elements in the output.

    Returns:
        (1,) or (N,), the mean dice score for the classes in the target, reduced or for each sample.
    """
    labels = torch.unique(target[target.nonzero(as_tuple=True)])  # Identify classes (that are not background)

    # Compute the dice score for each individual class
    dices = torch.stack([dice(input, target, label, reduction=reduction)
                         for label in labels])

    mean_dice = torch.mean(dices, dim=0)  # Compute the mean dice over all classes
    return mean_dice
