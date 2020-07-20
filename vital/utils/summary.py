from collections import OrderedDict
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn


def summary_info(
    model: nn.Module,
    input_size: Union[List[Tuple[int, ...]], Tuple[int, ...]],
    device: torch.device,
    batch_size: int = -1,
    dtypes=None,
) -> Tuple[str, Tuple[int, int]]:
    """Computes info to display a summary of a network to the console, similar to `model.summary()` in Keras.

    Implementation taken from: https://github.com/sksq96/pytorch-summary

    Modifications:
        - Modified parameters to force `device` to be specified
        - Added support for modules with multiple outputs
        - Minor coding standards improvements
        - Added docstring and type hints

    Args:
        model: network for which to print the summary.
        input_size: shape of the input to the network's `forward` function.
                    Can be multiple inputs, in which the input size should be a list of tuples, each indicating the
                    shape of one input.
        device: device on which the model is located.
        batch_size: value to use for the batch size when printing the input/output shape of the model's layers.
        dtypes: data types of the input to the model. Defaults to `FloatTensor` if not specified.

    Returns:
        - model summary string
        - information about the parameters
            - total number of parameters
            - total number of trainable parameters
    """
    if dtypes is None:
        dtypes = [torch.FloatTensor] * len(input_size)

    summary_str = ""

    def register_hook(module: nn.Module) -> None:
        def hook(module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor) -> None:
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device) for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer, str(summary[layer]["output_shape"]), "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        # multiple outputs to the layer
        output_size = summary[layer]["output_shape"]
        if not isinstance(summary[layer]["output_shape"][0], Sequence):
            output_size = [output_size]

        total_output += sum(np.prod(out_size) for out_size in output_size)
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ())) * batch_size * 4.0 / (1024 ** 2.0))
    total_output_size = abs(2.0 * total_output * 4.0 / (1024 ** 2.0))  # x2 for gradients
    total_params_size = abs(total_params * 4.0 / (1024 ** 2.0))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params - trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    return summary_str, (total_params, trainable_params)