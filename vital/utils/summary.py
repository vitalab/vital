from collections import OrderedDict
from functools import reduce
from operator import mul
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
from torch import Tensor, nn


def summary_info(model: nn.Module, example_input_array: Any, batch_size: int = -1) -> Tuple[str, Tuple[int, int]]:
    """Computes info to display a summary of a network to the console, similar to `model.summary()` in Keras.

    Modifications:
        - Modified parameters to force `device` to be specified
        - Added support for modules with multiple outputs
        - Minor coding standards improvements
        - Added docstring and type hints

    References:
        - Implementation inspired by https://github.com/sksq96/pytorch-summary.

    Args:
        model: Network for which to print the summary.
        example_input_array: Example of an arbitrary data structure that ``model`` would take as input for inference,
            i.e. input with the expected shape but possibly dummy data. The arbitrary input can be any (nested)
            combination of sequential data structures (e.g. tuples, lists, etc.).
        batch_size: Value to use for the batch size when printing the input/output shape of the model's layers.

    Returns:
        - Model summary string
        - Information about the parameters
            - Total number of parameters
            - Total number of trainable parameters
    """
    summary_str = ""

    def register_hook(module: nn.Module) -> None:
        def hook(module: nn.Module, input: Tuple[torch.Tensor], output: Union[torch.Tensor, Tuple, List, Dict]) -> None:
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, dict):
                output = list(output.values())
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
    if not isinstance(example_input_array, Sequence):
        example_input_array = [example_input_array]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*example_input_array)

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
            layer, str(summary[layer]["output_shape"]), "{0:,}".format(summary[layer]["nb_params"])
        )
        total_params += summary[layer]["nb_params"]

        # multiple outputs to the layer
        output_size = summary[layer]["output_shape"]
        if not isinstance(summary[layer]["output_shape"][0], Sequence):
            output_size = [output_size]

        total_output += sum(reduce(mul, out_size) for out_size in output_size)
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    def find_input_size(input: Any) -> int:
        if isinstance(input, Tensor):  # if current input is a tensor
            input_size = reduce(mul, input.shape[1:])  # compute the total size of the current input
        elif not input:  # base case: no more inputs
            input_size = 0
        else:  # if current element is a sequence
            # total size is the (recursive) size of the current element + size of the remaining elements
            input_size = find_input_size(input[0]) + find_input_size(input[1:])
        return input_size

    # assume 4 bytes/number (float on cuda).
    total_input = find_input_size(example_input_array)
    total_input_size = abs(total_input * batch_size * 4.0 / (1024 ** 2.0))
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
