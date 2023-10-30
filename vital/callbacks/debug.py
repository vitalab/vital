import re
from typing import Dict, Optional, Sequence, Type, Union

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CometLogger, Logger
from pytorch_lightning.loggers.logger import DummyLogger
from torch import nn

from vital.utils.importlib import import_from_module


def _log_layers_histograms(
    logger: Logger,
    layers: Dict[str, nn.Module],
    step: int = None,
    include_weight: bool = True,
    include_grad: bool = True,
) -> None:
    # Define adapter to log histogram w.r.t. the logger
    match logger:
        case DummyLogger():

            def _log_histogram(name: str, data: np.ndarray) -> None:
                pass

        case CometLogger():

            def _log_histogram(name: str, data: np.ndarray) -> None:  # noqa: F811
                logger.experiment.log_histogram_3d(values=data, name=name, step=step)

        case _:
            raise NotImplementedError(f"Logging histogram not implemented for '{logger.__class__.__name__}' logger.")

    for layer_name, layer in layers.items():
        # Extract the relevant parameters from the current layer
        match layer:
            case nn.MultiheadAttention():
                params_to_log = {"in_proj": layer.in_proj_weight, "out_proj": layer.out_proj.weight}
            case nn.Linear() | nn.LayerNorm():
                params_to_log = {"": layer.weight}
            case _:
                raise NotImplementedError(
                    f"Logging layer weights/grads not implemented for '{layer.__class__.__name__}' layer."
                )

        # Log the layer's parameters/gradients
        for params_name, params in params_to_log.items():
            name = f"{layer_name}_{params_name}" if params_name else layer_name

            if include_weight:
                _log_histogram(f"{name}_weight", params.detach().cpu().numpy())
            if include_grad and params.grad is not None:
                # Log grad if they're available. They might not be available in some cases, e.g. un-trainable layers
                _log_histogram(f"{name}_grad", params.grad.cpu().numpy())


class LayersHistogramsLogger(Callback):
    """Logs weights and/or gradients from layers of a model at given training steps."""

    def __init__(
        self,
        layer_types: Sequence[Union[str, Type[nn.Module]]],
        submodules: Sequence[str] = None,
        log_every_n_steps: int = 50,
        include_weight: bool = True,
        include_grad: bool = True,
    ):
        """Initializes class instance.

        Args:
            layer_types: Types or classpaths of layers for which to log histograms.
            submodules: Name of the fields (e.g. 'encoder', 'classifier', etc.) corresponding to fields inside which to
                search for matching layers. If none is provided, the Lightning module will be inspected starting from
                its root.
            log_every_n_steps: Frequency at which to log the attention weights computed during the forward pass.
            include_weight: Whether to log the layers' weights.
            include_grad: Whether to log the layers' gradients.
        """
        self.layer_types = [
            layer_type if not isinstance(layer_type, str) else import_from_module(layer_type)
            for layer_type in layer_types
        ]
        self.submodule_names = submodules

        self.log_every_n_steps = log_every_n_steps
        self.train_step = 0
        self.include_weight = include_weight
        self.include_grad = include_grad

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        """Identifies layers for which to log weights/gradients during training.

        Args:
            trainer: `Trainer` used in the experiment.
            pl_module: `LightningModule` used in the experiment.
            stage: Current stage (e.g. fit, test, etc.) of the experiment.
        """

        def _convert_camel_case_to_snake_case(string: str) -> str:
            return re.sub("(?!^)([A-Z]+)", r"_\1", string).lower()

        # Extract the requested submodule from the root module
        submodules_to_inspect = {"self": pl_module}
        if self.submodule_names:
            submodules_to_inspect = {}
            for submodule_name in self.submodule_names:
                # For each submodule, (recursively) follow the chain of attributes to get the actual submodule
                module = pl_module
                for submodule_name in submodule_name.split("."):
                    module = getattr(module, submodule_name)
                submodules_to_inspect[submodule_name] = module

        # Get references to the specific layers to watch, prepending the submodule they're from to the layer name
        self.layers_to_log = {
            f"{submodule_name}.{_convert_camel_case_to_snake_case(layer.__class__.__name__)}_{layer_idx}": layer
            for submodule_name, submodule in submodules_to_inspect.items()
            for layer_type in self.layer_types
            for layer_idx, layer in enumerate(layer for layer in submodule.modules() if isinstance(layer, layer_type))
        }

    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Logs histograms of the weights/gradients for each previously selected layer.

        Args:
            trainer: `Trainer` used in the experiment.
            pl_module: `LightningModule` used in the experiment.
        """
        if (self.train_step % self.log_every_n_steps) == 0:
            _log_layers_histograms(
                trainer.logger,
                self.layers_to_log,
                step=self.train_step,
                include_weight=self.include_weight,
                include_grad=self.include_grad,
            )

        self.train_step += 1
