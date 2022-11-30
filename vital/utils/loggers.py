import pandas as pd
from pytorch_lightning.loggers import CometLogger, Logger
from pytorch_lightning.loggers.logger import DummyLogger


def log_figure(logger: Logger, **kwargs) -> None:
    """Logs a figure using a Lightning logger.

    Args:
        logger: Logger with which to log the figure.
        **kwargs: Additional parameters to pass along to the figure-logging function for the specific logger being
            used (e.g. Tensorboard, Comet, etc.).
    """
    match logger:
        case DummyLogger():
            pass
        case CometLogger():
            if kwargs.get("step") == 0:
                # Since figures logged at step 0 are not shown by Comet, workaround by assigning step 1 in this case
                kwargs["step"] = 1
            logger.experiment.log_figure(**kwargs)
        case _:
            raise NotImplementedError(f"Logging figures not implemented for '{logger.__class__.__name__}' logger.")


def log_dataframe(logger: Logger, data: pd.DataFrame, **kwargs) -> None:
    """Logs a pandas Dataframe using a Lightning logger.

    Args:
        logger: Logger with which to log the dataframe.
        data: Dataframe to log.
        **kwargs: Additional parameters to pass along to the dataframe-logging function for the specific logger being
            used (e.g. Tensorboard, Comet, etc.).
    """
    match logger:
        case DummyLogger():
            pass
        case CometLogger():
            logger.experiment.log_table(tabular_data=data, **kwargs)
        case _:
            raise NotImplementedError(f"Logging dataframe not implemented for '{logger.__class__.__name__}' logger.")
