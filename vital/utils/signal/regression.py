import numpy as np
from sklearn.kernel_ridge import KernelRidge

from vital.utils.decorators import wrap_pad


@wrap_pad()
def kernel_ridge_regression(
    signal: np.ndarray, kernel: str = "linear", alpha: float = 1, **kernel_kwargs
) -> np.ndarray:
    """Learns a kernel ridge regression between 1D feature and target values ``y``.

    Args:
        signal: Target values of the regression.
        kernel: Name of the kernel to use for the kernel ridge regression.
        alpha: Regularization strength for the kernel ridge regression.
        kernel_kwargs: Additional parameters for the kernel function.

    Returns:
        The values predicted by the fitted kernel ridge regression.
    """
    krr_kwargs = {}
    if (gamma := kernel_kwargs.pop("gamma", None)) is not None:
        krr_kwargs["gamma"] = gamma
    if (degree := kernel_kwargs.pop("degree", None)) is not None:
        krr_kwargs["degree"] = degree
    if (coef0 := kernel_kwargs.pop("coef0", None)) is not None:
        krr_kwargs["coef0"] = coef0
    feat = np.linspace(0, 1, len(signal))[:, np.newaxis]
    krr = KernelRidge(alpha=alpha, kernel=kernel, **krr_kwargs, kernel_params=kernel_kwargs)
    krr.fit(feat, signal)
    fit = krr.predict(feat).reshape(signal.shape)

    return fit
