import logging
from typing import Any, Dict, Iterable, Iterator, Union

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


def embedding_scatterplot(
    data: pd.DataFrame, plots_kwargs: Iterable[Dict[str, Any]], umap_kwargs: Dict[str, Any] = None, data_tag: str = None
) -> Iterator[Axes]:
    """Generates 2D scatter plots of some data, reducing its dimensionality to 2 using UMAP if it's not already 2D.

    Args:
        data: Dataframe with each column representing a dimension of the data, and relevant metadata being stored in a
            multiindex.
        plots_kwargs: Sets of kwargs to use to generate different versions of the scatter plot, e.g. modifying the
            variables used for hue and/or style.
        umap_kwargs: If the data has more than 2 dimensions, UMAP is used to reduce the dimensionality of the data for
            plotting purposes. This parameter is passed along to the UMAP estimator's `init`.
        data_tag: String describing the data used in the titles/logs, etc. If not specified, it defaults to 'data'.

    Returns:
        An iterator over the generated scatter plots.
    """
    if not data_tag:
        data_tag = "data"

    # Make sure the encodings are 2D
    if len(data.columns) == 1:
        raise ValueError(
            "Plotting scatter plots of some data distribution is only supported for data of dimensionality > 1."
        )
    elif len(data.columns) == 2:
        plot_title = f"2D {data_tag}"
    else:  # len(encoding_dims) > 2
        if umap_kwargs is None:
            umap_kwargs = {}
        plot_title = f"2D UMAP embedding of the {len(data.columns)}D {data_tag}"
        logger.info(f"Generating 2D UMAP embedding of {len(data.columns)}D {data_tag}...")
        umap_embedding = umap.UMAP(**umap_kwargs).fit_transform(data)

        # Update the encodings dataframe with the new UMAP embedding
        data = data.drop(labels=data.columns, axis="columns")
        data[[0, 1]] = umap_embedding

    # Generate a plot of the embedding for each set of plot kwargs provided
    for plot_kwargs in plots_kwargs:
        with sns.axes_style("darkgrid"):
            scatterplot = sns.scatterplot(data=data, x=0, y=1, **plot_kwargs)
        scatterplot.set(title=plot_title)

        yield scatterplot


def plot_heatmap(
    heatmap: Union[np.ndarray, pd.DataFrame], rescale_above_n_elems: int = 10, cmap="viridis", fmt=".2f", annot_kws=None
) -> Axes:
    """Plots a heatmap, automatically scaling dimensions so that the labels remain readable.

    Args:
        heatmap: Heatmap data.
        rescale_above_n_elems: If the number of rows/columns in the heatmap is higher than this threshold, the size of
            the figure along that dimension is scaled so that the tick labels and annotations become visibly smaller,
            instead of overlapping and becoming unreadable.
        cmap: Colormap to pass along to `seaborn`'s `heatmap` function.
        fmt: String formatting passed for annotations passed along to `seaborn`'s `heatmap` function.
        annot_kws: Additional annotation keyword arguments to pass along to `seaborn`'s `heatmap` function.

    Returns:
        The generated heatmap.
    """
    default_figsize = matplotlib.rcParams["figure.figsize"]
    x_scaling = max(1.0, heatmap.shape[1] / rescale_above_n_elems)
    y_scaling = max(1.0, heatmap.shape[0] / rescale_above_n_elems)
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(figsize=(default_figsize[0] * x_scaling, default_figsize[1] * y_scaling))
        sns.heatmap(data=heatmap, annot=True, annot_kws=annot_kws, fmt=fmt, square=True, cmap=cmap, ax=ax)
    return ax
