from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sklearn.decomposition import PCA


def pca_analysis(
    pca: PCA,
    samples: pd.DataFrame,
    sweep_coeffs: Dict[str, float],
    max_n_components: int = None,
    sweep_plots_kwargs: Dict[str, Any] = None,
    embedding_plots_kwargs: Dict[str, Any] = None,
    output_dir: Path = Path.cwd(),
) -> None:
    """Wrapper around multiple PCA analysis functions that calls them and saves the plots they produce.

     Functions called internally:
        - `analyze_pca_wrt_n_components`
        - `sweep_pca_dims`
        - `visualize_embedding_pairwise_dims`

    Args:
        pca: PCA sklearn estimator that has already been fitted to some data.
        samples: (N, W), Data samples on which to analyze the performance of the PCA.
        sweep_coeffs: Coefficients by which to multiply the diff between sweeps, in our case the stddev, when
            sweeping the PCA dimensions.
        max_n_components: Maximum number of components with which to perform PCA. If not provided, or if too high,
            the value will be determined from the `samples`, as the minimum between N and W.
        sweep_plots_kwargs: Parameters passed along to customize the PCA dimensions' sweeping plots.
        embedding_plots_kwargs: Parameter passed along to customize the PCA embedding's plots.
        output_dir: Root directory where to save the generated plots.
    """
    if sweep_plots_kwargs is None:
        sweep_plots_kwargs = {}

    samples_embedding = pd.DataFrame(pca.transform(samples), index=samples.index)

    def _save_cur_fig(title: str, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        title_pathified = title.lower().replace("/", "_").replace(" ", "_")
        plt.savefig(folder / f"{title_pathified}.png")
        plt.close()  # Close the figure to avoid contamination between plots

    # Analyze PCA variance explanation
    title, plot = analyze_pca_wrt_n_components(pca, max_n_components=max_n_components)
    _save_cur_fig(title, output_dir)

    # Plot reconstructions of average samples +/- some coefficient of stddev, for each PCA dimension
    for title, plot in sweep_embedding_dims(
        pca.inverse_transform, samples_embedding.to_numpy(), sweep_coeffs, plots_kwargs=sweep_plots_kwargs
    ):
        _save_cur_fig(title, output_dir / "sweep")

    # Visualize embedding by plotting scatter plots over pairs of dimensions at a time
    for title, plot in visualize_embedding_pairwise_dims(
        samples_embedding, max_embedding_dim=max_n_components, plots_kwargs=embedding_plots_kwargs
    ):
        _save_cur_fig(title, output_dir / "embedding")


def analyze_pca_wrt_n_components(pca: PCA, max_n_components: int = None) -> Tuple[str, Axes]:
    """Analyzes the variance explanation of a PCA model on its training data w.r.t. `n_components`.

    Args:
        pca: PCA sklearn estimator that has already been fitted to some data.
        max_n_components: Number of components at which to stop the analysis. If not provided, defaults to analyzing
            all the components.

    Returns:
        A title and plot of the variance explanation of the PCA model on its training data w.r.t. `n_components`.
    """
    if max_n_components is None:
        max_n_components = pca.n_components_

    # Plot explained variance w.r.t. PCA components
    title = "explained_variance_ratio w.r.t. n_components"
    with sns.axes_style("darkgrid"):
        lineplot = sns.lineplot(
            x=np.arange(1, max_n_components + 1), y=np.cumsum(pca.explained_variance_ratio_[:max_n_components])
        )
    lineplot.set(title=title, xlabel="n_components", ylabel="explained_variance_ratio")

    return title, lineplot


def sweep_embedding_dims(
    decode_fn: Callable[[np.ndarray], np.ndarray],
    samples_embedding: np.ndarray,
    sweep_coeffs: Dict[str, float],
    plots_kwargs: Dict[str, Any],
) -> Iterator[Tuple[str, Axes]]:
    """Produces reconstruction plots of average samples +/- some coefficient of stddev, for each PCA dimension.

    Args:
        decode_fn: Function that takes as input batches of embeddings and decodes them to their original value.
        samples_embedding: (N, M), Samples that have been compressed by PCA. These will serve to compute global
            statistics from which to reconstruct representative samples of the data.
        sweep_coeffs: Coefficients by which to multiply the diff between sweeps, in our case the stddev, when
            sweeping the PCA dimensions.
        plots_kwargs: Parameters passed along to the plots' axes' `set` method to customize them.

    Returns:
        An iterator over titles and plots of reconstruction sweeps for each PCA dimension.
    """
    if plots_kwargs is None:
        plots_kwargs = {}

    mean_sample, stddevs = np.mean(samples_embedding, axis=0), np.std(samples_embedding, axis=0)
    for dim, (dim_val, dim_stddev) in enumerate(zip(mean_sample, stddevs)):
        # Sweep the dimension and reconstruct the sample corresponding to the manipulated encoding
        dim_sweep_reconstructions = {"μ": decode_fn(mean_sample[None, :])[0]}
        for sweep_tag, coeff in sweep_coeffs.items():
            mean_sample[dim] = dim_val + (coeff * dim_stddev)
            dim_sweep_reconstructions[sweep_tag] = decode_fn(mean_sample[None, :])[0]

        # Convert the reconstructions to a dataframe
        time_index = np.linspace(0, 1, num=len(dim_sweep_reconstructions["μ"]))
        dim_sweep_df = (
            pd.concat(
                {
                    sweep_tag: pd.DataFrame({"time": time_index, "val": attr_data})
                    for sweep_tag, attr_data in dim_sweep_reconstructions.items()
                }
            )
            .rename_axis(["sweep", None])
            .reset_index(0)
            .reset_index(drop=True)
        )

        # Plot the sweep
        title = f"sweep_dim={dim}"
        with sns.axes_style("darkgrid"):
            lineplot = sns.lineplot(data=dim_sweep_df, x="time", y="val", style="sweep")
        lineplot.set(title=title, **plots_kwargs)

        yield title, lineplot

        # Restore the original value in the sweeped dimension before moving on to the next dimension
        mean_sample[dim] = dim_val


def visualize_embedding_pairwise_dims(
    samples_embedding: pd.DataFrame, max_embedding_dim: int = None, plots_kwargs: Dict[str, Any] = None
) -> Iterator[Tuple[str, Axes]]:
    """Produces scatter plots of samples over pairs of dimensions at a time, to visualize the embedding.

    Args:
        samples_embedding: (N, M), Samples that have been compressed. The index will reset as columns internally to use
            its info to style the plot.
        max_embedding_dim: Embedding dimension at which to stop plotting. If not provided, defaults to plot as many
            pairs of dimensions as possible.
        plots_kwargs: Parameters passed along to the plots' axes' `set` method to customize them.

    Returns:
        An iterator over titles and scatter plots of samples over pairs of dimensions at a time.
    """
    if plots_kwargs is None:
        plots_kwargs = {}

    if max_embedding_dim is None:
        max_embedding_dim = samples_embedding.shape[-1]

    # Move indexing information to columns to make it available for plotting
    samples_embedding = samples_embedding.reset_index()
    for dim in range(0, max_embedding_dim, 2):
        title = f"Dims {dim}/{dim + 1}"
        with sns.axes_style("darkgrid"):
            scatterplot = sns.scatterplot(data=samples_embedding, x=dim, y=dim + 1, **plots_kwargs)
        scatterplot.set(title=title)

        yield title, scatterplot
