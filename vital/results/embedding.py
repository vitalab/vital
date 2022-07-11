import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from vital.results.processor import ResultsProcessor
from vital.utils.parsing import StoreDictKeyPair

logger = logging.getLogger(__name__)


class GroupsEmbeddingPlots(ResultsProcessor):
    """Abstract class that plots the UMAP embedding of groups of results in a 2D space."""

    desc = "groups_embedding"
    ProcessingOutput = np.ndarray

    def __init__(self, embedding_kwargs: Mapping[str, Any] = None, interactive: bool = False, **kwargs):
        """Initializes class instance.

        Args:
            embedding_kwargs: Parameters to pass to the UMAP estimator.
            interactive: If ``True``, use the interactive plot; otherwise, save a custom set of figures for later
                viewing.
        """
        super().__init__(output_name=self.desc, **kwargs)
        embedding_kwargs = embedding_kwargs if embedding_kwargs else {}
        self.umap = umap.UMAP(**embedding_kwargs)
        self.interactive = interactive

    def aggregate_outputs(self, outputs: Mapping[str, ProcessingOutput], output_path: Path) -> None:
        """Embeds the gathered high-dimensional results using UMAP and plots the generated embedding.

        Args:
            outputs: Mapping between each result in the results collection and its data to embed.
            output_path: Folder in which to save the plotted embeddings (if not in interactive mode).
        """
        #  Formats data required to plot the groups' distribution in high dimensional latent space.
        labels, values, encodings = [], [], []
        indices_in_groups = []
        for group_id, encoding in outputs.items():
            labels.append([group_id] * len(encoding))  # Uniform label for all samples in the group
            values.append(np.linspace(0, 1, num=len(encoding)))  # Normalized index of each sample in the group
            encodings.append(encoding)  # High dimensional samples to embed
            indices_in_groups.append(np.arange(len(encoding)))  # index of each sample in the group

        # Convert lists to arrays, since umap.plot requires arrays
        labels = np.hstack(labels)
        values = np.hstack(values)
        encodings = np.vstack(encodings)
        indices_in_groups = np.hstack(indices_in_groups)

        logger.info("Generating UMAP embedding for results ...")
        mapper = self.umap.fit(encodings)

        # Ensure that matplotlib is using 'agg' backend
        # to avoid possible 'Could not connect to any X display' errors
        # when no X server is available, e.g. in remote terminal
        plt.switch_backend("agg")

        if self.interactive:
            # Format information for hover tooltips in the interactive plot
            hover_data = pd.DataFrame({"group_id": labels, "index_in_group": indices_in_groups})

            self.show_interactive_plot(mapper, labels, hover_data)

        else:
            self.save_points_plots(mapper, labels, values, output_path)
            plt.close("all")

    def show_interactive_plot(self, mapper: umap.UMAP, labels: np.ndarray, hover_data: pd.DataFrame) -> None:
        """Plots an interactive view of the UMAP embedding.

        Args:
            mapper: Trained UMAP object that has a 2D embedding.
            labels: Labels (assumed integer or categorical) for each data sample.
                See ``umap.plot.interactive`` for more info.
            hover_data: Tooltip data. Each column will be used as information within the tooltip.
                ``See umap.plot.interactive for more info``.
        """
        raise NotImplementedError

    def save_points_plots(self, mapper: umap.UMAP, labels: np.ndarray, values: np.ndarray, output_folder: Path) -> None:
        """Saves plots of the UMAP embedding (using possibly different coloring schemes) as images.

        Args:
            mapper: Trained UMAP object that has a 2D embedding.
            labels: Labels (assumed integer or categorical) for each data sample.
                See ``umap.plot.points`` for more info.
            values: Values (assumed float or continuous) for each data sample.
                See ``umap.plot.points`` for more info.
            output_folder: Folder in which to save the plotted embeddings.
        """
        raise NotImplementedError

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser with support for generic groups embedding and results collection arguments.

        Returns:
            Parser object with support for generic groups embedding and results collection arguments.
        """
        parser = super().build_parser()
        parser.add_argument(
            "--interactive", action="store_true", help="Enable UMAP interactive plot, instead of saving scatter plots"
        )
        parser.add_argument(
            "--embedding_kwargs",
            action=StoreDictKeyPair,
            default=dict(),
            help="Parameters to pass to the UMAP estimator",
        )
        return parser
