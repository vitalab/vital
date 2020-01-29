from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import umap

from vital.logs.logger import Logger
from vital.utils.delegate import delegate_inheritance


@delegate_inheritance()
class GroupsEmbeddingLogger(Logger):
    """Abstract class that allows to visualize the UMAP embedding of groups of results in a 2D space."""
    desc = 'groups_embedding'
    Log = np.ndarray

    def __init__(self, embedding_params: Dict[str, Any] = None, interactive: bool = False, **kwargs):
        """
        Args:
            embedding_params: parameters to initialize the UMAP object.
            interactive: whether to use interactive plot (``True``) or save a custom set of figures for later viewing
                        (``False``).
        """
        super().__init__(output_name_template="{}", **kwargs)
        embedding_params = embedding_params if embedding_params else {}
        self.umap = umap.UMAP(**embedding_params)
        self.interactive = interactive

    def aggregate_logs(self, logs: Dict[str, Log], output_path: Path):
        """ Embeds the high-dimensional gathered from the results using UMAP and plots the generated embedding.

        Args:
            logs: mapping between each result in the iterable results and its data to embed.
            output_path: the folder in which to save the plotted embeddings (if not in interactive mode).
        """
        #  Formats data required to plot the groups' distribution in high dimensional latent space.
        labels, values, encodings = [], [], []
        indices_in_groups = []
        for group_id, encoding in logs.items():
            labels.append([group_id] * len(encoding))  # Uniform label for all samples in the group
            values.append(np.linspace(0, 1, num=len(encoding)))  # Normalized index of each sample in the group
            encodings.append(encoding)  # High dimensional samples to embed
            indices_in_groups.append(np.arange(len(encoding)))  # index of each sample in the group

        # Convert lists to arrays, since umap.plot requires arrays
        labels = np.hstack(labels)
        values = np.hstack(values)
        encodings = np.vstack(encodings)
        indices_in_groups = np.hstack(indices_in_groups)

        print(f"Generating UMAP embedding for results ...")
        mapper = self.umap.fit(encodings)

        if self.interactive:
            # Format information for hover tooltips in the interactive plot
            hover_data = pd.DataFrame({'group_id': labels,
                                       'index_in_group': indices_in_groups})

            self.show_interactive_plot(mapper, labels, hover_data)

        else:
            output_path.mkdir(parents=True, exist_ok=True)
            self.save_points_plots(mapper, labels, values, output_path)

    def show_interactive_plot(self, mapper: umap.UMAP, labels: np.ndarray, hover_data: pd.DataFrame):
        """ Plots an interactive view of the UMAP embedding.

        Args:
            mapper: a trained UMAP object that has a 2D embedding.
            labels: labels (assumed integer or categorical) for each data sample.
                    See umap.plot.interactive for more info.
            hover_data: tooltip data. Each column will be used as information within the tooltip.
                        See umap.plot.interactive for more info.
        """
        raise NotImplementedError

    def save_points_plots(self, mapper: umap.UMAP, labels: np.ndarray, values: np.ndarray, output_folder: Path):
        """ Saves plots of the UMAP embedding (using possibly many different coloring schemes) as images.

        Args:
            mapper: a trained UMAP object that has a 2D embedding.
            labels: labels (assumed integer or categorical) for each data sample.
                    See umap.plot.points for more info.
            values: values (assumed float or continuous) for each data sample.
                    See umap.plot.points for more info.
            output_folder: the folder in which to save the plotted embeddings.
        """
        raise NotImplementedError

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """ Creates parser with support for generic groups embedding and iterable logger arguments.

        Returns:
          parser object with support for generic groups embedding and iterable logger arguments.
        """
        parser = super().build_parser()
        parser.add_argument("--interactive", action='store_true',
                            help="Enable UMAP interactive plot, instead of saving scatter plots")
        # TODO Add CL arguments for UMAP init
        return parser
