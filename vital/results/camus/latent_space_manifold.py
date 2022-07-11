from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import umap.plot

from vital.data.camus.config import CamusTags
from vital.results.camus.utils.data_struct import ViewResult
from vital.results.camus.utils.itertools import PatientViews
from vital.results.embedding import GroupsEmbeddingPlots


class LatentSpaceManifoldPlots(GroupsEmbeddingPlots):
    """Class that plots the UMAP embedding of a latent space manifold in a 2D space."""

    ResultsCollection = PatientViews

    def process_result(self, result: ViewResult) -> Tuple[str, "LatentSpaceManifoldPlots.ProcessingOutput"]:
        """Extracts latent space encodings from a view result.

        Args:
            result: Data structure holding latent space encodings for a view result.

        Returns:
            - Identifier of the patient view.
            - Latent space encodings of the data samples from the patient view.
        """
        return result.id, result[f"{CamusTags.pred}/{CamusTags.encoding}"].data

    def show_interactive_plot(  # noqa: D102
        self, mapper: umap.UMAP, labels: np.ndarray, hover_data: pd.DataFrame
    ) -> None:
        p = umap.plot.interactive(mapper, labels=labels, hover_data=hover_data)
        umap.plot.show(p)

    def save_points_plots(  # noqa: D102
        self, mapper: umap.UMAP, labels: np.ndarray, values: np.ndarray, output_folder: Path
    ) -> None:
        # TODO Fix too populated legend when labelling each patient views
        axis = umap.plot.points(mapper, labels=labels, color_key_cmap="prism")
        axis.figure.savefig(output_folder / "label_by_groups.png")

        axis = umap.plot.points(mapper, values=values, cmap="plasma")
        axis.figure.savefig(output_folder / "label_by_index_in_group.png")


def main():
    """Run the script."""
    LatentSpaceManifoldPlots.main()


if __name__ == "__main__":
    main()
