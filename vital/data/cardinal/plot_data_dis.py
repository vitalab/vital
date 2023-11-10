from pathlib import Path

from matplotlib import pyplot as plt

from vital.data.cardinal.config import TabularAttribute
from vital.data.cardinal.utils.data_dis import plot_patients_distribution
from vital.data.cardinal.utils.itertools import Patients


def main():
    """Run the script."""
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = Patients.add_args(parser)
    parser.add_argument(
        "--plot_attributes",
        type=TabularAttribute,
        nargs="+",
        choices=list(TabularAttribute),
        default=list(TabularAttribute),
        help="Patients' tabular attributes whose distributions to compare pairwise",
    )
    parser.add_argument(
        "--subsets",
        type=Path,
        nargs="+",
        help="Path to plain-text files listing subsets of patients to plot with different hues",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    subsets, plot_attributes = kwargs.pop("subsets"), kwargs.pop("plot_attributes")

    # Read the lists of patients in each subset from their respective files
    if subsets:
        subsets = {subset_file.stem: subset_file.read_text().splitlines() for subset_file in subsets}

    # Plot the distribution of the requested attributes, either across all patients or for each subset
    _ = plot_patients_distribution(Patients(**kwargs), plot_attributes, subsets=subsets, progress_bar=True)
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
