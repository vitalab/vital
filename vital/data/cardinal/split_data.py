from pathlib import Path

from vital.data.cardinal.config import TabularAttribute
from vital.data.cardinal.utils.data_dis import generate_patients_splits
from vital.data.cardinal.utils.itertools import Patients
from vital.utils.parsing import int_or_float


def main():
    """Run the script."""
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = Patients.add_args(parser)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where to save the files listing the patients making up each subset",
    )
    parser.add_argument(
        "--train_name", type=str, default="train", help="Name to give to the file for the training subset"
    )
    parser.add_argument("--test_name", type=str, default="test", help="Name to give to the file for the test subset")
    parser.add_argument(
        "--stratify_attr",
        required=True,
        type=TabularAttribute,
        choices=list(TabularAttribute),
        help="Name of the tabular attribute whose distribution in each of the subset should be similar",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=5,
        help="If `stratify_attr` is a continuous attribute, number of bins into which to categorize the values, to "
        "ensure each bin is distributed representatively in the split.",
    )
    parser.add_argument(
        "--test_size",
        type=int_or_float,
        default=0.2,
        help="If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the "
        "test split. If int, represents the absolute number of test samples.",
    )
    parser.add_argument(
        "--seed", type=int, help="Seed to control the shuffling applied to the data before applying the split"
    )
    args = parser.parse_args()
    kwargs = vars(args)

    output_dir, train_name, test_name, stratify_attr, bins, test_size, seed = (
        kwargs.pop("output_dir"),
        kwargs.pop("train_name"),
        kwargs.pop("test_name"),
        kwargs.pop("stratify_attr"),
        kwargs.pop("bins"),
        kwargs.pop("test_size"),
        kwargs.pop("seed"),
    )

    patient_ids_train, patient_ids_test = generate_patients_splits(
        Patients(**kwargs), stratify_attr, bins=bins, test_size=test_size, seed=seed, progress_bar=True
    )

    # Save the generated split
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{train_name}.txt").write_text("\n".join(patient_ids_train))
    (output_dir / f"{test_name}.txt").write_text("\n".join(patient_ids_test))


if __name__ == "__main__":
    main()
