import csv
import logging
from pathlib import Path
from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks
from tqdm.auto import tqdm

from vital.data.cardinal.config import CardinalTag, TimeSeriesAttribute
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.utils.data_struct import View
from vital.data.cardinal.utils.itertools import Views
from vital.utils.format.native import squeeze
from vital.utils.logging import configure_logging
from vital.utils.parsing import int_or_float

logger = logging.getLogger(__name__)


def plot_sequence_peaks(
    sequence: np.ndarray,
    peaks: np.ndarray,
    cycle_peaks: np.ndarray = None,
    prominences: np.ndarray = None,
    pad_width: int = None,
    save_dir: Path = None,
) -> None:
    """Plots the sequence with the peaks, and optionally with their prominences clearly marked.

    Args:
        sequence: Sequence to plot.
        peaks: Indices of the peaks identified in the sequence (1st element of the pair returned by
            `scipy.signal.find_peaks`).
        cycle_peaks: If not all peaks in `peaks` were determined to mark the beginning/end of the cycle, this lists the
            indices of the peaks marking the beginning/end of the cycle. Therefore, this must an integer array usable
            as indices. The peaks marked by `cycle_peaks` are displayed in green, while other peaks that are not part
            of the cycle are displayed in red.
        prominences: Prominences of the peaks identified in the sequence (provided in the `properties` dictionary
            returned by `scipy.signal.find_peaks`).
        pad_width: Width of the padding at the beginning/end of the sequence to crop from the plot.
        save_dir: Directory to save the plot. If None, the plot is shown interactively instead of being saved.
    """
    import seaborn as sns
    from matplotlib import pyplot as plt

    sequence = sequence.copy()
    peaks = peaks.copy()
    if cycle_peaks is None:
        cycle_peaks = np.arange(len(peaks))
    else:
        cycle_peaks = cycle_peaks.copy()

    if save_dir:
        # Ensure that matplotlib is using 'agg' backend
        # to avoid possible leak of file handles if matplotlib defaults to another backend
        plt.switch_backend("agg")

    # Function that allows to inspect caller functions to add useful information to the plot titles
    def reach(name):
        import inspect

        for f in inspect.stack():
            if name in f[0].f_locals:
                return f[0].f_locals[name]
        return None

    view_id = "/".join(reach("view").id)

    if pad_width:
        sequence = sequence[pad_width:-pad_width]  # Remove the padding from the sequence
        peaks -= pad_width  # Correct the indices of the peaks to match the sequence with removed padding

        # Determine which identified peaks fall within the non-padded sequence
        peaks_in_sequence_mask = (0 <= peaks) & (peaks < len(sequence))

        # Remove the peaks that fall outside the sequence, and update the data pointing to specific peaks
        peaks = peaks[peaks_in_sequence_mask]
        cycle_peaks -= np.flatnonzero(peaks_in_sequence_mask)[0]
        if prominences is not None:
            prominences = prominences[peaks_in_sequence_mask]

    # Plot the sequence itself
    with sns.axes_style("darkgrid"):
        plot = sns.lineplot(data=sequence)
    plot.set(title=view_id, xlabel="frame", ylabel="val")

    # Plot the peaks
    cycle_peaks_mask = np.zeros(len(peaks), dtype=bool)
    cycle_peaks_mask[cycle_peaks] = True
    plot.scatter(peaks[cycle_peaks_mask], sequence[peaks[cycle_peaks_mask]], marker="^", color="green")
    if any(~cycle_peaks_mask):
        plot.scatter(peaks[~cycle_peaks_mask], sequence[peaks[~cycle_peaks_mask]], marker="^", color="red")

    # Annotate the peaks with the exact value of the frame
    for peak in peaks:
        plot.annotate(f"{peak}", (peak, sequence[peak]), xytext=(1, 4), textcoords="offset points")

    # Plot the prominences
    if prominences is not None:
        contour_heights = sequence[peaks] - prominences
        plot.vlines(x=peaks, ymin=contour_heights, ymax=sequence[peaks], linestyles="dashed")

    if save_dir:
        # Save the plot to disk
        plt.savefig(save_dir / (view_id.replace("/", "_") + ".png"))
    else:
        # Show the plot interactively
        plt.show()
    plt.close()  # Close the figure to avoid contamination between plots


def identify_cycle(
    sequence: np.ndarray,
    peaks_distance: Union[int, float, Tuple[float, int]] = (0.5, 20),
    peaks_height: Union[int, float] = 0.5,
    peaks_prominence: Union[int, float] = 0.15,
    debug_peaks_plots_dir: Path = None,
) -> Tuple[int, int]:
    """Identifies beginning/end frames of the cycle with the most similar beginning/end frames in the sequence.

    Notes:
        - The (ratio, upper limit) interpretation of `peaks_distance` is intended to help handle standard B-mode and
          Doppler sequences (with reduced frame rates compared to B-mode) at the same time. On one hand, the ratio
          ensures that the distance between peaks can be reduced for short sequences, i.e. sequences with poor temporal
          resolution like Doppler ones, to still allow two peaks in a sequence. On the other hand, the upper limit
          is there to skipping over cycles entirely for long sequences covering multiple cycles.

    Args:
        sequence: Cyclical signal in which to identify peaks delimiting the most "cyclical" cycle in the sequence.
        peaks_distance: Minimum distance between peaks in the sequence for it to be considered a cycle. If an integer,
            this is the absolute minimum number of frames. If a float, this is the fraction of the total number of
            frames in the sequence. If a tuple, the first element is the fraction of the total number of frames in the
            sequence, and the second element is an upper limit on the minimum number of frames (above which the ratio
            is disregarded and the upper limit is used instead).
        peaks_height: Minimum height of a peak to be considered as a beginning/end frame. If an integer, this is the
            absolute height. If a float, this is the fraction of the difference between the sequence's highest peak
            and lowest valley.
        peaks_prominence: Minimum prominence of a peak to be considered as a beginning/end frame. If an integer, this is
            the absolute prominence. If a float, this is the fraction of the difference between the sequence's highest
            peak and lowest valley.
        debug_peaks_plots_dir: Directory where to save plots of the peaks found in each sequence, meant for debugging
            the cycle finding algorithm. No information to introspect the behavior of the algorithm is displayed/saved.

    Returns:
        Indices of the beginning/end frames of the cycle with the most similar beginning/end frames in the sequence.
    """
    min_max_diff = np.max(sequence) - np.min(sequence)
    num_frames = len(sequence)

    # Determine the (absolute) distance between peaks for both of them as beginning/end frames
    if isinstance(peaks_distance, float):
        peaks_distance = int(len(sequence) * peaks_distance)
    elif isinstance(peaks_distance, tuple):
        ratio, upper_limit = peaks_distance
        peaks_distance = min(int(len(sequence) * ratio), upper_limit)

    # Determine the minimum (absolute) value of a peak to be considered as a beginning/end frame
    if isinstance(peaks_height, float):
        peaks_height = np.min(sequence) + (peaks_height * min_max_diff)

    # Determine the minimum (absolute) prominence of a peak to be considered as a beginning/end frame
    if isinstance(peaks_prominence, float):
        peaks_prominence = min_max_diff * peaks_prominence

    # Mirror pad the sequence to detect enable peaks at the very beginning of the sequences,
    # even if the value is already decreasing at the start
    # Pad width is also generous to allow for the prominence to remain a relevant criterion
    pad_width = peaks_distance
    sequence = np.pad(sequence, pad_width, mode="reflect")
    # Slightly decrease the padded regions to avoid favoring peaks there  over peaks in the original sequence
    sequence[:pad_width] -= 1
    sequence[-pad_width:] -= 1

    # Estimate the beginning/end frames of cycles in the sequence by finding the peaks in the sequence
    peaks, properties = find_peaks(sequence, height=peaks_height, distance=peaks_distance, prominence=peaks_prominence)
    plot_sequence_kwargs = {
        "sequence": sequence,
        "peaks": peaks,
        "prominences": properties["prominences"],
        "pad_width": pad_width,
        "save_dir": debug_peaks_plots_dir,
    }

    # Determine which identified peaks fall within the non-padded sequence
    peaks_in_sequence_mask = (pad_width <= peaks) & (peaks < (num_frames + pad_width))

    if sum(peaks_in_sequence_mask) <= 1:
        if debug_peaks_plots_dir:
            plot_sequence_peaks(**plot_sequence_kwargs, cycle_peaks=np.array([], dtype=int))
        raise RuntimeError(
            "Unable to identify both the beginning and end of a cycle in the provided sequence. The provided sequence "
            "might not contain a full cardiac cycle."
        )

    # Select the consecutive peaks with the smallest difference between them as the cycle to extract
    num_peaks_in_begin_pad = np.flatnonzero(peaks_in_sequence_mask)[0]
    peaks_in_sequence_diffs = np.abs(np.diff(sequence[peaks[peaks_in_sequence_mask]]))
    min_diff_idx = np.argmin(peaks_in_sequence_diffs) + num_peaks_in_begin_pad

    if debug_peaks_plots_dir:
        plot_sequence_peaks(**plot_sequence_kwargs, cycle_peaks=np.array([min_diff_idx, min_diff_idx + 1]))

    # Determine the frames of the selected peaks in the original, non-padded sequence
    peaks -= pad_width
    return peaks[min_diff_idx], peaks[min_diff_idx + 1]


def clip_view(
    view: View,
    mask_tag: str,
    identify_cycle_fn: Callable[[np.ndarray], Tuple[int, int]] = None,
    clip_frames: Tuple[int, int] = None,
    return_clip_frames: bool = False,
) -> Union[View, Tuple[View, Tuple[int, int]]]:
    """Clips a view so that it contains the data of exactly one full cardiac cycle.

    Args:
        view: View data and metadata.
        mask_tag: Tag of the segmentation mask to use to identify the beginning/end of the cycle.
        identify_cycle_fn: Function to use to identify the beginning/end of the cycle. Takes as input a curve in which
            to identify peaks (i.e. the LV area curve) and outputs the frames corresponding to the beginning and end of
            a cycle. Optional if pre-determined `clip_frames` are provided.
        clip_frames: Pre-determined frames at which to clip the beginning and end of the cycle. When provided, this
            by-passes calling `identify_cycle_fn`.
        return_clip_frames: Return the first and last frames at which the sequences of the view were clipped, along
            with the clipped view.

    Returns:
        View with its temporal data and metadata arrays clipped to exactly one full cardiac cycle.
        If `return_clip_frames` is True, then also the start and stop frames at which the sequences of the view were
        clipped.
    """
    if clip_frames is None and identify_cycle_fn is None:
        raise ValueError("If you do not provide ")

    if mask_tag not in view.data:
        patient_id, view_tag = view.id
        raise RuntimeError(
            f"Cannot identify start and end of cardiac cycle in view '{view_tag}' from patient '{patient_id}' because "
            f"no segmentation mask was provided for the view."
        )
    lv_area_curve = np.squeeze(view.attrs[mask_tag][TimeSeriesAttribute.lv_area])

    if clip_frames:
        # Use the pre-identified frames to clip the sequence, if provided
        ed_1_idx, ed_2_idx = clip_frames
    else:
        # Automatically detect frames which mark the start/end of the cycle
        ed_1_idx, ed_2_idx = identify_cycle_fn(lv_area_curve)

    # Iterate over each item available for the view, and clip its length if it has a time dimension
    clipped_data = {tag: data[ed_1_idx : ed_2_idx + 1] for tag, data in view.data.items()}
    clipped_attrs = {
        tag: {
            attr: data[ed_1_idx : ed_2_idx + 1] if attr in TimeSeriesAttribute else data for attr, data in attrs.items()
        }
        for tag, attrs in view.attrs.items()
    }

    # Build a new `View` object for the clipped data
    clipped_view = View(id=view.id, data=clipped_data, attrs=clipped_attrs)

    if return_clip_frames:
        return clipped_view, (ed_1_idx, ed_2_idx)
    else:
        return clipped_view


def main():
    """Run the script."""
    from argparse import ArgumentParser

    configure_logging(log_to_console=True, console_level=logging.INFO)

    parser = ArgumentParser()
    parser = Views.add_args(parser)
    parser.add_argument(
        "--mask_tag",
        type=str,
        default=CardinalTag.mask,
        help="Tag of the segmentation mask to use to identify the beginning/end of the cycle",
    )
    parser.add_argument(
        "--pre_clip_frames",
        type=Path,
        help="YAML file containing pre-determined frames at which to clip some of the sequences. For these sequences, "
        "providing the frames where to clip will by-pass the algorithm used to determine the beginning/end frames of "
        "the cycle.",
    )
    parser.add_argument(
        "--peaks_distance",
        type=int_or_float,
        nargs="+",
        default=(0.5, 20),
        help="Minimum distance between peaks in the LV area curve for it to be considered a cycle. If an integer, this "
        "is the absolute minimum number of frames. If a float, this is the fraction of the total number of frames in "
        "the sequence. If a tuple, the first element is the fraction of the total number of frames in the sequence, "
        "and the second element is an upper limit on the minimum number of frames (above which the ratio is "
        "disregarded and the upper limit is used instead).",
    )
    parser.add_argument(
        "--peaks_height",
        type=int_or_float,
        default=0.5,
        help="Minimum height of a peak in the LV area curve to be considered as a beginning/end frame. If an integer, "
        "this is the absolute height. If a float, this is the fraction of the difference between the sequence's "
        "highest peak and lowest valley.",
    )
    parser.add_argument(
        "--peaks_prominence",
        type=int_or_float,
        default=0.15,
        help="Minimum prominence of a peak in the LV area curve to be considered as a beginning/end frame. If an "
        "integer, this is the absolute prominence. If a float, this is the fraction of the difference between the "
        "sequence's highest peak and lowest valley.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("full_cycles"),
        help="Root directory under which to save the sequences clipped at one full cardiac cycle",
    )
    parser.add_argument(
        "--clip_frames_log",
        type=Path,
        help="CSV file where to log the frames at which each view was clipped, to help debug the view clipping "
        "algorithm",
    )
    parser.add_argument(
        "--debug_peaks_plots_dir",
        type=Path,
        help="Directory where to save plots of the peaks found in each sequence, meant for debugging the cycle finding "
        "algorithm",
    )
    args = parser.parse_args()
    kwargs = vars(args)
    mask_tag, pre_clip_frames, output_dir, clip_log, debug_peaks_plots_dir = (
        kwargs.pop("mask_tag"),
        kwargs.pop("pre_clip_frames"),
        kwargs.pop("output_dir"),
        kwargs.pop("clip_frames_log"),
        kwargs.pop("debug_peaks_plots_dir"),
    )
    peaks_distance, peaks_height, peaks_prominence = (
        kwargs.pop("peaks_distance"),
        kwargs.pop("peaks_height"),
        kwargs.pop("peaks_prominence"),
    )

    # Check that the format of `peaks_distance` is valid
    peaks_distance = squeeze(tuple(peaks_distance))
    if len(peaks_distance) > 2:
        raise ValueError(
            f"`peaks_distance` must be either a single int/float, or a tuple of length 2, but got {peaks_distance}"
        )

    if pre_clip_frames:
        # Read the clip
        with open(pre_clip_frames) as f:
            pre_clip_frames = yaml.safe_load(f)

        # Format the resulting dictionary from views nested in patients to `View.Id` keys
        pre_clip_frames = {
            View.Id(patient_id, ViewEnum[view]): indices
            for patient_id, views in pre_clip_frames.items()
            for view, indices in views.items()
        }
    else:
        pre_clip_frames = {}

    # Check that the output directories exists
    output_dir.mkdir(parents=True, exist_ok=True)
    if clip_log:
        clip_log.parent.mkdir(parents=True, exist_ok=True)
    if debug_peaks_plots_dir:
        debug_peaks_plots_dir.mkdir(parents=True, exist_ok=True)

    # Clip the views
    views = Views(**kwargs)
    clip_frames = {}
    for view_id, view in tqdm(
        views.items(), desc="Clipping views to only contain one full cardiac cycle", unit=views.desc
    ):
        ed_1_idx, ed_2_idx = None, None
        try:
            clipped_view, (ed_1_idx, ed_2_idx) = clip_view(
                view,
                mask_tag,
                lambda x: identify_cycle(
                    x,
                    peaks_distance=peaks_distance,
                    peaks_height=peaks_height,
                    peaks_prominence=peaks_prominence,
                    debug_peaks_plots_dir=debug_peaks_plots_dir,
                ),
                clip_frames=pre_clip_frames.get(view_id),
                return_clip_frames=True,
            )
            clipped_view.save(output_dir, subdir_levels=["patient"])
        except Exception:
            patient_id, view_tag = view_id
            logger.exception(
                f"Failed to clip '{view_tag}' view from patient '{patient_id}' because of the following exception:"
            )
        finally:
            # Accumulate the indices at which the views were clipped
            clip_frames[view_id] = {"ED1": ed_1_idx, "ED2": ed_2_idx}

    # Write the indices at which the views were clipped to a single CSV file
    if clip_log:
        # Force 'Int64' dtype to avoid missing entries from converting the indices to floats
        clip_frames = pd.DataFrame.from_dict(clip_frames, orient="index", dtype="Int64")
        clip_frames = clip_frames.rename_axis(["patient", "view"])

        # Compute statistics of missing indices (denoting either failures of the algorithm or bad quality data)
        cycles_na = clip_frames.isna()[["ED1"]].rename(columns={"ED1": "cycles n/a"})
        views_na_stats = cycles_na.groupby(level="view").sum().astype("Int64")
        # Order the views' statistics according to the views' usual ordering
        views_na_stats = views_na_stats.reindex([view for view in ViewEnum if view in views_na_stats.index])
        patients_na_stats = cycles_na.groupby(level="patient").any().sum().astype("Int64")
        patients_na_stats = pd.DataFrame.from_dict({"patients": patients_na_stats}, orient="index")

        cycles_na_stats = pd.concat([views_na_stats, patients_na_stats])
        cycles_na_stats.to_csv(clip_log, quoting=csv.QUOTE_NONNUMERIC)
        with clip_log.open("a") as f:
            f.write("\n")
        clip_frames.to_csv(clip_log, mode="a", quoting=csv.QUOTE_NONNUMERIC, na_rep="n/a")


if __name__ == "__main__":
    main()
