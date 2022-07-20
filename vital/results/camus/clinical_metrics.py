import logging
from typing import Optional, Tuple, Union

import numpy as np

from vital.data.camus.config import CamusTags, Instant, Label, View
from vital.metrics.camus.clinical.utils import compute_clinical_metrics_by_patient
from vital.results.camus.utils.data_struct import PatientResult, ViewResult
from vital.results.camus.utils.itertools import Patients
from vital.results.metrics import Metrics

logger = logging.getLogger(__name__)


class ClinicalMetrics(Metrics):
    """Class that computes clinical metrics on the results and saves them to csv."""

    desc = "clinical_metrics"
    ResultsCollection = Patients
    input_choices = [f"{CamusTags.pred}/{CamusTags.raw}", f"{CamusTags.pred}/{CamusTags.post}"]
    target_choices = [f"{CamusTags.gt}/{CamusTags.raw}"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._required_views = (View.A2C, View.A4C)
        self._required_instants = (Instant.ED, Instant.ES)

    def _check_and_extract_data(
        self, result: PatientResult
    ) -> Optional[Union[ViewResult, Tuple[ViewResult, ViewResult]]]:
        """Extracts relevant data to compute clinical metrics, from what is available in the result.

        Args:
            result: Data structure holding all the relevant information to compute the requested metrics for a single
                patient.

        Returns:
            If both A2C and A4C views of the patient are available, return both of them, in that order. If one is
            missing from the result, return the available view. If neither views are available, return `None`.
        """
        data = None
        avail_views = [view for view in self._required_views if view in result.views]
        for view in avail_views:
            view_data = result.views[view]
            if view_data.num_frames != 2:
                logger.warning(
                    f"Impossible to compute clinical metrics for '{result.id}' because of mismatched frames in the "
                    f"{view} view. Computing clinical metrics requires {len(self._required_instants)} key frames, "
                    f"{self._required_instants}, but the view counts {view_data.num_frames} frames."
                )
                break
        else:
            if set(avail_views) == set(self._required_views):
                data = result.views[View.A2C], result.views[View.A4C]
            else:
                assert len(avail_views) == 1
                avail_view = avail_views[0]
                logger.warning(
                    f"Cannot compute clinical metrics for '{result.id}' using Simpson's biplane method because of "
                    f"missing views. Falling back to estimating them from the {avail_view} 2D plane."
                )
                data = result.views[avail_view]
        return data

    def process_result(self, result: PatientResult) -> Tuple[str, "Metrics.ProcessingOutput"]:
        """Computes clinical metrics on data from a patient.

        Args:
            result: Data structure holding all the relevant information to compute the requested metrics for a single
                patient.

        Returns:
            - Identifier of the result for which the metrics where computed.
            - Mapping between the metrics and their value for the patient.
        """
        data = self._check_and_extract_data(result)
        if data is None:
            metrics = {"lv_edv_error": np.nan, "lv_esv_error": np.nan, "lv_ef_error": np.nan}
        elif isinstance(data, ViewResult):

            def _compute_lv_area(data_tag: str) -> Tuple[int, int]:
                ed, es = data[data_tag].data
                return np.isin(ed, Label.LV).sum(), np.isin(es, Label.LV).sum()

            gt_lv_ed_area, gt_lv_es_area = _compute_lv_area(self.target_tag)
            lv_ed_area, lv_es_area = _compute_lv_area(self.input_tag)
            gt_lv_ef = (gt_lv_ed_area - gt_lv_es_area) / gt_lv_ed_area
            lv_ef = (lv_ed_area - lv_es_area) / lv_ed_area

            metrics = {
                "lv_ed_area_error": 100 * abs(lv_ed_area - gt_lv_ed_area) / gt_lv_ed_area,
                "lv_es_area_error": 100 * abs(lv_es_area - gt_lv_es_area) / gt_lv_es_area,
                "lv_ef_error": 100 * abs(lv_ef - gt_lv_ef),
            }
        else:  # `view_data` is a tuple with the data from both views
            # Extract intermediate data structures from `result` object
            a2c_view, a4c_view = data

            # Extract ED and ES from predicted data for both views
            preds = a2c_view[self.input_tag].data, a4c_view[self.input_tag].data

            # Extract ED and ES from reference data for both views
            gts = a2c_view[self.target_tag].data, a4c_view[self.target_tag].data

            metrics = compute_clinical_metrics_by_patient(
                preds, gts, a2c_view.voxelspacing[1:], a4c_view.voxelspacing[1:]
            )
        return result.id, metrics


def main():
    """Run the script."""
    ClinicalMetrics.main()


if __name__ == "__main__":
    main()
