import logging
from pathlib import Path
from typing import List, Sequence

from vital.data.cardinal.config import IMG_FILENAME_PATTERN, IMG_FORMAT
from vital.data.cardinal.config import View as ViewEnum

logger = logging.getLogger(__name__)


def views_avail_by_patient(data_roots: Sequence[Path], patient_id: str) -> List[ViewEnum]:
    """Searches for files related to a patient, to list all the views for which data is available for the patient.

    Args:
        data_roots: Root directories inside which to search recursively for files related to the patient.
        patient_id: ID of the patient for whom to search.

    Returns:
        Views for which data exists for the patient.
    """

    def extract_view_from_filename(filename: Path) -> ViewEnum:
        return ViewEnum(filename.name.split("_")[1])

    # Collect all files related to the patient from the multiple root directories
    patient_files_pattern = IMG_FILENAME_PATTERN.format(patient_id=patient_id, view="*", tag="*", ext=IMG_FORMAT)
    patient_files = []
    for data_root in data_roots:
        # Search recursively inside the provided directory
        patient_files.extend(data_root.rglob(patient_files_pattern))

    # Identify all the unique views across all the files related to the patient
    view_by_files = [extract_view_from_filename(patient_file) for patient_file in patient_files]
    views = sorted(set(view_by_files), key=lambda view: list(ViewEnum).index(view))
    return views
