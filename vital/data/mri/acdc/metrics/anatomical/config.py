import numpy as np

thresholds = {
    "lv_concavity": {"max": 4},
    "rv_concavity": {"max": 30},
    "myo_concavity": {"max": 0},
    "lv_circularity": {"min": 0.2},
    "myo_circularity": {"min": 0.2},
}
domains = {
    "lv_concavity": {"min": 0, "max": np.inf},
    "rv_concavity": {"min": 0, "max": np.inf},
    "myo_concavity": {"min": 0, "max": np.inf},
    "lv_circularity": {"min": 0, "max": np.inf},
    "myo_circularity": {"min": 0, "max": np.inf},
    "holes_in_lv": {"min": 0, "max": 65536},
    "holes_in_rv": {"min": 0, "max": 65536},
    "holes_in_myo": {"min": 0, "max": 65536},
    "disconnectivity_in_lv": {"min": 0, "max": 65536},
    "disconnectivity_in_rv": {"min": 0, "max": 65536},
    "disconnectivity_in_myo": {"min": 0, "max": 65536},
    "holes_between_lv_and_myo": {"min": 0, "max": 65536},
    "holes_between_rv_and_myo": {"min": 0, "max": 65536},
    "rv_disconnected_from_myo": {"min": 0, "max": 65536},
    "frontier_between_lv_and_rv": {"min": 0, "max": 65536},
    "frontier_between_lv_and_background": {"min": 0, "max": 65536},
}

ideal_value = {
    "lv_concavity": 0,
    "rv_concavity": 0,
    "myo_concavity": 0,
    "lv_circularity": 1,
    "myo_circularity": 1,
    "holes_in_lv": 0,
    "holes_in_rv": 0,
    "holes_in_myo": 0,
    "disconnectivity_in_lv": 0,
    "disconnectivity_in_rv": 0,
    "disconnectivity_in_myo": 0,
    "holes_between_lv_and_myo": 0,
    "holes_between_rv_and_myo": 0,
    "rv_disconnected_from_myo": 0,
    "frontier_between_lv_and_rv": 0,
    "frontier_between_lv_and_background": 0,
}
