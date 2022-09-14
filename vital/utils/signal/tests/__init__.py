import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Extends default JSONEncoder to serialize numpy arrays as JSON arrays."""

    def default(self, obj):  # noqa: D102
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def serialize_signals_to_json(signals: Dict[str, Tuple[np.ndarray, np.ndarray]], output_file: Path) -> None:
    """Writes signal data in a human-interpretable serialized JSON format.

    Args:
        signals: Signal data to serialize.
        output_file: Path where to write the JSON serialized data.
    """
    json_str = json.dumps(signals, cls=NumpyEncoder, indent=2)
    output_file.write_text(json_str)


def load_signals_from_json(json_file: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Loads signal data from its serialized JSON format, casting arrays to numpy arrays in the process.

    Args:
        json_file: Path of the serialized JSON data.

    Returns:
        Signal data, with signals properly cast to numpy arrays.
    """
    with open(json_file) as f:
        signals = json.load(f)
    return {signal_tag: (np.array(signal), np.array(target)) for signal_tag, (signal, target) in signals.items()}
