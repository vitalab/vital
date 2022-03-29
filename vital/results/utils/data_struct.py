from dataclasses import dataclass
from typing import Dict, Mapping, Union

import numpy as np

Attributes = Mapping[str, np.ndarray]  #: Data structure that mimics an h5py.AttributeManager


@dataclass
class Dataset:
    """Data structure that mimics an h5py.Dataset."""

    data: np.ndarray
    attrs: Attributes


@dataclass
class Group:
    """Data structure that mimics an h5py.Group."""

    data: Dict[str, Union["Group", Dataset]]
    attrs: Attributes


class GroupItemMixin(Group):
    """Utility class to provide easy syntax to access substructures of HDF5-like hierarchical data structures."""

    def __getitem__(self, key: str) -> Union[Group, Dataset]:
        """Allows h5py-like syntax, with slash-separated keys, to access multiple levels deep in nested groups/datasets.

        Args:
            key: Key, or nested keys, for accessing data inside the ``data`` group. Slashes ("/") delimit the nesting of
                the keys.

        Returns:
            Data pointed to by the nested keys, starting from the ``data`` root.
        """
        keys_by_level = key.split("/")
        nested_item = self
        for key in keys_by_level:
            nested_item = nested_item.data[key]
        return nested_item
