from dataclasses import dataclass
from typing import Dict, ItemsView, List, Mapping, Optional, TypeVar, Union, ValuesView

import h5py
import numpy as np

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class LazyDict(Dict[_KT, _VT]):
    """Dictionary allowing callable values that are evaluated lazily upon first access."""

    def __getitem__(self, k: _KT) -> _VT:
        """Accessor that tests if the requested value is callable, evaluating and saving the result if it is."""
        v = super().__getitem__(k)
        if callable(v):
            v = v()
        self[k] = v
        return v

    def get(self, k: _KT, default: Optional[_VT] = None) -> Optional[_VT]:
        """Necessary override of `get` to call `__getitem__` to trigger the evaluation of the value."""
        if k in self:
            return self.__getitem__(k)
        return default

    def items(self) -> ItemsView[_KT, _VT]:
        """Necessary override of default `dict_items` with a structure that triggers the evaluation of the values."""
        return ItemsView(self)

    def values(self) -> ValuesView[_VT]:
        """Necessary override of default `dict_values` with a structure that triggers the evaluation of the values."""
        return ValuesView(self)


Attributes = Mapping[str, np.ndarray]  # Data structure that mimics an h5py.AttributeManager


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

    def __getitem__(self, key: str) -> Union["Group", Dataset]:
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


def load_leaf_group_from_hdf5(group: h5py.Group, keep_indices: List[int] = None) -> Group:
    """Creates a `Group` object from a leaf `h5py.Group`, i.e. a group containing no nested groups, only datasets.

    Args:
        group: HDF5 group to load as a `Group` object.
        keep_indices: Indices by which to index datasets/attributes along their 1st dimension.

    Returns:
        `Group` object created from the content of the `h5py.Group`.
    """

    def _load_attributes_from_hdf5(attrs: h5py.AttributeManager, keep_indices: List[int] = None) -> Attributes:
        return {name: val if not keep_indices else val[keep_indices] for name, val in attrs.items()}

    def _load_dataset_from_hdf5(ds: h5py.Dataset, keep_indices: List[int] = None) -> Dataset:
        return Dataset(
            ds if not keep_indices else ds[keep_indices],
            _load_attributes_from_hdf5(ds.attrs, keep_indices=keep_indices),
        )

    return Group({ds: _load_dataset_from_hdf5(group[ds], keep_indices=keep_indices) for ds in group}, dict(group.attrs))


def filter_leaf_group(group: Group, keep_indices: int) -> Group:
    """Filters an existing `Group` object to only keep som indices along the 1st dimension of the datasets/attributes.

    Args:
        group: Group whose datasets and attributes to filter.
        keep_indices: Indices by which to index datasets/attributes along their 1st dimension.

    Returns:
        Group with datasets/attributes filtered according to the indices specified in `keep_indices`.
    """
    return Group(
        {
            ds_name: Dataset(ds.data[keep_indices], {name: val[keep_indices] for name, val in ds.attrs.items()})
            for ds_name, ds in group.data.items()
        },
        group.attrs,
    )
