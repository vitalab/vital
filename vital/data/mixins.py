from typing import List

from vital.systems.system import SystemDataManagerMixin


class StructuredDataMixin(SystemDataManagerMixin):
    """``SystemDataManagerMixin`` mixin for datasets where data has more structure than only the target labels."""

    def train_group_ids(self, *args, **kwargs) -> List[str]:
        """Lists the IDs of the different groups/clusters samples in the training data can belong to.

        Args:
            *args: Positional arguments that parameterize the requested data structure.
            **kwargs: Keyword arguments that parameterize the requested data structure.

        Returns:
            IDs of the different groups/clusters samples in the training data can belong to.
        """
        raise NotImplementedError

    def val_group_ids(self, *args, **kwargs) -> List[str]:
        """Lists the IDs of the different groups/clusters samples in the validation data can belong to.

        Args:
            *args: Positional arguments that parameterize the requested data structure.
            **kwargs: Keyword arguments that parameterize the requested data structure.

        Returns:
            IDs of the different groups/clusters samples in the validation data can belong to.
        """
        raise NotImplementedError
