import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import hydra.utils
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.trainer.states import TrainerFn
from torch.utils.data import DataLoader, MapDataPipe
from torchvision.transforms import transforms

from vital.data.cardinal.config import CardinalTag, TabularAttribute, TimeSeriesAttribute
from vital.data.cardinal.config import View as ViewEnum
from vital.data.cardinal.datapipes import PatientData, build_datapipes
from vital.data.cardinal.utils.itertools import Patients
from vital.data.config import DataParameters, Subset
from vital.data.data_module import VitalDataModule
from vital.utils.config import instantiate_config_node_leaves

logger = logging.getLogger(__name__)

PREDICT_DATALOADERS_SUBSETS = [Subset.TRAIN, Subset.VAL, Subset.TEST]


class CardinalDataModule(VitalDataModule):
    """Implementation of the ``VitalDataModule`` for the Cardinal dataset."""

    def __init__(
        self,
        patients_kwargs: Dict[str, Any],
        process_patient_kwargs: Dict[str, Any],
        transform_patient_kwargs: Dict[Subset, Dict[str, Any]] = None,
        datapipes_kwargs: Dict[str, Any] = None,
        subsets: Dict[Union[str, Subset], Union[str, Path]] = None,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            patients_kwargs: Parameters to forward to the `Patients` sequence.
            process_patient_kwargs: Parameters to forward to the `PatientProcessor` datapipe.
            transform_patient_kwargs: Mapping between subsets and sequences of transforms to apply to the attributes.
                In the cases of the `tabular_attrs_transforms` and `time_series_attrs_transforms` keys, the value
                can be an OmegaConf DictConfig that needs to be instantiated recursively to give a callable transform.
            datapipes_kwargs: Parameters to forward to the datapipes factory.
            subsets: Paths to files listing the patients to include in each subset of the data. Patients whose data is
                available in the `data_roots` but who are not listed in the `subsets` files will be ignored. If no such
                file is specified for a subset, then all patients from the `Patients` sequence will be included.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        # If instantiating the datamodule from a Hydra config, make sure to cast nested configs to primitive python
        # containers + instantiate any objects where it is needed, to avoid unexpected side effects later on
        if isinstance(patients_kwargs, DictConfig):
            patients_kwargs = OmegaConf.to_container(patients_kwargs, resolve=True)
        else:
            # If the config does not come from Hydra, manually add the '_target_' field
            patients_kwargs["_target_"] = ".".join([Patients.__module__, Patients.__qualname__])
        self._partial_patients = hydra.utils.instantiate(patients_kwargs, _partial_=True)

        if isinstance(process_patient_kwargs, DictConfig):
            process_patient_kwargs = OmegaConf.to_container(process_patient_kwargs, resolve=True)
        self._process_patient_kwargs = process_patient_kwargs

        if transform_patient_kwargs is None:
            transform_patient_kwargs = {}
        elif isinstance(transform_patient_kwargs, DictConfig):
            transform_patient_kwargs = OmegaConf.to_container(transform_patient_kwargs, resolve=True)

            # Hard-coded keys of groups that need custom instantiation logic
            transform_groups = ["tabular_attrs_transforms", "time_series_attrs_transforms"]

            for subset, subset_xform_kwargs in transform_patient_kwargs.items():
                for kwarg_name, kwarg_val in subset_xform_kwargs.items():
                    # Instantiate lists of callable transforms inside transforms.Compose to automatically chain them
                    if kwarg_name in transform_groups:
                        subset_xform_kwargs[kwarg_name] = {
                            attr_or_any: transforms.Compose(
                                instantiate_config_node_leaves(
                                    transforms_listconfig, f"{subset} {kwarg_name.removesuffix('s')}"
                                )
                            )
                            for attr_or_any, transforms_listconfig in kwarg_val.items()
                        }

        self._transform_patient_kwargs = transform_patient_kwargs

        self._subsets_lists = {}
        if subsets:
            # Make sure the key for each subset is a `Subset` enum instance + read the lists of patients
            self._subsets_lists = {
                Subset[subset.upper()]
                if isinstance(subset, Subset)
                else subset: Path(subset_file).read_text().splitlines()
                for subset, subset_file in subsets.items()
            }
        self.subsets_patients = {}  # To populate in `setup`

        if datapipes_kwargs is None:
            datapipes_kwargs = {}
        elif isinstance(datapipes_kwargs, DictConfig):
            datapipes_kwargs = OmegaConf.to_container(datapipes_kwargs, resolve=True)
        self._datapipes_kwargs = datapipes_kwargs

        # Load an example item to dynamically detect the shape(s) of the different data modalities
        train_dp = self._build_subset_datapipes(Subset.TRAIN)
        first_item = train_dp[0]
        modalities_shapes = {}
        if ViewEnum.A4C in first_item:
            # If image data is available

            # Add the shapes of each sequence, by process of elimination. Since a views' data contains only sequences
            # and the masks' attributes, any entry that is not an attribute is an image
            modalities_shapes.update(
                {
                    item_data_tag: item_data.shape
                    for item_data_tag, item_data in first_item[ViewEnum.A4C].items()
                    if item_data_tag not in TimeSeriesAttribute
                }
            )
            # Add the normalized size of the time-series attributes: (num_attrs, attr_len)
            first_time_series_attr = list(first_item[ViewEnum.A4C])[0]
            modalities_shapes[CardinalTag.time_series_attrs] = (
                sum(
                    1
                    for view in ViewEnum
                    for view_entry in first_item.get(view, [])
                    if view_entry in TimeSeriesAttribute
                ),  # Number of attributes across all views
                len(first_item[ViewEnum.A4C][first_time_series_attr]),  # Attribute shape
            )
        # Add the number of tabular attributes
        num_tab_attrs = sum(1 for tab_attrs in TabularAttribute if tab_attrs in first_item)
        if num_tab_attrs:
            modalities_shapes[CardinalTag.tabular_attrs] = (num_tab_attrs, 1)

        super().__init__(data_params=DataParameters(in_shape=modalities_shapes), **kwargs)

    def _build_subset_datapipes(self, subset: Subset) -> MapDataPipe[PatientData]:
        """Builds a series of datapipes for processing the data of the requested subset.

        Args:
            subset: Dataset subset to be processed by the datapipes.

        Returns:
            Datapipe for processing the data of the requested subset.
        """
        patients = self._partial_patients(include_patients=self._subsets_lists.get(subset))
        self.subsets_patients[subset] = patients
        return build_datapipes(
            patients,
            process_patient_kwargs=self._process_patient_kwargs,
            transform_patient_kwargs=self._transform_patient_kwargs[subset],
            **self._datapipes_kwargs,
        )

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D102
        # Determine subset to setup given the stage of training
        subsets_to_setup = []
        match stage:
            case TrainerFn.FITTING:
                subsets_to_setup.extend([Subset.TRAIN, Subset.VAL])
            case TrainerFn.VALIDATING:
                subsets_to_setup.append(Subset.VAL)
            case TrainerFn.TESTING:
                subsets_to_setup.append(Subset.TEST)
            case TrainerFn.PREDICTING:
                subsets_to_setup.extend(PREDICT_DATALOADERS_SUBSETS)

        # Update the available collections of patients and datasets
        self.datasets.update({subset: self._build_subset_datapipes(subset) for subset in subsets_to_setup})

    def predict_dataloader(self) -> Sequence[DataLoader]:  # noqa: D102
        return [
            DataLoader(self.datasets[subset], batch_size=None, num_workers=self.num_workers, pin_memory=True)
            for subset in PREDICT_DATALOADERS_SUBSETS
        ]


@hydra.main(version_base=None, config_path="../../config/data", config_name="cardinal")
def main(cfg: DictConfig):
    """Test loading a batch from the datamodule's train dataloader."""
    import hydra

    logger.info(f"Instantiating Cardinal datamodule based on the following Hydra config: \n{OmegaConf.to_yaml(cfg)}")
    datamodule = hydra.utils.instantiate(cfg, _recursive_=False)

    datamodule.setup(stage=TrainerFn.FITTING)
    val_dl = datamodule.val_dataloader()

    def print_items_shape(mapping: Dict[str, Any], level: int = 0) -> None:
        """Logs the shapes of values in an arbitrarily nested dict, indenting at each level of nesting."""
        indent = "  " * level
        for item_k, item_v in mapping.items():
            if isinstance(item_v, dict):
                logger.info(indent + f"Nested key: {item_k}")
                print_items_shape(item_v, level=level + 1)
            else:
                logger.info(indent + f"Key: {item_k}, shape: {item_v.shape}")

    logger.info(
        f"Validation batches dataloader split in '{len(val_dl)}' batches. The content of the batches are detailed "
        f"below."
    )
    for batch_idx, batch in enumerate(val_dl):
        logger.info(f"Content of batch {batch_idx}: ")
        print_items_shape(batch)


if __name__ == "__main__":
    from dotenv import load_dotenv

    from vital.utils.config import register_omegaconf_resolvers

    # Load environment variables from `.env` file if it exists
    # Load before hydra main to allow for setting environment variables with ${oc.env:ENV_NAME}
    load_dotenv()
    register_omegaconf_resolvers()

    main()
