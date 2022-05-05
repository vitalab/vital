import shutil
from typing import Any, Dict, Optional, Sequence, Union

import h5py
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor

from vital.data.camus.config import CamusTags, seg_save_options
from vital.data.camus.dataset import get_segmentation_attributes
from vital.data.camus.utils.register import CamusRegisteringTransformer
from vital.data.config import Subset
from vital.utils.format.numpy import to_categorical
from vital.utils.image.transform import resize_image


class CamusPredictionWriter(BasePredictionWriter):
    """Implementation of the prediction writer that saves predictions for the CAMUS dataset in a HDF5 file."""

    def __init__(self):
        super().__init__(write_interval="batch")

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        """Removes results potentially left behind by previous runs of the callback in the same directory."""
        shutil.rmtree(pl_module.log_dir / "results", ignore_errors=True)  # Delete leftover results processors' outputs
        (pl_module.log_dir / "test.h5").unlink(missing_ok=True)  # Delete leftover predictions dataset

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Union[Tensor, Dict[str, Tensor]],
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Saves ground truths and predictions for a patient inside a HDF5 file.

        Args:
            trainer: `Trainer` used in the experiment.
            pl_module: `LightningModule` used in the experiment.
            prediction: Either only the predicted segmentation, or a dictionary of predictions identified by tags, with
                the predicted segmentation accessible under the `CamusTags.pred` key.
                The predicted segmentations is in format (N, ``out_channels``, H, W), and its values can either be the
                raw, unnormalized scores predicted by the model or the segmentation in one-hot format.
                The other predications must be tensors, and their dataset will be named after their key in the
                prediction dictionary.
            batch_indices: Indices of all the batches whose outputs are provided.
            batch: The current batch used by the model to give its prediction.
            batch_idx: Index of the current batch.
            dataloader_idx: Index of the current dataloader.
        """
        # Extract the main prediction, i.e. the predicted segmentation, and the auxiliary predictions
        aux_predictions = {}
        if isinstance(prediction, Tensor):
            pred_view_seg = prediction
        else:
            pred_view_seg = prediction.pop(CamusTags.pred)
            aux_predictions = prediction

        # Collect the metadata related to the batch's data
        batch_metadata = trainer.datamodule.dataset(subset=Subset.PREDICT).get_view_metadata(batch_idx)
        patient_id, view = batch_metadata.id.split("/")

        # Init objects to process the predictions
        registering_transformer = CamusRegisteringTransformer(
            num_classes=pl_module.hparams.data_params.out_shape[0],
            crop_shape=pl_module.hparams.data_params.in_shape[1:],
        )

        with h5py.File(pl_module.log_dir / "test.h5", "a") as dataset:
            # Create the file hierarchy where to save the batch's predictions
            patient_group = dataset.require_group(patient_id)
            view_group = patient_group.create_group(view)

            # Save the relevant data for a posteriori result analysis
            full_resolution_prediction = np.empty_like(batch_metadata.gt)
            pred_view_seg = to_categorical(pred_view_seg.detach().cpu().numpy(), channel_axis=1)
            for instant, pred_instant_seg in enumerate(pred_view_seg):

                # Format the predictions to fit with the groundtruth
                if batch_metadata.registering:  # Undo registering on predictions
                    registering_parameters = {
                        reg_step: batch_metadata.registering[reg_step][instant]
                        for reg_step in CamusRegisteringTransformer.registering_steps
                    }
                    pred_instant_seg = registering_transformer.undo_registering(
                        pred_instant_seg, registering_parameters
                    )
                else:
                    height, width = batch_metadata.gt.shape[1:]  # Extract images' original dimensions
                    pred_instant_seg = resize_image(pred_instant_seg, (width, height))

                full_resolution_prediction[instant] = pred_instant_seg

            for tag, data in [(CamusTags.gt, batch_metadata.gt), (CamusTags.pred, full_resolution_prediction)]:
                data_group = view_group.create_group(tag)
                ds = data_group.create_dataset(CamusTags.raw, data=data, **seg_save_options)
                # Save shape attributes of the segmentation as dataset attributes
                for attr, attr_val in get_segmentation_attributes(data, pl_module.hparams.data_params.labels).items():
                    ds.attrs[attr] = attr_val.squeeze()

            # Save auxiliary predictions
            pred_group = view_group[CamusTags.pred]
            for tag, aux_prediction in aux_predictions.items():
                pred_group.create_dataset(tag, data=aux_prediction.detach().cpu().numpy())

            # Save metadata
            view_group.attrs[CamusTags.voxelspacing] = batch_metadata.voxelspacing
            view_group.attrs[CamusTags.instants] = list(batch_metadata.instants)  # Indicate available instants
            view_group.attrs.update(batch_metadata.instants)  # Indicate clinically important instants' frames
            if batch_metadata.registering:  # Save registering parameters if the VAE was trained on registered data
                view_group.attrs.update(batch_metadata.registering)
