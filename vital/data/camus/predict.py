import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import h5py
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.callbacks.prediction_writer import WriteInterval
from torch import Tensor
from tqdm.auto import tqdm

from vital.data.camus.config import CamusTags, img_save_options, seg_save_options
from vital.data.camus.dataset import get_segmentation_attributes
from vital.data.camus.utils.register import CamusRegisteringTransformer
from vital.data.config import Tags
from vital.results.camus.utils.itertools import PatientViews
from vital.utils.format.numpy import to_categorical
from vital.utils.image.process import PostProcessor
from vital.utils.image.transform import resize_image

logger = logging.getLogger(__name__)


class CamusPredictionWriter(BasePredictionWriter):
    """Implementation of the prediction writer that saves predictions for the CAMUS dataset in a HDF5 file."""

    def __init__(
        self,
        write_path: Union[str, Path] = None,
        postprocessors: Sequence[PostProcessor] = None,
        write_post_predictions_only: bool = False,
        epoch_end_progress_bar: bool = True,
    ):
        """Initializes class instance.

        Args:
            write_path: Path of the output HDF5 dataset where to save the predictions.
            postprocessors: Callable post-processor objects to use to process the predictions.
            write_post_predictions_only: Whether to skip saving the predictions for each batch (assuming they've already
                been saved) and go straight to post-processing the predictions at the end of the prediction epoch.
            epoch_end_progress_bar: Whether to display on progress bar in the epoch end hook when post-processing the
                predictions.
        """
        write_interval = WriteInterval.BATCH
        if postprocessors:
            write_interval = WriteInterval.BATCH_AND_EPOCH
        if write_post_predictions_only:
            if not postprocessors:
                raise ValueError(
                    "`CamusPredictionWriter` called to only post-process predictions, but no functions to post-process "
                    "the results were provided."
                )
            write_interval = WriteInterval.EPOCH

        super().__init__(write_interval=write_interval)
        self._write_path = Path(write_path) if write_path else None
        self._epoch_end_progress_bar = epoch_end_progress_bar
        self._postprocessors = postprocessors
        if self._postprocessors is None:
            self._postprocessors = []

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        """Removes results potentially left behind by previous runs of the callback in the same directory."""
        # Write to the same directory as the experiment logger if no custom path is provided
        if self._write_path is None:
            self._write_path = pl_module.log_dir / "test.h5"

        # Delete leftover predictions dataset
        self._write_path.unlink(missing_ok=True)

    def postprocess_prediction(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> Dict[str, Tensor]:
        """Post-processes the prediction made on a batch of data.

        Args:
            trainer: `Trainer` used in the experiment.
            pl_module: `LightningModule` used in the experiment.
            prediction:  Batch of prediction, or a dictionary containing the batch of prediction under the 'pred' key.

        Returns:
            Dictionary containing the original prediction (under the 'pred' key), the post-processed prediction
            (under the 'post_pred' key), and any original content of `prediction` along with additional outputs of the
            post-processors.
        """
        if isinstance(prediction, np.ndarray):
            output = {Tags.pred: prediction}
        elif isinstance(prediction, dict) and pl_module.hparams.mask_tag in prediction:
            prediction[Tags.pred] = prediction.pop(pl_module.hparams.mask_tag)
            output = prediction
        else:
            raise RuntimeError(
                f"Could not post-process the predictions in '{type(self)}', because we could not infer how to unpack "
                f"the predictions from '{type(pl_module)}'s `predict_step`. Please only return the predictions to"
                f"post-process, or include them in a dictionary under the '{pl_module.hparams.mask_tag}' key."
            )

        output[Tags.post_pred] = output[Tags.pred]
        for postprocessor in self._postprocessors:
            post_pred = postprocessor(output[Tags.post_pred])
            if isinstance(post_pred, dict):
                post_pred[Tags.post_pred] = post_pred.pop(f"post_{pl_module.hparams.mask_tag}")
                output.update(post_pred)
            elif isinstance(post_pred, np.ndarray):
                output[Tags.post_pred] = post_pred
            else:
                raise RuntimeError(
                    f"Unsupported output type from postprocessor '{type(postprocessor)}' in the postprocessing loop. "
                    f"Modify the postprocessor to return either a '{np.ndarray}' or a '{dict}', or remove the "
                    f"postprocessor altogether."
                )

        return output

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
                The other predictions must be tensors, and their dataset will be named after their key in the prediction
                dictionary.
            batch_indices: Indices of all the batches whose outputs are provided.
            batch: The current batch used by the model to give its prediction.
            batch_idx: Index of the current batch.
            dataloader_idx: Index of the current dataloader.
        """
        # Extract the main prediction, i.e. the predicted segmentation, and the auxiliary predictions
        aux_predictions = {}
        if isinstance(prediction, Tensor):  # Segmentation model
            pred_view_seg = prediction
        else:  # Autoencoder model
            pred_view_seg = prediction.pop(pl_module.hparams.mask_tag)
            aux_predictions = prediction

        # Collect the metadata related to the batch's data
        view_metadata = batch[CamusTags.metadata]
        patient_id, view = batch[CamusTags.id].split("/")

        # Init objects to process the predictions
        registering_transformer = CamusRegisteringTransformer(
            num_classes=pl_module.hparams.data_params.out_shape[0],
            crop_shape=pl_module.hparams.data_params.in_shape[1:],
        )

        with h5py.File(self._write_path, "a") as dataset:
            # Create the file hierarchy where to save the batch's predictions
            patient_group = dataset.require_group(patient_id)
            view_group = patient_group.create_group(view)

            # Save the relevant data for a posteriori result analysis
            full_resolution_prediction = np.empty_like(view_metadata.gt)
            pred_view_seg = to_categorical(pred_view_seg.detach().cpu().numpy(), channel_axis=1)
            for instant, pred_instant_seg in enumerate(pred_view_seg):
                # Format the predictions to fit with the groundtruth
                if view_metadata.registering:  # Undo registering on predictions
                    registering_parameters = {
                        reg_step: view_metadata.registering[reg_step][instant]
                        for reg_step in CamusRegisteringTransformer.registering_steps
                    }
                    pred_instant_seg = registering_transformer.undo_registering(
                        pred_instant_seg, registering_parameters
                    )
                else:
                    height, width = view_metadata.gt.shape[1:]  # Extract images' original dimensions
                    pred_instant_seg = resize_image(pred_instant_seg, (width, height))

                full_resolution_prediction[instant] = pred_instant_seg

            for tag, data in [(CamusTags.gt, view_metadata.gt), (CamusTags.pred, full_resolution_prediction)]:
                data_group = view_group.create_group(tag)
                ds = data_group.create_dataset(CamusTags.raw, data=data, **seg_save_options)
                # Save shape attributes of the segmentation as dataset attributes
                for attr, attr_val in get_segmentation_attributes(data, pl_module.hparams.data_params.labels).items():
                    ds.attrs[attr] = attr_val

            # Save auxiliary predictions
            pred_group = view_group[CamusTags.pred]
            for tag, aux_prediction in aux_predictions.items():
                pred_group.create_dataset(tag, data=aux_prediction.detach().cpu().numpy())

            # Save metadata
            view_group.attrs[CamusTags.voxelspacing] = view_metadata.voxelspacing
            view_group.attrs[CamusTags.instants] = list(view_metadata.instants)  # Indicate available instants
            view_group.attrs.update(view_metadata.instants)  # Indicate clinically important instants' frames
            if view_metadata.registering:  # Save registering parameters if the VAE was trained on registered data
                view_group.attrs.update(view_metadata.registering)

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ) -> None:
        """Post-processes the predictions saved for each batch and saves the post-processed results.

        Args:
            trainer: `Trainer` used in the experiment.
            pl_module: `LightningModule` used in the experiment.
            predictions: Collection of predictions accumulated over the batches. The parameter is there to respect the
                signature of the parent function, but it is not used here. Rather, the predictions to post-process are
                read directly from the file they were saved to.
            batch_indices: Indices of all the batches whose outputs are provided.
        """
        # Create an iterator over the results written for each batch
        patient_views_iter = PatientViews(
            results_path=self._write_path, sequence="full_cycle", use_sequence=trainer.datamodule.hparams.use_sequence
        )

        # Set up the feedback to the user
        progress_msg = f"Post-processing '{self._write_path}' predictions"
        if self._epoch_end_progress_bar:
            patient_views_iter = tqdm(
                patient_views_iter, total=len(patient_views_iter), unit=patient_views_iter.desc, desc=progress_msg
            )
        else:
            logger.info(progress_msg)

        # Initialize the file to write to as a copy of the predictions already saved
        with h5py.File(self._write_path, "a") as dataset:
            # For each group of results
            for view_result in patient_views_iter:
                # Post-process the raw predictions
                post_dict = self.postprocess_prediction(
                    trainer, pl_module, view_result[f"{CamusTags.pred}/{CamusTags.raw}"].data
                )
                post, enc = post_dict[CamusTags.post_pred], post_dict.get(pl_module.hparams.get("encoding_tag"))

                # Write the post-processed predictions to the file
                view_pred_group = dataset.require_group(f"{view_result.id}/{CamusTags.pred}")
                post_ds = view_pred_group.require_dataset(CamusTags.post, shape=post.shape, **seg_save_options)
                post_ds[...] = post
                for attr, attr_val in get_segmentation_attributes(post, pl_module.hparams.data_params.labels).items():
                    post_ds.attrs[attr] = attr_val

                if enc is not None:
                    enc_ds = view_pred_group.require_dataset(CamusTags.encoding, shape=enc.shape, **img_save_options)
                    enc_ds[...] = enc
