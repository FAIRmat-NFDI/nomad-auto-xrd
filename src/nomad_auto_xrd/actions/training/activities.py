import json
import os

import tensorflow as tf
from nomad.app.v1.routers.uploads import get_upload_with_read_access
from nomad.datamodel import User
from nomad.orchestrator.utils import get_upload_files
from temporalio import activity

from nomad_auto_xrd.actions.training.models import (
    CreateTrainedModelEntryInput,
    TrainModelInput,
)
from nomad_auto_xrd.models import TrainModelOutput
from nomad_auto_xrd.schema import AutoXRDModel


class TemporalHeartbeatCallback(tf.keras.callbacks.Callback):
    """Callback to send Temporal heartbeats during training."""

    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs

    def on_train_begin(self, logs=None):
        activity.heartbeat('Training started...')

    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1
        progress_msg = f'Epoch {epoch_num}/{self.total_epochs} - '

        if logs:
            progress_msg += (
                f'Loss: {logs.get("loss", 0):.4f}, '
                f'Acc: {logs.get("categorical_accuracy", 0):.4f}, '
                f'Val Loss: {logs.get("val_loss", 0):.4f}, '
                f'Val Acc: {logs.get("val_categorical_accuracy", 0):.4f}'
            )

        activity.heartbeat(progress_msg)

    def on_train_end(self, logs=None):
        activity.heartbeat('Training completed!')


@activity.defn
async def train_model(data: TrainModelInput) -> TrainModelOutput:
    """
    Activity to train a machine learning model.
    """
    from nomad_auto_xrd.training import train

    # Run training within the upload folder
    original_path = os.path.abspath(os.curdir)
    upload_files = get_upload_files(data.upload_id, data.user_id)
    upload_raw_path = os.path.join(upload_files.os_path, 'raw')
    try:
        os.chdir(upload_raw_path)
        output = train(
            working_directory=data.working_directory,
            simulation_settings=data.simulation_settings,
            training_settings=data.training_settings,
            includes_pdf=data.includes_pdf,
            callbacks=[TemporalHeartbeatCallback(data.training_settings.num_epochs)],
        )
    finally:
        os.chdir(original_path)

    if output.xrd_model_path is None:
        raise ValueError('XRD model path is None, training may have failed.')

    return output


@activity.defn
async def create_trained_model_entry(data: CreateTrainedModelEntryInput) -> None:
    """
    Activity to create a trained model entry in the same upload.
    """

    upload_files = get_upload_files(data.upload_id, data.user_id)
    # upload = get_upload_with_read_access(
    #     data.upload_id,
    #     User(user_id=data.user_id),
    #     include_others=True,
    # )

    # fname = os.path.join('inference_result.archive.json')
    # with open(fname, 'w', encoding='utf-8') as f:
    #     json.dump({'data': inference_result.m_to_dict(with_root_def=True)}, f, indent=4)
    # upload.process_upload(
    #     file_operations=[
    #         dict(op='ADD', path=fname, target_dir=result.cif_dir, temporary=True)
    #     ],
    #     only_updated_files=True,
    # )
