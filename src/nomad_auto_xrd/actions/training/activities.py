import os
from dataclasses import asdict

import tensorflow as tf
from nomad.datamodel.context import ServerContext
from nomad.orchestrator.utils import get_upload, get_upload_files
from temporalio import activity

from nomad_auto_xrd.actions.training.models import (
    CreateTrainedModelEntryInput,
    TrainModelInput,
)
from nomad_auto_xrd.models import TrainModelOutput
from nomad_auto_xrd.schema import AutoXRDModel, ReferenceStructure


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

    model = AutoXRDModel(
        working_directory=data.working_directory,
        includes_pdf=data.includes_pdf,
    )
    model.m_setdefault('simulation_settings')
    model.m_setdefault('training_settings')
    model.training_settings = model.training_settings.m_from_dict(
        asdict(data.training_settings)
    )

    model.simulation_settings = model.simulation_settings.m_from_dict(
        asdict(data.simulation_settings)
    )
    model.simulation_settings.structure_files = [
        os.path.join(data.working_directory, path)
        for path in data.simulation_settings.structure_files
    ]

    model.xrd_model = data.xrd_model_path.split('/raw/', 1)[-1]
    model.pdf_model = (
        data.pdf_model_path.split('/raw/', 1)[-1] if data.pdf_model_path else None
    )
    model.wandb_run_url_xrd = data.wandb_run_url_xrd
    model.wandb_run_url_pdf = data.wandb_run_url_pdf
    model.reference_structures = [
        ReferenceStructure(
            name=os.path.basename(cif_path).split('.cif')[0],
            cif_file=cif_path.split('/raw/', 1)[-1],
        )
        for cif_path in data.reference_structure_paths
    ]

    context = ServerContext(get_upload(data.upload_id, data.user_id))
    archive_name = os.path.join(data.working_directory, 'auto_xrd_model.archive.json')
    with context.update_entry(archive_name, process=True, write=True) as archive:
        archive['data'] = model.m_to_dict(with_root_def=True)
