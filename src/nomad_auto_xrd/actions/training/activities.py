import json
import os
from dataclasses import asdict

import tensorflow as tf
from temporalio import activity

from nomad_auto_xrd.actions.training.models import (
    CreateTrainedModelEntryInput,
    TrainModelInput,
)
from nomad_auto_xrd.common.models import TrainModelOutput


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
    from nomad.actions.utils import get_upload_files

    from nomad_auto_xrd.common.training import train

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

    from nomad.actions.utils import get_action_status, get_upload, get_upload_files
    from nomad.client import parse
    from nomad.datamodel.context import ServerContext
    from nomad.utils import hash as m_hash
    from nomad_measurements.utils import get_reference

    from nomad_auto_xrd.schema_packages.schema import (
        AutoXRDModel,
        AutoXRDModelReference,
        ReferenceStructure,
    )

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

    upload = get_upload(data.upload_id, data.user_id)
    archive_name = 'auto_xrd_model.archive.json'
    with open(archive_name, 'w', encoding='utf-8') as f:
        json.dump({'data': model.m_to_dict(with_root_def=True)}, f, indent=4)
    upload.process_upload(
        file_operations=[
            dict(
                op='ADD',
                path=archive_name,
                target_dir=data.working_directory,
                temporary=True,
            )
        ],
        only_updated_files=True,
    )
    reference = get_reference(
        data.upload_id,
        m_hash(data.upload_id, os.path.join(data.working_directory, archive_name)),
    )

    # Add a reference to the model entry in the training entry
    context = ServerContext(get_upload(data.upload_id, data.user_id))
    upload_files = get_upload_files(data.upload_id, data.user_id)
    upload_raw_path = os.path.join(upload_files.os_path, 'raw')
    archive_name = os.path.join(upload_raw_path, data.mainfile)
    with context.update_entry(data.mainfile, process=True, write=True) as archive:
        parsed_archive = parse(archive_name)[0]
        parsed_archive.data.outputs.append(AutoXRDModelReference(reference=reference))
        parsed_archive.data.trigger_run_action = False
        parsed_archive.data.action_status = get_action_status(
            parsed_archive.data.action_id
        ).name
        archive['data'] = parsed_archive.data.m_to_dict(with_root_def=True)

    # TODO: use the following code once the bug with parse_level=None is fixed
    # context = ServerContext(get_upload(data.upload_id, data.user_id))
    # archive_name = os.path.join(data.working_directory, 'auto_xrd_model.archive.json')
    # with context.update_entry(archive_name, write=True, process=True) as archive:
    #     archive['data'] = model.m_to_dict(with_root_def=True)
