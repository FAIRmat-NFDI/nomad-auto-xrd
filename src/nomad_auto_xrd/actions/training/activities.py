import os
from dataclasses import asdict

from temporalio import activity

from nomad_auto_xrd.actions.training.models import (
    CreateTrainedModelEntryInput,
    TrainModelInput,
)
from nomad_auto_xrd.common.models import TrainModelOutput


@activity.defn
async def train_model(data: TrainModelInput) -> TrainModelOutput:
    """
    Activity to train a machine learning model.
    """
    from nomad_auto_xrd.common.training import train
    from nomad_auto_xrd.common.utils import get_upload

    # Run training within the upload folder
    original_path = os.path.abspath(os.curdir)
    upload = get_upload(data.upload_id, data.user_id)
    upload_raw_path = os.path.join(upload.upload_files.os_path, 'raw')
    try:
        os.chdir(upload_raw_path)
        output = train(
            working_directory=data.working_directory,
            simulation_settings=data.simulation_settings,
            training_settings=data.training_settings,
            includes_pdf=data.includes_pdf,
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

    from nomad.datamodel.context import ServerContext
    from nomad.utils import generate_entry_id
    from nomad_measurements.utils import get_reference

    from nomad_auto_xrd.common.utils import get_upload
    from nomad_auto_xrd.schema_packages.schema import (
        AutoXRDModel,
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

    context = ServerContext(get_upload(data.upload_id, data.user_id))
    rel_mainfile_path = os.path.join(
        data.working_directory, 'auto_xrd_model.archive.json'
    )

    # Create an entry for the trained model and generate its reference
    with context.update_entry(rel_mainfile_path, write=True, process=True) as archive:
        archive['data'] = model.m_to_dict(with_root_def=True)
    reference = get_reference(
        data.upload_id,
        generate_entry_id(data.upload_id, rel_mainfile_path),
    )

    # Add a reference to the model entry in the training entry
    with context.update_entry(data.mainfile, process=True, write=True) as archive:
        archive['data']['outputs'] = [{'reference': reference}]
        archive['data']['trigger_run_action'] = False
        archive['data']['action_id'] = data.action_id
        archive['data']['action_status'] = 'COMPLETED'

    ## The following code is an alternative way to add the model entry to the upload
    # archive_name = 'auto_xrd_model.archive.json'
    # with open(archive_name, 'w', encoding='utf-8') as f:
    #     json.dump({'data': model.m_to_dict(with_root_def=True)}, f, indent=4)
    # upload.process_upload(
    #     file_operations=[
    #         dict(
    #             op='ADD',
    #             path=archive_name,
    #             target_dir=data.working_directory,
    #             temporary=True,
    #         )
    #     ],
    #     only_updated_files=True,
    # )
