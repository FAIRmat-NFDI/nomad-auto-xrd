from temporalio import activity

from nomad_auto_xrd.actions.training.models import (
    SimulationSettingsInput,
    TrainingSettingsInput,
)
from nomad_auto_xrd.actions.training.models import (
    UserInput as TrainingWorkflowUserInput,
)
from nomad_auto_xrd.actions.training_wandb.models import UserInput


@activity.defn
async def prepare_training_workflow_inputs(
    data: UserInput,
) -> TrainingWorkflowUserInput:
    from nomad.datamodel.context import ServerContext
    from nomad.utils import generate_entry_id

    from nomad_auto_xrd.common.utils import get_upload

    # using the context to get the entry archive
    context = ServerContext(get_upload(data.upload_id, data.user_id))
    if not context.raw_path_exists(data.mainfile):
        raise FileNotFoundError(f'Mainfile {data.mainfile} not found in upload.')
    entry_id = generate_entry_id(data.upload_id, data.mainfile)
    entry_archive = context.load_archive(
        entry_id, data.upload_id, context.installation_url
    )

    simulation_settings = SimulationSettingsInput(
        structure_files=entry_archive.data.simulation_settings.structure_files,
        max_texture=float(entry_archive.data.simulation_settings.max_texture),
        min_domain_size=float(
            entry_archive.data.simulation_settings.min_domain_size.magnitude
        ),
        max_domain_size=float(
            entry_archive.data.simulation_settings.max_domain_size.magnitude
        ),
        max_strain=float(entry_archive.data.simulation_settings.max_strain),
        num_patterns=int(entry_archive.data.simulation_settings.num_patterns),
        min_angle=float(entry_archive.data.simulation_settings.min_angle.magnitude),
        max_angle=float(entry_archive.data.simulation_settings.max_angle.magnitude),
        max_shift=float(entry_archive.data.simulation_settings.max_shift.magnitude),
        separate=entry_archive.data.simulation_settings.separate,
        impur_amt=float(entry_archive.data.simulation_settings.impur_amt),
        skip_filter=entry_archive.data.simulation_settings.skip_filter,
        include_elems=entry_archive.data.simulation_settings.include_elems,
    )
    training_settings = TrainingSettingsInput(
        num_epochs=int(entry_archive.data.training_settings.num_epochs),
        batch_size=int(entry_archive.data.training_settings.batch_size),
        learning_rate=float(entry_archive.data.training_settings.learning_rate),
        seed=int(entry_archive.data.training_settings.seed),
        test_fraction=float(entry_archive.data.training_settings.test_fraction),
        enable_wandb=True,  # Enable W&B logging
        wandb_project=data.wandb_project,
        wandb_entity=data.wandb_entity,
    )
    training_workflow_user_input = TrainingWorkflowUserInput(
        upload_id=data.upload_id,
        user_id=data.user_id,
        mainfile=data.mainfile,
        simulation_settings=simulation_settings,
        training_settings=training_settings,
    )
    return training_workflow_user_input
