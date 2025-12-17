from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

from nomad_auto_xrd.actions.training.activities import setup_training_artifacts

with workflow.unsafe.imports_passed_through():
    from nomad_auto_xrd.actions.training.activities import (
        create_trained_model_entry,
        train_model,
    )
    from nomad_auto_xrd.actions.training.models import (
        CreateTrainedModelEntryInput,
        SetupTrainingArtifactsInput,
        TrainModelInput,
        UserInput,
    )


@workflow.defn(name='Training Workflow')
class TrainingWorkflow:
    @workflow.run
    async def run(self, data: UserInput) -> str:
        retry_policy = RetryPolicy(maximum_attempts=1)
        includes_pdf = True
        setup_training_artifacts_output = await workflow.execute_activity(
            setup_training_artifacts,
            SetupTrainingArtifactsInput(
                upload_id=data.upload_id,
                user_id=data.user_id,
                working_directory=workflow.info().workflow_id,
                simulation_settings=data.simulation_settings,
                test_fraction=data.training_settings.test_fraction,
                includes_pdf=includes_pdf,
            ),
            start_to_close_timeout=timedelta(hours=24),
            retry_policy=retry_policy,
        )
        training_output = await workflow.execute_activity(
            train_model,
            TrainModelInput(
                upload_id=data.upload_id,
                user_id=data.user_id,
                training_settings=data.training_settings,
                working_directory=workflow.info().workflow_id,
                xrd_dataset_path=setup_training_artifacts_output.xrd_dataset_path,
                pdf_dataset_path=setup_training_artifacts_output.pdf_dataset_path,
            ),
            start_to_close_timeout=timedelta(hours=24),
            # TODO: uncomment during NOMAD logger integration
            # heartbeat_timeout=timedelta(hours=1),
            retry_policy=retry_policy,
        )
        create_entry_input = CreateTrainedModelEntryInput(
            upload_id=data.upload_id,
            user_id=data.user_id,
            mainfile=data.mainfile,
            action_instance_id=workflow.info().workflow_id,
            working_directory=workflow.info().workflow_id,
            trained_model_name=data.trained_model_name,
            includes_pdf=includes_pdf,
            simulation_settings=data.simulation_settings,
            training_settings=data.training_settings,
            reference_structure_paths=(
                setup_training_artifacts_output.reference_structure_paths
            ),
            xrd_model_path=training_output.xrd_model_path,
            pdf_model_path=training_output.pdf_model_path,
            wandb_run_url_xrd=training_output.wandb_run_url_xrd,
            wandb_run_url_pdf=training_output.wandb_run_url_pdf,
        )
        await workflow.execute_activity(
            create_trained_model_entry,
            create_entry_input,
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=retry_policy,
        )
        return training_output
