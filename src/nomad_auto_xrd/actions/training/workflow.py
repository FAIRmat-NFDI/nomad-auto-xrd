from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from nomad_auto_xrd.actions.training.activities import (
        create_trained_model_entry,
        setup_training_artifacts,
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
        retry_policy = RetryPolicy(
            maximum_attempts=3,
            initial_interval=timedelta(seconds=10),
            backoff_coefficient=2.0,
        )
        heartbeat_timeout = timedelta(minutes=5)
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
            heartbeat_timeout=heartbeat_timeout,
            start_to_close_timeout=timedelta(
                minutes=0.5
                * data.simulation_settings.num_patterns
                * len(data.simulation_settings.structure_files)
            ),
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
            heartbeat_timeout=heartbeat_timeout,
            start_to_close_timeout=timedelta(
                minutes=10 * data.training_settings.num_epochs
            ),
            retry_policy=retry_policy,
        )
        create_entry_input = CreateTrainedModelEntryInput(
            upload_id=data.upload_id,
            user_id=data.user_id,
            mainfile=data.mainfile,
            action_instance_id=workflow.info().workflow_id,
            working_directory=workflow.info().workflow_id,
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
            heartbeat_timeout=heartbeat_timeout,
            start_to_close_timeout=timedelta(hours=2),
            retry_policy=retry_policy,
        )
        return training_output
