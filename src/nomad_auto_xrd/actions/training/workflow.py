from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from nomad_auto_xrd.actions.training.activities import (
        create_trained_model_entry,
        train_model,
    )
    from nomad_auto_xrd.actions.training.models import (
        CreateTrainedModelEntryInput,
        TrainModelInput,
        UserInput,
    )


@workflow.defn
class TrainingWorkflow:
    @workflow.run
    async def run(self, data: UserInput) -> str:
        workflow_id = workflow.info().workflow_id
        working_directory = f'auto_xrd_model_{workflow_id}'
        includes_pdf = True

        training_output = await workflow.execute_activity(
            train_model,
            TrainModelInput(
                upload_id=data.upload_id,
                user_id=data.user_id,
                mainfile=data.mainfile,
                simulation_settings=data.simulation_settings,
                training_settings=data.training_settings,
                working_directory=working_directory,
                includes_pdf=includes_pdf,
            ),
            start_to_close_timeout=timedelta(hours=2),
            # TODO: uncomment during NOMAD logger integration
            # heartbeat_timeout=timedelta(hours=1),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=10),
                maximum_attempts=3,
                backoff_coefficient=2.0,
            ),
        )
        create_entry_input = CreateTrainedModelEntryInput(
            action_id=workflow_id,
            upload_id=data.upload_id,
            user_id=data.user_id,
            mainfile=data.mainfile,
            simulation_settings=data.simulation_settings,
            training_settings=data.training_settings,
            working_directory=working_directory,
            includes_pdf=includes_pdf,
            xrd_model_path=training_output.xrd_model_path,
            pdf_model_path=training_output.pdf_model_path,
            wandb_run_url_xrd=training_output.wandb_run_url_xrd,
            wandb_run_url_pdf=training_output.wandb_run_url_pdf,
            reference_structure_paths=training_output.reference_structure_paths,
        )
        await workflow.execute_activity(
            create_trained_model_entry,
            create_entry_input,
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=10),
                maximum_attempts=3,
                backoff_coefficient=2.0,
            ),
        )
        return training_output
