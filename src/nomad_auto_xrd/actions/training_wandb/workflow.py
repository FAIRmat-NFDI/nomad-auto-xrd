from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from nomad_auto_xrd.actions.training.workflow import TrainingWorkflow
    from nomad_auto_xrd.actions.training_wandb.activities import (
        prepare_training_workflow_inputs,
    )
    from nomad_auto_xrd.actions.training_wandb.models import UserInput


@workflow.defn(name='Training Workflow with WandB Logging')
class WandBTrainingWorkflow:
    @workflow.run
    async def run(self, data: UserInput) -> str:
        retry_policy = RetryPolicy(maximum_attempts=1)
        training_workflow_input = await workflow.execute_activity(
            prepare_training_workflow_inputs,
            data,
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=retry_policy,
        )
        # with
        output = await workflow.execute_child_workflow(
            TrainingWorkflow.run,
            training_workflow_input,
            id=f'{workflow.info().workflow_id}_training_workflow',
            parent_close_policy=workflow.ParentClosePolicy.TERMINATE,
            retry_policy=retry_policy,
        )
        return output
