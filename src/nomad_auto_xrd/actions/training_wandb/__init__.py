from nomad.actions import TaskQueue
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad.config.models.plugins import ActionEntryPoint


class AutoXRDTrainingWandBEntryPoint(ActionEntryPoint):
    """
    Entry point for the nomad-auto-xrd training actions that uses Weights and Biases
    for logging.
    """

    def load(self):
        from nomad.actions import Action

        from nomad_auto_xrd.actions.training_wandb.activities import (
            prepare_training_workflow_inputs,
        )
        from nomad_auto_xrd.actions.training_wandb.workflow import WandBTrainingWorkflow

        return Action(
            task_queue=self.task_queue,
            workflow=WandBTrainingWorkflow,
            activities=[prepare_training_workflow_inputs],
        )


training_wandb_action = AutoXRDTrainingWandBEntryPoint(
    name='AutoXRD Training with WandB',
    task_queue=TaskQueue.CPU,
    description='Train Auto XRD models along with Weights and Biases logging.',
)
