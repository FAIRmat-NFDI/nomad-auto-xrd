from nomad.actions import TaskQueue
from pydantic import Field
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad.config.models.plugins import ActionEntryPoint


class AutoXRDTrainingEntryPoint(ActionEntryPoint):
    """
    Entry point for the nomad-auto-xrd training actions
    """

    task_queue: str = Field(
        default=TaskQueue.CPU, description='Determines the task queue for this action'
    )

    def load(self):
        from nomad.actions import Action

        from nomad_auto_xrd.actions.training.activities import (
            create_trained_model_entry,
            setup_training_artifacts,
            train_model,
        )
        from nomad_auto_xrd.actions.training.workflow import TrainingWorkflow

        return Action(
            task_queue=self.task_queue,
            workflow=TrainingWorkflow,
            activities=[
                setup_training_artifacts,
                train_model,
                create_trained_model_entry,
            ],
        )


training_action = AutoXRDTrainingEntryPoint()
