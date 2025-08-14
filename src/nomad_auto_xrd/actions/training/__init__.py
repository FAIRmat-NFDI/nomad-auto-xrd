from nomad.config.models.plugins import WorkflowEntryPoint
from nomad.orchestrator.base import Action
from nomad.orchestrator.shared.constant import TaskQueue


class AutoXRDTrainingEntryPoint(WorkflowEntryPoint):
    """
    Entry point for the nomad-auto-xrd training actions
    """

    def load(self):
        from nomad_auto_xrd.actions.training.activities import (
            create_trained_model_entry,
            train_model,
        )
        from nomad_auto_xrd.actions.training.workflow import TrainingWorkflow

        return Action(
            workflow=TrainingWorkflow,
            activities=[train_model, create_trained_model_entry],
            task_queue=TaskQueue.CPU,
        )


training_entry_point = AutoXRDTrainingEntryPoint()
