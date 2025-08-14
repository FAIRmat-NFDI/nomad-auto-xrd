from nomad.orchestrator.base import BaseWorkflowHandler
from nomad.orchestrator.shared.constant import TaskQueue
from pydantic import BaseModel


class AutoXRDTrainingEntryPoint(BaseModel):
    entry_point_type: str = 'workflow'

    def load(self):
        from nomad_auto_xrd.actions.training.activities import train_model
        from nomad_auto_xrd.actions.training.workflow import TrainingWorkflow

        return BaseWorkflowHandler(
            task_queue=TaskQueue.CPU,
            workflows=[TrainingWorkflow],
            activities=[train_model],
        )


training_entry_point = AutoXRDTrainingEntryPoint()
