from nomad.actions import TaskQueue
from pydantic import Field
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from nomad.config.models.plugins import ActionEntryPoint


class AutoXRDAnalysisEntryPoint(ActionEntryPoint):
    """
    Entry point for the nomad-auto-xrd analysis actions
    """

    task_queue: str = Field(
        default=TaskQueue.CPU, description='Determines the task queue for this action'
    )

    def load(self):
        from nomad.actions import Action

        from nomad_auto_xrd.actions.analysis.activities import (
            analyze,
            update_analysis_entry,
        )
        from nomad_auto_xrd.actions.analysis.workflow import AnalysisWorkflow

        return Action(
            task_queue=self.task_queue,
            workflow=AnalysisWorkflow,
            activities=[analyze, update_analysis_entry],
        )


analysis_action = AutoXRDAnalysisEntryPoint()
