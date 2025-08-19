from dataclasses import asdict
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from nomad_auto_xrd.actions.analysis.activities import (
        analyze,
        update_analysis_entry,
    )
    from nomad_auto_xrd.actions.analysis.models import (
        AnalyzeInput,
        UpdateAnalysisEntryInput,
        UserInput,
    )


@workflow.defn
class AnalysisWorkflow:
    @workflow.run
    async def run(self, data: UserInput) -> str:
        workflow_id = workflow.info().workflow_id
        working_directory = f'./auto_xrd_inference_{workflow_id}'
        analyze_input = AnalyzeInput(
            **asdict(data),
            working_directory=working_directory,
        )
        result = await workflow.execute_activity(
            analyze,
            analyze_input,
            start_to_close_timeout=timedelta(hours=2),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=10),
                maximum_attempts=3,
                backoff_coefficient=2.0,
            ),
        )
        await workflow.execute_activity(
            update_analysis_entry,
            UpdateAnalysisEntryInput(
                **asdict(data),
                analysis_result=result,
            ),
            start_to_close_timeout=timedelta(seconds=600),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=10),
                backoff_coefficient=1,
            ),
        )
        return result
