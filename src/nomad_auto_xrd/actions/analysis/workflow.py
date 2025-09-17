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
        for xrd_measurement_entry in data.xrd_measurement_entries:
            await workflow.execute_activity(
                analyze,
                AnalyzeInput(
                    upload_id=data.upload_id,
                    user_id=data.user_id,
                    working_directory=working_directory,
                    analysis_settings=data.analysis_settings,
                    xrd_measurement_entry=xrd_measurement_entry,
                ),
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
                action_id=workflow_id,
                analysis_result=result,
            ),
            start_to_close_timeout=timedelta(seconds=600),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=10),
                maximum_attempts=3,
                backoff_coefficient=2.0,
            ),
        )
        return result
