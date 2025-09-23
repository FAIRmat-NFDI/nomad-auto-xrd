from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from nomad_auto_xrd.actions.analysis.activities import (
        analyze,
        simulate_reference_patterns,
        update_analysis_entry,
    )
    from nomad_auto_xrd.actions.analysis.models import (
        AnalysisResult,
        AnalyzeInput,
        SimulateReferencePatternsInput,
        UpdateAnalysisEntryInput,
        UserInput,
    )


@workflow.defn
class AnalysisWorkflow:
    @workflow.run
    async def run(self, data: UserInput) -> str:
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=10),
            maximum_attempts=3,
            backoff_coefficient=2.0,
        )
        workflow_id = workflow.info().workflow_id
        working_directory = f'auto_xrd_inference_{workflow_id}'
        results = []
        for xrd_measurement_entry in data.xrd_measurement_entries:
            results.append(
                await workflow.execute_activity(
                    analyze,
                    AnalyzeInput(
                        upload_id=data.upload_id,
                        user_id=data.user_id,
                        working_directory=working_directory,
                        analysis_settings=data.analysis_settings,
                        xrd_measurement_entry=xrd_measurement_entry,
                    ),
                    start_to_close_timeout=timedelta(days=1),
                    retry_policy=retry_policy,
                )
            )
        result: AnalysisResult = results[0]
        if len(results) > 1:
            for res in results[1:]:
                result.merge(res)
        simulated_reference_patterns = await workflow.execute_activity(
            simulate_reference_patterns,
            SimulateReferencePatternsInput(
                user_id=data.user_id,
                model_upload_id=data.analysis_settings.auto_xrd_model.upload_id,
                cif_paths=data.analysis_settings.auto_xrd_model.reference_structure_paths,
                wavelength=data.analysis_settings.wavelength,
                min_two_theta=data.analysis_settings.min_angle,
                max_two_theta=data.analysis_settings.max_angle,
            ),
            start_to_close_timeout=timedelta(hours=2),
            retry_policy=retry_policy,
        )
        await workflow.execute_activity(
            update_analysis_entry,
            UpdateAnalysisEntryInput(
                upload_id=data.upload_id,
                user_id=data.user_id,
                mainfile=data.mainfile,
                action_id=workflow_id,
                analysis_result=result,
                simulated_reference_patterns=simulated_reference_patterns,
            ),
            start_to_close_timeout=timedelta(seconds=600),
            retry_policy=retry_policy,
        )
        return result
