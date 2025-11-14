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
            maximum_attempts=3,
            initial_interval=timedelta(seconds=10),
            backoff_coefficient=2.0,
        )
        heartbeat_timeout = timedelta(minutes=5)
        results: list[AnalysisResult] = []
        for idx, xrd_measurement_entry in enumerate(data.xrd_measurement_entries):
            results.append(
                await workflow.execute_activity(
                    analyze,
                    AnalyzeInput(
                        upload_id=data.upload_id,
                        user_id=data.user_id,
                        working_directory=workflow.info().workflow_id,
                        analysis_iter=idx,
                        analysis_settings=data.analysis_settings,
                        xrd_measurement_entry=xrd_measurement_entry,
                    ),
                    heartbeat_timeout=heartbeat_timeout,
                    start_to_close_timeout=timedelta(hours=24),
                    retry_policy=retry_policy,
                )
            )
        simulated_reference_patterns = await workflow.execute_activity(
            simulate_reference_patterns,
            SimulateReferencePatternsInput(
                user_id=data.user_id,
                model_upload_id=data.analysis_settings.auto_xrd_model.upload_id,
                cif_paths=data.analysis_settings.auto_xrd_model.reference_structure_paths,
                wavelength=data.analysis_settings.wavelength,
                min_angle=data.analysis_settings.min_angle,
                max_angle=data.analysis_settings.max_angle,
            ),
            heartbeat_timeout=heartbeat_timeout,
            start_to_close_timeout=timedelta(
                minutes=5
                * len(data.analysis_settings.auto_xrd_model.reference_structure_paths)
            ),
            retry_policy=retry_policy,
        )
        await workflow.execute_activity(
            update_analysis_entry,
            UpdateAnalysisEntryInput(
                upload_id=data.upload_id,
                user_id=data.user_id,
                mainfile=data.mainfile,
                action_instance_id=workflow.info().workflow_id,
                xrd_measurement_entries=data.xrd_measurement_entries,
                analysis_results=results,
                simulated_reference_patterns=simulated_reference_patterns,
            ),
            heartbeat_timeout=heartbeat_timeout,
            start_to_close_timeout=timedelta(hours=2),
            retry_policy=retry_policy,
        )
        return results
