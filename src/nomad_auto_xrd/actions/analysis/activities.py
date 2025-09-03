import os
import shutil
import tempfile

from temporalio import activity

from nomad_auto_xrd.actions.analysis.models import (
    AnalyzeInput,
    UpdateAnalysisEntryInput,
)
from nomad_auto_xrd.common.models import AnalysisResult


@activity.defn
async def analyze(data: AnalyzeInput) -> AnalysisResult:
    """
    Activity to run auto xrd analysis on the given data.
    """

    from nomad_auto_xrd.common.analysis import XRDAutoAnalyzer
    from nomad_auto_xrd.common.utils import get_upload

    # Run training within the upload folder
    original_path = os.path.abspath(os.curdir)
    upload = get_upload(data.upload_id, data.user_id)
    upload_raw_path = os.path.join(upload.upload_files.os_path, 'raw')

    try:
        os.chdir(upload_raw_path)
        os.makedirs(data.working_directory, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = XRDAutoAnalyzer(temp_dir, data.analysis_settings)
            result = analyzer.eval(data.analysis_inputs)
            # Move the plots from `temp_dir` to a `Plots` folder within the
            # working directory
            plots_dir = os.path.join(data.working_directory, 'Plots')
            os.makedirs(plots_dir, exist_ok=True)
            for result_iter, plot_path in enumerate(result.plot_paths):
                new_plot_path = os.path.join(plots_dir, os.path.basename(plot_path))
                if os.path.exists(plot_path):
                    shutil.copy2(plot_path, new_plot_path)
                    result.plot_paths[result_iter] = new_plot_path
    finally:
        os.chdir(original_path)

    return result


@activity.defn
async def update_analysis_entry(data: UpdateAnalysisEntryInput) -> None:
    """
    Activity to create update the inference entry in the same upload.
    """
    from nomad.datamodel.context import ServerContext

    from nomad_auto_xrd.common.analysis import to_nomad_data_results_section
    from nomad_auto_xrd.common.utils import get_upload

    result_sections = [
        nomad_data_result.m_to_dict()
        for nomad_data_result in to_nomad_data_results_section(data.analysis_result)
    ]
    upload = get_upload(data.upload_id, data.user_id)
    context = ServerContext(upload)

    with context.update_entry(data.mainfile, process=True, write=True) as archive:
        archive['data']['results'] = result_sections
        archive['data']['trigger_run_action'] = False
        archive['data']['action_id'] = data.action_id
        archive['data']['action_status'] = 'COMPLETED'

    # Context manager method only fetches the data from the input mainfile of the ELN
    # entry. This won't reflect the data that was updated/added by normalization not
    # part of the double-saving (save once to populate the ELN fields, save again to
    # populate the input mainfile based on ELN fields). If this data is needed, we
    # need to use `nomad.search` to fetch the data from the message pack archives.
