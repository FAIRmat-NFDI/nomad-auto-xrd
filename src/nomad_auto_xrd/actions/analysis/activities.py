import os
import tempfile

from nomad.actions.utils import get_upload, get_upload_files
from temporalio import activity

from nomad_auto_xrd.actions.analysis.models import (
    AnalyzeInput,
    UpdateAnalysisEntryInput,
)
from nomad_auto_xrd.models import AnalysisResult


@activity.defn
async def analyze(data: AnalyzeInput) -> AnalysisResult:
    """
    Activity to run auto xrd analysis on the given data.
    """
    from nomad_auto_xrd.analysis import XRDAutoAnalyser

    # Run training within the upload folder
    original_path = os.path.abspath(os.curdir)
    upload_files = get_upload_files(data.upload_id, data.user_id)
    upload_raw_path = os.path.join(upload_files.os_path, 'raw')

    try:
        os.chdir(upload_raw_path)
        os.makedirs(data.working_directory, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = XRDAutoAnalyser(temp_dir, data.analysis_settings)
            result = analyzer.eval(data.analysis_inputs)
    finally:
        os.chdir(original_path)

    return result


@activity.defn
async def update_analysis_entry(data: UpdateAnalysisEntryInput) -> None:
    """
    Activity to create update the inference entry in the same upload.
    """
    from nomad.client import parse
    from nomad.datamodel.context import ServerContext

    from nomad_auto_xrd.analysis import populate_analysis_entry

    context = ServerContext(get_upload(data.upload_id, data.user_id))
    upload_files = get_upload_files(data.upload_id, data.user_id)
    upload_raw_path = os.path.join(upload_files.os_path, 'raw')
    archive_name = os.path.join(upload_raw_path, data.mainfile)
    with context.update_entry(data.mainfile, process=True, write=True) as archive:
        parsed_archive = parse(archive_name)[0]
        populate_analysis_entry(parsed_archive.data, data.analysis_result)
        parsed_archive.data.trigger_run_analysis = False
        archive['data'] = parsed_archive.data.m_to_dict(with_root_def=True)
