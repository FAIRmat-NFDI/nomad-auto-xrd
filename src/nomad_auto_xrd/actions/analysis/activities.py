import os
import shutil
import tempfile

from temporalio import activity

from nomad_auto_xrd.actions.analysis.models import (
    AnalyzeInput,
    SimulatedReferencePattern,
    SimulateReferencePatternsInput,
    UpdateAnalysisEntryInput,
)
from nomad_auto_xrd.common.models import AnalysisResult
from nomad_auto_xrd.common.utils import pattern_preprocessor, read_entry_archive


@activity.defn
async def analyze(data: AnalyzeInput) -> AnalysisResult:
    """
    Activity to run auto xrd analysis on the given data.
    """

    from nomad_auto_xrd.common.analysis import XRDAutoAnalyzer
    from nomad_auto_xrd.common.utils import get_upload

    # Read the entry archive to get the XRD measurement data
    archive = read_entry_archive(
        data.xrd_measurement_entry.entry_id,
        data.xrd_measurement_entry.upload_id,
        data.user_id,
    )
    analysis_inputs = pattern_preprocessor(archive)

    # Run analysis within the upload folder
    original_path = os.path.abspath(os.curdir)
    upload = get_upload(data.upload_id, data.user_id)
    upload_raw_path = os.path.join(upload.upload_files.os_path, 'raw')

    try:
        # if the upload id of the trained model is different from the upload id of the
        # analysis, create symlinks to the required model artifacts directory
        created_symlinks = []
        if data.analysis_settings.auto_xrd_model.upload_id != data.upload_id:
            trained_model_upload = get_upload(
                data.analysis_settings.auto_xrd_model.upload_id, data.user_id
            )
            trained_model_upload_raw_path = os.path.join(
                trained_model_upload.upload_files.os_path,
                'raw',
            )
            src_path = os.path.abspath(
                os.path.join(
                    trained_model_upload_raw_path,
                    data.analysis_settings.auto_xrd_model.working_directory,
                )
            )
            dest_path = os.path.join(
                upload_raw_path,
                data.analysis_settings.auto_xrd_model.working_directory,
            )
            if os.path.exists(dest_path):
                if os.path.islink(dest_path):
                    os.unlink(dest_path)
                else:
                    raise FileExistsError(
                        f'Cannot create symlink, path already exists: {dest_path}'
                    )
            os.symlink(
                src=src_path,
                dst=dest_path,
                target_is_directory=True,
            )
            created_symlinks.append(dest_path)

        os.chdir(upload_raw_path)
        os.makedirs(data.working_directory, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = XRDAutoAnalyzer(temp_dir, data.analysis_settings)
            result = analyzer.eval(analysis_inputs)
            result.reduced_spectra = None  # save space
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
        for link in created_symlinks:
            if os.path.exists(link):
                os.unlink(link)

    return result


@activity.defn
async def simulate_reference_patterns(
    data: SimulateReferencePatternsInput,
) -> list[SimulatedReferencePattern]:
    """
    Activity to simulate reference patterns from the given CIF files.
    """
    from nomad.datamodel.context import ServerContext

    from nomad_auto_xrd.common.utils import get_upload, simulate_pattern

    simulated_patterns = []

    upload = get_upload(data.model_upload_id, data.user_id)
    context = ServerContext(upload)
    for cif_path in data.cif_paths:
        with context.raw_file(cif_path) as file:
            two_theta, intensity = simulate_pattern(
                file.name,
                data.wavelength,
                (data.min_angle, data.max_angle),
            )
            simulated_pattern = SimulatedReferencePattern(
                cif_path=cif_path,
                two_theta=two_theta,
                intensity=intensity,
            )
            simulated_patterns.append(simulated_pattern)

    return simulated_patterns


@activity.defn
async def update_analysis_entry(data: UpdateAnalysisEntryInput) -> None:
    """
    Activity to create update the inference entry in the same upload.
    """
    from nomad.datamodel.context import ServerContext

    from nomad_auto_xrd.common.analysis import to_nomad_data_results_section
    from nomad_auto_xrd.common.utils import get_upload

    result_sections = [
        to_nomad_data_results_section(xrd_measurement_entry, analysis_result).m_to_dict(
            with_root_def=True
        )
        for xrd_measurement_entry, analysis_result in zip(
            data.xrd_measurement_entries, data.analysis_results
        )
    ]
    upload = get_upload(data.upload_id, data.user_id)
    context = ServerContext(upload)

    with context.update_entry(data.mainfile, process=True, write=True) as archive:
        archive['data']['results'] = result_sections
        archive['data']['trigger_run_action'] = False
        archive['data']['action_id'] = data.action_id
        archive['data']['action_status'] = 'COMPLETED'
        archive['data']['analysis_settings']['simulated_reference_patterns'] = [
            {
                'name': os.path.basename(pattern.cif_path).split('.cif')[0],
                'two_theta': pattern.two_theta,
                'intensity': pattern.intensity,
            }
            for pattern in data.simulated_reference_patterns
        ]

    # Context manager method only fetches the data from the input mainfile of the ELN
    # entry. This won't reflect the data that was updated/added by normalization not
    # part of the double-saving (save once to populate the ELN fields, save again to
    # populate the input mainfile based on ELN fields). If this data is needed, we
    # need to use `nomad.search` to fetch the data from the message pack archives.
