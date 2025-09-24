#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import psutil
import tensorflow as tf
from autoXRD import spectrum_analysis, visualizer
from nomad.datamodel.metainfo.basesections import SectionReference
from nomad_analysis.utils import get_reference
from tqdm import tqdm

from nomad_auto_xrd.common.models import (
    AnalysisInput,
    AnalysisResult,
    AnalysisSettingsInput,
    AutoXRDModelInput,
    XRDMeasurementEntry,
)
from nomad_auto_xrd.common.utils import pattern_preprocessor
from nomad_auto_xrd.schema_packages.schema import (
    AutoXRDAnalysis,
    AutoXRDAnalysisResult,
    IdentifiedPhase,
    MultiPatternAnalysisResult,
    SinglePatternAnalysisResult,
)

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger


def get_total_memory_mb():
    """Get the total memory usage of the current process and its children."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss
    # Add memory of all child processes
    for child in process.children(recursive=True):
        try:
            mem += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return mem / 1024 / 1024  # Convert to MB


def convert_to_serializable(obj):
    """Convert non-serializable objects like numpy arrays to serializable formats."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def extract_min_max_angles(patterns):
    """Extract the minimum and maximum two-theta angles from the given patterns."""
    all_angles = []
    for pattern in patterns:
        all_angles.extend(pattern['two_theta_angles'])
    return min(all_angles), max(all_angles)


def extract_min_max_angles_from_folder(folder_path):
    """Extract the minimum and maximum two-theta angles from spectra files in the given folder."""  # noqa: E501
    all_angles = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.xy'):
            with open(os.path.join(folder_path, filename)) as f:
                for line in f:
                    try:
                        angle = float(line.split()[0])
                        all_angles.append(angle)
                    except ValueError:
                        continue  # Skip lines that do not contain valid data
    return min(all_angles), max(all_angles)


@dataclass
class AnalysisSettings:
    structure_references_directory: str | None = (
        None  # Defaults to the directory containing the CIF files from the archive if not provided  # noqa: E501
    )
    patterns_folder_directory: str = 'temp_patterns'
    xrd_model: str | None = (
        None  # Defaults to the XRD model file from the archive if not provided
    )
    pdf_model: str | None = None
    max_phases: int = 5
    cutoff_intensity: float = 0.05
    min_confidence: float = 10.0
    wavelength: str = 'CuKa'
    unknown_threshold: float = 0.2
    show_reduced: bool = False
    include_pdf: bool = False
    parallel: bool = False
    raw: bool = False
    show_individual: bool = False
    min_angle: float | None = None
    max_angle: float | None = None


def analyze_pattern(  # noqa: PLR0912, PLR0915
    settings: AnalysisSettings, archive: 'EntryArchive', logger: Optional['BoundLogger']
) -> None:
    references_folder = (
        os.path.dirname(archive['data']['cif_files'][0])
        if settings.structure_references_directory is None
        else settings.structure_references_directory
    )
    spectra_folder = settings.patterns_folder_directory

    start = time.time()

    # Check for spectra
    if not os.path.exists(spectra_folder) or len(os.listdir(spectra_folder)) == 0:
        if logger:
            logger.error(
                f'Please provide at least one pattern in the {spectra_folder} directory.'  # noqa: E501
            )
        else:
            print(
                f'ERROR: Please provide at least one pattern in the {spectra_folder} directory.'  # noqa: E501
            )
        return

    results = {'XRD': None, 'PDF': None, 'Merged': None}

    # XRD/PDF ensemble requires all predictions
    final_conf = settings.min_confidence
    if settings.include_pdf:
        settings.min_confidence = 10.0

    model_path = (
        settings.xrd_model if settings.xrd_model else archive['data']['xrd_model_file']
    )

    # Get predictions from XRD analysis
    results['XRD'] = AnalysisResult(
        *spectrum_analysis.main(
            spectra_folder,
            references_folder,
            settings.max_phases,
            settings.cutoff_intensity,
            settings.min_confidence,
            settings.wavelength,
            settings.min_angle,
            settings.max_angle,
            settings.parallel,
            model_path,
            is_pdf=False,
        )
    )

    if settings.include_pdf and settings.pdf_model:
        # Get predictions from PDF analysis
        model_path = settings.pdf_model
        results['PDF'] = AnalysisResult(
            *spectrum_analysis.main(
                spectra_folder,
                references_folder,
                settings.max_phases,
                settings.cutoff_intensity,
                settings.min_confidence,
                settings.wavelength,
                settings.min_angle,
                settings.max_angle,
                settings.parallel,
                model_path,
                is_pdf=True,
            )
        )

        # Merge results
        results['Merged'] = AnalysisResult(
            *spectrum_analysis.merge_results(
                {'XRD': results['XRD'].to_dict(), 'PDF': results['PDF'].to_dict()},
                final_conf,
                settings.max_phases,
            )
        )
    else:
        results['Merged'] = results['XRD']

    # Process results
    for idx, (
        spectrum_fname,
        phase_set,
        confidence,
        backup_set,
        heights,
        final_spectrum,
    ) in enumerate(
        zip(
            results['Merged'].filenames,
            results['Merged'].phases,
            results['Merged'].confidences,
            results['Merged'].backup_phases,
            results['Merged'].scale_factors,
            results['Merged'].reduced_spectra,
        )
    ):
        # Display phase ID info
        if logger:
            logger.info(f'Filename: {spectrum_fname}')
            logger.info(f'Predicted phases: {phase_set}')
            logger.info(f'Confidence: {confidence}')
        else:
            print(f'Filename: {spectrum_fname}')
            print(f'Predicted phases: {phase_set}')
            print(f'Confidence: {confidence}')

        # Check for unknown peaks
        if phase_set and 'None' not in phase_set:
            remaining_I = max(final_spectrum)
            if remaining_I > settings.unknown_threshold:
                if logger:
                    logger.warning(
                        f'some peaks (I ~ {int(remaining_I)}%) were not identified.'
                    )
                else:
                    print(
                        f'WARNING: some peaks (I ~ {int(remaining_I)}%) were not identified.'  # noqa: E501
                    )
        else:
            if logger:
                logger.warning('No phases were identified')
            else:
                print('WARNING: No phases were identified')
            continue

        # Show backup predictions
        if settings.show_individual:
            if logger:
                logger.info(f'XRD predicted phases: {results["XRD"].phases[idx]}')
                logger.info(f'XRD confidence: {results["XRD"].confidences[idx]}')
                if settings.include_pdf:
                    logger.info(f'PDF predicted phases: {results["PDF"].phases[idx]}')
                    logger.info(f'PDF confidence: {results["PDF"].confidences[idx]}')
            else:
                print(f'XRD predicted phases: {results["XRD"].phases[idx]}')
                print(f'XRD confidence: {results["XRD"].confidences[idx]}')
                if settings.include_pdf:
                    print(f'PDF predicted phases: {results["PDF"].phases[idx]}')
                    print(f'PDF confidence: {results["PDF"].confidences[idx]}')

        # Plot the results
        phasenames = [f'{phase}.cif' for phase in phase_set]
        visualizer.main(
            spectra_folder,
            spectrum_fname,
            phasenames,
            heights,
            final_spectrum,
            settings.min_angle,
            settings.max_angle,
            settings.wavelength,
            save=False,
            show_reduced=settings.show_reduced,
            inc_pdf=settings.include_pdf,
            plot_both=False,
            raw=settings.raw,
        )

    end = time.time()
    if logger:
        logger.info(f'Total time: {round(end - start, 1)} sec')
    else:
        print(f'Total time: {round(end - start, 1)} sec')

    # Convert results to a JSON serializable format
    serializable_results = convert_to_serializable(results['Merged'].to_dict())

    # Save the results dictionary as a JSON file
    results_file = 'results.json'
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    if logger:
        logger.info(f'Results saved to {results_file}')
    else:
        print(f'Results saved to {results_file}')


def run_analysis_with_patterns_archive(
    settings: AnalysisSettings,
    model_archive_path: str,
    patterns_archive_path: str,
    results_file: str,
) -> None:
    # Load the NOMAD archive data for the model metadata
    with open(model_archive_path) as file:
        archive_data = json.load(file)

    # Load the NOMAD archive data for the patterns
    with open(patterns_archive_path) as file:
        patterns_data = json.load(file)

    # Extract patterns data from the patterns archive
    patterns = patterns_data['archive']['results']['properties']['structural'][
        'diffraction_pattern'
    ]

    # If the provided spectra folder already exists and has files, ask whether to overwrite or use existing files  # noqa: E501
    if (
        os.path.exists(settings.patterns_folder_directory)
        and len(os.listdir(settings.patterns_folder_directory)) > 0
    ):
        user_choice = (
            input(
                f"The folder '{settings.patterns_folder_directory}' already contains spectra files. Do you want to (o)verwrite or (u)se existing files? (o/u): "  # noqa: E501
            )
            .strip()
            .lower()
        )
        if user_choice == 'o':
            # Overwrite: Clear the folder and write new spectra files
            for filename in os.listdir(settings.patterns_folder_directory):
                if filename.endswith('.xy'):
                    os.remove(
                        os.path.join(settings.patterns_folder_directory, filename)
                    )
            settings.min_angle, settings.max_angle = extract_min_max_angles(patterns)
            for i, pattern in enumerate(patterns):
                with open(
                    f'{settings.patterns_folder_directory}/spectrum_{i}.xy', 'w'
                ) as f:
                    two_theta_angles = pattern['two_theta_angles']
                    intensities = pattern['intensity']
                    for angle, intensity in zip(two_theta_angles, intensities):
                        f.write(f'{angle} {intensity}\n')
        elif user_choice == 'u':
            # Use existing files: Recalculate min and max angles from existing spectra
            settings.min_angle, settings.max_angle = extract_min_max_angles_from_folder(
                settings.patterns_folder_directory
            )
        else:
            print(
                "Invalid choice. Please enter 'o' to overwrite or 'u' to use existing files."  # noqa: E501
            )
            return
    else:
        # Otherwise, create spectra files from the archive and calculate min and max angles  # noqa: E501
        settings.min_angle, settings.max_angle = extract_min_max_angles(patterns)
        os.makedirs(settings.patterns_folder_directory, exist_ok=True)
        for i, pattern in enumerate(patterns):
            with open(
                f'{settings.patterns_folder_directory}/spectrum_{i}.xy', 'w'
            ) as f:
                two_theta_angles = pattern['two_theta_angles']
                intensities = pattern['intensity']
                for angle, intensity in zip(two_theta_angles, intensities):
                    f.write(f'{angle} {intensity}\n')

    # Extract settings from the archive data if needed
    archive = archive_data  # Placeholder for the actual archive loading process

    # Run analysis
    analyze_pattern(
        settings, archive, logger=None
    )  # Replace 'logger=None' with an actual logger instance if available


def run_analysis_existing_spectra(
    settings: AnalysisSettings, model_archive_path: str, results_file: str
) -> None:
    # Load the NOMAD archive data for the model metadata
    with open(model_archive_path) as file:
        archive_data = json.load(file)

    # Extract settings from the archive data if needed
    archive = archive_data  # Placeholder for the actual archive loading process

    # Run analysis with existing spectra files in `settings.patterns_folder_directory`
    settings.min_angle, settings.max_angle = extract_min_max_angles_from_folder(
        settings.patterns_folder_directory
    )

    # Run analysis
    analyze_pattern(
        settings, archive, logger=None
    )  # Replace 'logger=None' with an actual logger instance if available


class XRDAutoAnalyzer:
    """
    A class to handle XRD analysis using the XRD-AutoAnalyzer.
    This class provides methods to prepare data, setup model, run analysis, and
    visualize results.
    """

    def __init__(
        self,
        working_directory: str,
        analysis_settings: AnalysisSettingsInput,
        logger: 'BoundLogger | None' = None,
    ):
        """
        Initializes the XRDAutoAnalyzer.

        Args:
            working_directory (str): The directory where the analysis will be performed.
            data_preprocessor (Callable | None): A function to preprocess the XRD data
                entries. If None, the default preprocessor will be used.
        """
        self.working_directory = working_directory
        if not os.path.exists(self.working_directory):
            raise NameError(
                f'The working directory "{self.working_directory}" does not exist.'
            )
        self.logger = logger
        self.analysis_settings = analysis_settings
        self.xrd_model_path, self.pdf_model_path, self.reference_structure_m_proxies = (
            self._model_setup(analysis_settings.auto_xrd_model)
        )

    def _generate_xy_file(self, analysis_input: AnalysisInput) -> None:
        """
        Generates .xy file from the processed data and saves them in the working
        directory under 'Spectra'.

        Args:
            analysis_input (AnalysisInput): Processed data containing filename,
                two_theta, and intensity values.
        """
        spectra_dir = os.path.join(self.working_directory, 'Spectra')
        os.makedirs(spectra_dir, exist_ok=True)
        with open(
            os.path.join(
                spectra_dir, f'{analysis_input.filename.rsplit(".", 1)[0]}.xy'
            ),
            'w',
            encoding='utf-8',
        ) as f:
            for angle, intensity in zip(
                analysis_input.two_theta, analysis_input.intensity
            ):
                f.write(f'{angle} {intensity}\n')

    def _remove_xy_file(self, filename: str) -> None:
        """
        Removes the .xy file corresponding to the given filename from the 'Spectra'
        directory in the working directory.

        Args:
            filename (str): The name of the XRD rawfile whose .xy file should be
                removed.
        """
        spectra_dir = os.path.join(self.working_directory, 'Spectra')
        file_path = os.path.join(spectra_dir, f'{filename.rsplit(".", 1)[0]}.xy')

        os.remove(file_path)

    def _model_setup(
        self,
        model: AutoXRDModelInput,
    ) -> tuple[str, None | str, dict[str, str]]:
        """
        Sets up the model for the analysis by creating symlinks to the reference CIF
        files and model files.

        Args:
            model (AutoXRDModelInput): The AutoXRDModelInput containing model paths.

        Returns:
            tuple[None | str, None | str, dict[str, str]]: A tuple containing the paths
                to the XRD model file, PDF model file, and a dictionary of m_proxies for
                the reference structures.
        """
        xrd_model_path = None
        pdf_model_path = None

        # Create a dictionary to store the m_proxies of the sections with
        # reference structures
        reference_structure_m_proxies = dict()
        reference_structures_dir = os.path.join(self.working_directory, 'References')
        os.makedirs(reference_structures_dir, exist_ok=True)
        for i, cif_path in enumerate(model.reference_structure_paths):
            if os.path.exists(cif_path):
                os.symlink(
                    os.path.abspath(cif_path),
                    os.path.join(reference_structures_dir, os.path.basename(cif_path)),
                )
                reference_structure_m_proxies[
                    os.path.basename(cif_path).split('.cif')[0]
                ] = get_reference(
                    model.upload_id,
                    model.entry_id,
                    f'data/reference_structures/{i}',
                )
            else:
                (self.logger.warning if self.logger else print)(
                    f'Reference file "{cif_path}" does not exist.'
                )

        models_dir = os.path.join(self.working_directory, 'Models')
        os.makedirs(models_dir, exist_ok=True)
        if model.xrd_model_path and os.path.exists(model.xrd_model_path):
            xrd_model_path = os.path.join(
                models_dir, os.path.basename(model.xrd_model_path)
            )
            os.symlink(os.path.abspath(model.xrd_model_path), xrd_model_path)
        else:
            raise FileNotFoundError(
                f'XRD model file "{model.xrd_model_path}" does not exist.'
            )

        if model.pdf_model_path and os.path.exists(model.pdf_model_path):
            pdf_model_path = os.path.join(
                models_dir, os.path.basename(model.pdf_model_path)
            )
            os.symlink(os.path.abspath(model.pdf_model_path), pdf_model_path)

        return xrd_model_path, pdf_model_path, reference_structure_m_proxies

    def eval(  # noqa: PLR0912
        self, analysis_inputs: list[AnalysisInput]
    ) -> AnalysisResult:
        """
        Runs the XRD analysis for the given inputs.
        This function orchestrates the analysis process, including loading the model,
        extracting patterns, and running the analysis to identify the phases.
        If multiple patterns are provided, it will run the analysis one by one for each
        pattern and append the results.

        Args:
            analysis_inputs (list[AnalysisInput]): List of AnalysisInput objects
                containing the data to be analyzed.

        Returns:
            AnalysisResult: An AnalysisResult object containing the results of the
                analysis, including identified phases, confidences, and plot paths.
        """

        all_results = AnalysisResult(
            filenames=[],
            phases=[],
            confidences=[],
            backup_phases=[],
            scale_factors=[],
            reduced_spectra=[],
            phases_m_proxies=[],
            xrd_results_m_proxies=[],
            plot_paths=[],
        )
        original_dir = os.getcwd()
        os.chdir(self.working_directory)
        pbar = tqdm(analysis_inputs, desc='Running analysis')
        for analysis_input in pbar:
            try:
                self._generate_xy_file(analysis_input)
                os.makedirs('temp', exist_ok=True)  # required for `spectrum_analysis`
                xrd_result = AnalysisResult(
                    *spectrum_analysis.main(
                        spectra_directory='Spectra',
                        reference_directory='References',
                        max_phases=self.analysis_settings.max_phases,
                        cutoff_intensity=self.analysis_settings.cutoff_intensity,
                        min_conf=self.analysis_settings.min_confidence,
                        wavelength=self.analysis_settings.wavelength,
                        min_angle=self.analysis_settings.min_angle,
                        max_angle=self.analysis_settings.max_angle,
                        parallel=self.analysis_settings.parallel,
                        model_path=os.path.join(
                            'Models', os.path.basename(self.xrd_model_path)
                        ),
                    )
                )
                if self.analysis_settings.include_pdf and self.pdf_model_path:
                    pdf_result = AnalysisResult(
                        *spectrum_analysis.main(
                            spectra_directory='Spectra',
                            reference_directory='References',
                            max_phases=self.analysis_settings.max_phases,
                            cutoff_intensity=self.analysis_settings.cutoff_intensity,
                            min_conf=self.analysis_settings.min_confidence,
                            wavelength=self.analysis_settings.wavelength,
                            min_angle=self.analysis_settings.min_angle,
                            max_angle=self.analysis_settings.max_angle,
                            parallel=self.analysis_settings.parallel,
                            model_path=os.path.join(
                                'Models', os.path.basename(self.pdf_model_path)
                            ),
                            is_pdf=True,
                        )
                    )
                    merged_results = AnalysisResult.from_dict(
                        spectrum_analysis.merge_results(
                            {
                                'XRD': xrd_result.to_dict(),
                                'PDF': pdf_result.to_dict(),
                            },
                            self.analysis_settings.min_confidence,
                            self.analysis_settings.max_phases,
                        )
                    )
                else:
                    merged_results = xrd_result

                # add m_proxy values of identified phases to the results
                phases_m_proxies = []
                for phase in merged_results.phases[0]:
                    if phase in self.reference_structure_m_proxies:
                        phases_m_proxies.append(
                            self.reference_structure_m_proxies[phase]
                        )
                    else:
                        phases_m_proxies.append(None)
                merged_results.phases_m_proxies = [phases_m_proxies]

                # add measurement_m_proxy to the results

                merged_results.xrd_results_m_proxies = [
                    analysis_input.measurement_m_proxy
                ]

                # plot the identified phases and add plot paths to the results
                visualizer.main(
                    'Spectra',
                    merged_results.filenames[0],
                    [
                        os.path.join(phase + '.cif')
                        for phase in merged_results.phases[0]
                    ],
                    merged_results.scale_factors[0],
                    merged_results.reduced_spectra[0],
                    self.analysis_settings.min_angle,
                    self.analysis_settings.max_angle,
                    self.analysis_settings.wavelength,
                    save=True,
                    show_reduced=self.analysis_settings.show_reduced,
                    inc_pdf=self.analysis_settings.include_pdf,
                    plot_both=False,
                    raw=self.analysis_settings.raw,
                    rietveld=False,
                )
                merged_results.plot_paths = [
                    os.path.join(
                        self.working_directory,
                        analysis_input.filename.rsplit('.', 1)[0] + '.png',
                    )
                ]

                all_results.merge(merged_results)

            except Exception as e:
                message = f'Error during analysis of {analysis_input.filename}: {e}'
                (self.logger.warning if self.logger else print)(message)
                continue
            finally:
                # remove the .xy files after analysis
                # clear tensorflow session to free up memory
                # log memory usage
                self._remove_xy_file(analysis_input.filename)
                tf.keras.backend.clear_session()
                pbar.set_postfix(mem=f'{get_total_memory_mb():.1f} MB')

        os.chdir(original_dir)

        # Convert the results to a serializable format
        for i, spectrum in enumerate(all_results.reduced_spectra):
            all_results.reduced_spectra[i] = (
                spectrum.tolist() if isinstance(spectrum, np.ndarray) else spectrum
            )

        return all_results


def to_nomad_data_results_section(
    xrd_measurement_entry: XRDMeasurementEntry, result: AnalysisResult
) -> AutoXRDAnalysisResult:
    """
    Transforms the analysis results into a list of NOMAD `AutoXRDAnalysisResult`
    sections.

    Args:
        results (AnalysisResult): The results from the analysis.
    """
    result_sections = []
    for (
        xrd_results_m_proxy,
        plot_path,
        phases,
        confidences,
        phases_m_proxies,
    ) in zip(
        result.xrd_results_m_proxies,
        result.plot_paths,
        result.phases,
        result.confidences,
        result.phases_m_proxies,
    ):
        result_section = SinglePatternAnalysisResult(
            xrd_results=SectionReference(reference=xrd_results_m_proxy),
            identified_phases_plot=plot_path,
            identified_phases=[
                IdentifiedPhase(
                    name=phase,
                    confidence=confidence,
                    reference_structure=phase_m_proxy,
                )
                for phase, confidence, phase_m_proxy in zip(
                    phases, confidences, phases_m_proxies
                )
            ],
        )
        result_sections.append(result_section)

    if len(result_sections) > 1:
        entry_data_proxy = get_reference(
            xrd_measurement_entry.upload_id, xrd_measurement_entry.entry_id, 'data'
        )
        return MultiPatternAnalysisResult(
            xrd_measurement=SectionReference(reference=entry_data_proxy),
            single_pattern_results=result_sections,
        )
    else:
        return result_sections[0]


def analyze(
    analysis_entry: 'AutoXRDAnalysis', logger: 'BoundLogger | None' = None
) -> AnalysisResult:
    """
    Runs the XRDAutoAnalyzer in a temporary directory for the given Auto XRD analysis
    entry, moves the plots to the 'Plots' directory, and populates the
    `analysis_entry.results` with the analysis results.

    Args:
        analysis_entry (AutoXRDAnalysis): NOMAD analysis section containing the XRD
            data and model information.

    Returns:
        AnalysisResult: The results of the analysis, including identified phases,
            confidences, and plot paths.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        analysis_inputs = []
        for input_reference_section in analysis_entry.inputs:
            xrd_measurement = input_reference_section.reference.m_parent.m_to_dict()
            analysis_inputs.extend(pattern_preprocessor(xrd_measurement, logger))
        if not analysis_inputs:
            raise ValueError('No valid XRD data found in the analysis entry inputs.')
        model_entry = analysis_entry.analysis_settings.auto_xrd_model
        model_input = AutoXRDModelInput(
            upload_id=model_entry.m_parent.metadata.upload_id,
            entry_id=model_entry.m_parent.metadata.entry_id,
            working_directory=model_entry.working_directory,
            includes_pdf=model_entry.includes_pdf,
            reference_structure_paths=[
                section.cif_file for section in model_entry.reference_structures
            ],
            xrd_model_path=model_entry.xrd_model,
            pdf_model_path=model_entry.pdf_model,
        )
        analysis_settings = AnalysisSettingsInput(
            auto_xrd_model=model_input,
            max_phases=analysis_entry.analysis_settings.max_phases,
            cutoff_intensity=analysis_entry.analysis_settings.cutoff_intensity,
            min_confidence=analysis_entry.analysis_settings.min_confidence,
            unknown_threshold=analysis_entry.analysis_settings.unknown_threshold,
            show_reduced=analysis_entry.analysis_settings.show_reduced,
            include_pdf=analysis_entry.analysis_settings.include_pdf,
            parallel=analysis_entry.analysis_settings.parallel,
            raw=analysis_entry.analysis_settings.raw,
            show_individual=analysis_entry.analysis_settings.show_individual,
            wavelength=analysis_entry.analysis_settings.wavelength.to(
                'angstrom'
            ).magnitude,
            min_angle=analysis_entry.analysis_settings.min_angle.to('degree').magnitude,
            max_angle=analysis_entry.analysis_settings.max_angle.to('degree').magnitude,
        )
        analyzer = XRDAutoAnalyzer(temp_dir, analysis_settings, logger)
        results = analyzer.eval(analysis_inputs)

        # Move the plot out of `temp_dir`
        plots_dir = os.path.join('Plots')
        os.makedirs(plots_dir, exist_ok=True)
        for result_iter, plot_path in enumerate(results.plot_paths):
            new_plot_path = os.path.join(plots_dir, os.path.basename(plot_path))
            if os.path.exists(plot_path):
                shutil.copy2(plot_path, new_plot_path)
                results.plot_paths[result_iter] = new_plot_path

    analysis_entry.results = to_nomad_data_results_section(results)

    return results


# Example usage
if __name__ == '__main__':
    settings = AnalysisSettings(
        structure_references_directory='References',
        patterns_folder_directory='temp_patterns',
        xrd_model='path/to/xrd_model.h5',
        max_phases=5,
        min_confidence=30,
    )

    # Choose either to run with existing spectra files or to create from a patterns archive  # noqa: E501
    run_type = input(
        'Run analysis with (1) existing spectra files or (2) create from patterns archive? (1/2): '  # noqa: E501
    ).strip()

    if run_type == '1':
        run_analysis_existing_spectra(
            settings, 'model_metadata.archive.json', 'custom_results.json'
        )
    elif run_type == '2':
        run_analysis_with_patterns_archive(
            settings,
            'model_metadata.archive.json',
            'patterns_archive.json',
            'custom_results.json',
        )
    else:
        print("Invalid choice. Please select either '1' or '2'.")
