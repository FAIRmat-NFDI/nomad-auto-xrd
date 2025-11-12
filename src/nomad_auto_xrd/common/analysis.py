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
import os
import shutil
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf
from autoXRD import spectrum_analysis, visualizer
from nomad_analysis.utils import get_reference
from tqdm import tqdm

from nomad_auto_xrd.common.models import (
    AnalysisInput,
    AnalysisResult,
    AnalysisSettingsInput,
    AutoXRDModelInput,
    XRDMeasurementEntry,
)
from nomad_auto_xrd.common.utils import get_total_memory_mb, pattern_preprocessor
from nomad_auto_xrd.schema_packages.schema import (
    AutoXRDAnalysis,
    AutoXRDAnalysisResult,
    IdentifiedPhase,
    MultiPatternAnalysisResult,
    SinglePatternAnalysisResult,
)

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger


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

    def _generate_xy_file(
        self,
        path: str,
        two_theta: list[float],
        intensity: list[float],
    ) -> None:
        """
        Generates .xy file from the processed data and saves them in the working
        directory under 'Spectra'.

        Args:
            path (str): The path to the .xy file to be generated.
            two_theta (list[float]): The 2-theta values for the XRD pattern.
            intensity (list[float]): The intensity values for the XRD pattern.
        """

        with open(path, 'w', encoding='utf-8') as f:
            for _two_theta, _intensity in zip(two_theta, intensity):
                f.write(f'{_two_theta} {_intensity}\n')

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
        for idx, analysis_input in enumerate(pbar):
            try:
                spectrum_xy_file = f'spectrum_{idx}.xy'
                self._generate_xy_file(
                    spectrum_xy_file,
                    analysis_input.two_theta,
                    analysis_input.intensity,
                )
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
                    spectrum_xy_file,
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
                    show_reduced=False,
                    inc_pdf=self.analysis_settings.include_pdf,
                    plot_both=False,
                    raw=False,
                    rietveld=False,
                )
                merged_results.plot_paths = [
                    os.path.join(
                        self.working_directory,
                        spectrum_xy_file.rsplit('.', 1)[0] + '.png',
                    )
                ]

                all_results.merge(merged_results)

            except Exception as e:
                message = (
                    f'Error during analysis of {idx}-th index of `analysis_inputs`: {e}'
                )
                (self.logger.warning if self.logger else print)(message)
                continue

            finally:
                # remove the .xy files after analysis
                # clear tensorflow session to free up memory
                # log memory usage
                if os.path.exists(spectrum_xy_file):
                    os.remove(spectrum_xy_file)
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
            xrd_results=xrd_results_m_proxy,
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
            xrd_measurement=entry_data_proxy,
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
            include_pdf=analysis_entry.analysis_settings.include_pdf,
            parallel=analysis_entry.analysis_settings.parallel,
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
