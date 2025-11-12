#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
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
from dataclasses import dataclass


@dataclass
class SimulationSettingsInput:
    """Class to represent simulation settings for model training."""

    structure_files: list[str]
    max_texture: float
    min_domain_size: float
    max_domain_size: float
    max_strain: float
    num_patterns: int
    min_angle: float
    max_angle: float
    max_shift: float
    separate: bool
    impur_amt: float
    skip_filter: bool
    include_elems: bool


@dataclass
class TrainingSettingsInput:
    """Class to represent training settings for model training."""

    # TODO make use of these fields in the training process.
    num_epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    test_fraction: float
    enable_wandb: bool = False  # implicitly disable W&B logging by default
    wandb_project: str | None = None
    wandb_entity: str | None = None


@dataclass
class SetupReferencePathsAndDatasetOutput:
    """Class to represent output of setting up reference paths and dataset."""

    reference_structure_paths: list[str]
    xrd_dataset_path: str
    pdf_dataset_path: str | None = None


@dataclass
class TrainModelOutput:
    """Class to represent output of model training."""

    xrd_model_path: str | None = None
    pdf_model_path: str | None = None
    wandb_run_url_xrd: str | None = None
    wandb_run_url_pdf: str | None = None


@dataclass
class AutoXRDModelInput:
    """Class to represent the AutoXRD model input."""

    upload_id: str
    entry_id: str
    working_directory: str
    reference_structure_paths: list[str]
    includes_pdf: bool
    xrd_model_path: str
    pdf_model_path: str | None = None


@dataclass
class AnalysisSettingsInput:
    """Class to represent analysis settings for model training."""

    auto_xrd_model: AutoXRDModelInput
    max_phases: int
    cutoff_intensity: float
    min_confidence: float
    include_pdf: bool
    parallel: bool
    wavelength: float
    min_angle: float
    max_angle: float


@dataclass
class AnalysisInput:
    """
    A data class to hold the XRD data input for analysis.

    Attributes:
        measurement_m_proxy (str): The m_proxy values inside the `data.results` section
            of the measurement entries used for phase identification.
        two_theta (list[float]): The two theta angles from the XRD pattern.
        intensity (list[float]): The intensity values corresponding to the two theta
            angles.
    """

    measurement_m_proxy: str
    two_theta: list[float]
    intensity: list[float]


@dataclass
class AnalysisResult:
    """
    A data class to hold the results of the analysis. As the analysis can be performed
    on multiple XRD files, this class is designed to store results for each file in
    lists.

    Attributes:
        filenames (list): List of filenames for the raw data files.
        phases (list[list]): Identified phases for each file.
        confidences (list[list]): Confidence levels for each identified phase for each
            file.
        backup_phases (list[list]): Backup phases identified during the analysis for
            each file.
        scale_factors (list[list]): Scale factors applied to the spectra.
        reduced_spectra (list[list] | None): Reduced spectra after analysis, if
            available.
        phases_m_proxies (list[list] | None): M-proxies for the identified phases, if
            available.
        xrd_results_m_proxies (list | None): M-proxies for the `data.results`
            section of XRD entries, if available.
        plot_paths (list | None): Paths to the generated plots, if any.
    """

    filenames: list
    phases: list[list]
    confidences: list[list]
    backup_phases: list[list]
    scale_factors: list[list]
    reduced_spectra: list[list] | None = None
    phases_m_proxies: list[list] | None = None
    xrd_results_m_proxies: list | None = None
    plot_paths: list | None = None

    def to_dict(self):
        return {
            'filenames': self.filenames,
            'phases': self.phases,
            'confs': self.confidences,
            'backup_phases': self.backup_phases,
            'scale_factors': self.scale_factors,
            'reduced_spectra': (
                self.reduced_spectra if self.reduced_spectra is not None else []
            ),
            'phases_m_proxies': (
                self.phases_m_proxies if self.phases_m_proxies is not None else []
            ),
            'xrd_results_m_proxies': (
                self.xrd_results_m_proxies
                if self.xrd_results_m_proxies is not None
                else []
            ),
            'plot_paths': self.plot_paths if self.plot_paths is not None else [],
        }

    @classmethod
    def from_dict(self, data):
        return AnalysisResult(
            filenames=list(data['filenames']),
            phases=list(data['phases']),
            confidences=list(data['confs']),
            backup_phases=list(data['backup_phases']),
            scale_factors=list(data['scale_factors']),
            reduced_spectra=list(data['reduced_spectra'])
            if 'reduced_spectra' in data
            else None,
            phases_m_proxies=list(data['phases_m_proxies'])
            if 'phases_m_proxies' in data
            else None,
            xrd_results_m_proxies=list(data['xrd_results_m_proxies'])
            if 'xrd_results_m_proxies' in data
            else None,
            plot_paths=list(data['plot_paths']) if 'plot_paths' in data else None,
        )

    def merge(self, other):
        """
        Merges another AnalysisResult into this one.
        """
        self.filenames.extend(other.filenames)
        self.phases.extend(other.phases)
        self.confidences.extend(other.confidences)
        self.backup_phases.extend(other.backup_phases)
        self.scale_factors.extend(other.scale_factors)

        if other.reduced_spectra:
            if self.reduced_spectra is None:
                self.reduced_spectra = []
            self.reduced_spectra.extend(other.reduced_spectra)

        if other.phases_m_proxies:
            if self.phases_m_proxies is None:
                self.phases_m_proxies = []
            self.phases_m_proxies.extend(other.phases_m_proxies)

        if other.xrd_results_m_proxies:
            if self.xrd_results_m_proxies is None:
                self.xrd_results_m_proxies = []
            self.xrd_results_m_proxies.extend(other.xrd_results_m_proxies)

        if other.plot_paths:
            if self.plot_paths is None:
                self.plot_paths = []
            self.plot_paths.extend(other.plot_paths)


@dataclass
class Phase:
    """
    A data class to hold the identified phase and its confidence level.
    """

    name: str
    confidence: float
    simulated_two_theta: list[float] | None = None
    simulated_intensity: list[float] | None = None


@dataclass
class PhasesPosition:
    """
    A data class to hold the identified phases and their positions for a sample.
    """

    x_position: float
    y_position: float
    x_unit: str
    y_unit: str
    phases: list[Phase]


@dataclass
class PatternAnalysisResult:
    """
    A data class to hold the results of the analysis for a single XRD pattern.
    """

    two_theta: list[float]
    intensity: list[float]
    phases: list[Phase]


@dataclass
class XRDMeasurementEntry:
    """
    Class to represent an XRD measurement entry.

    Attributes:
        entry_id (str): The entry ID of the XRD measurement.
        upload_id (str): The upload ID of the XRD measurement.
    """

    entry_id: str
    upload_id: str
