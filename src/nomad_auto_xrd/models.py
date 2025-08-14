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

    num_epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    test_fraction: float
    enable_wandb: bool
    wandb_project: str | None = None
    wandb_entity: str | None = None


@dataclass
class TrainModelOutput:
    """Class to represent output of model training."""

    xrd_model_path: str | None = None
    pdf_model_path: str | None = None
    wandb_run_url_xrd: str | None = None
    wandb_run_url_pdf: str | None = None
    reference_structure_paths: list[str] | None = None


@dataclass
class AnalysisInput:
    """
    A data class to hold the XRD data input for analysis.

    Attributes:
        filename (str): The name of the raw data file.
        measurement_m_proxy (str): The m_proxy values inside the `data.results` section
            of the measurement entries used for phase identification.
        two_theta (list[float]): The two theta angles from the XRD pattern.
        intensity (list[float]): The intensity values corresponding to the two theta
            angles.
    """

    filename: str
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
        reduced_spectra (list[list]): Reduced spectra after analysis.
        phases_m_proxies (list[list] | None): M-proxies for the identified phases, if
            available.
        xrd_measurement_m_proxies (list | None): M-proxies for the `data.results`
            section of XRD entries, if available.
        plot_paths (list | None): Paths to the generated plots, if any.
    """

    filenames: list
    phases: list[list]
    confidences: list[list]
    backup_phases: list[list]
    scale_factors: list[list]
    reduced_spectra: list[list]
    phases_m_proxies: list[list] | None = None
    xrd_measurement_m_proxies: list | None = None
    plot_paths: list | None = None

    def to_dict(self):
        return {
            'filenames': self.filenames,
            'phases': self.phases,
            'confs': self.confidences,
            'backup_phases': self.backup_phases,
            'scale_factors': self.scale_factors,
            'reduced_spectra': self.reduced_spectra,
            'phases_m_proxies': (
                self.phases_m_proxies if self.phases_m_proxies is not None else []
            ),
            'xrd_measurement_m_proxies': (
                self.xrd_measurement_m_proxies
                if self.xrd_measurement_m_proxies is not None
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
            reduced_spectra=list(data['reduced_spectra']),
            phases_m_proxies=list(data['phases_m_proxies'])
            if 'phases_m_proxies' in data
            else None,
            xrd_measurement_m_proxies=list(data['xrd_measurement_m_proxies'])
            if 'xrd_measurement_m_proxies' in data
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
        self.reduced_spectra.extend(other.reduced_spectra)

        if other.phases_m_proxies:
            if not self.phases_m_proxies:
                self.phases_m_proxies = []
            self.phases_m_proxies.extend(other.phases_m_proxies)

        if other.xrd_measurement_m_proxies:
            if not self.xrd_measurement_m_proxies:
                self.xrd_measurement_m_proxies = []
            self.xrd_measurement_m_proxies.extend(other.xrd_measurement_m_proxies)

        if other.plot_paths:
            if not self.plot_paths:
                self.plot_paths = []
            self.plot_paths.extend(other.plot_paths)
