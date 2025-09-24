from dataclasses import dataclass

from nomad_auto_xrd.common.models import (
    AnalysisResult,
    AnalysisSettingsInput,
    XRDMeasurementEntry,
)


@dataclass
class UserInput:
    """
    Class to represent user input for analysis.

    Attributes:
        upload_id (str): The upload ID of ELN entry triggering the analysis.
        user_id (str): The user ID of the user triggering the analysis.
        mainfile (str): The main file of the ELN entry.
        analysis_settings (AnalysisSettingsInput): Settings for the analysis.
        xrd_measurement_entries: list[XRDMeasurementEntry]: List of XRD measurement
            entries to analyze.
    """

    upload_id: str
    user_id: str
    mainfile: str
    analysis_settings: AnalysisSettingsInput
    xrd_measurement_entries: list[XRDMeasurementEntry]


@dataclass
class AnalyzeInput:
    """Class to represent input for analyze activity."""

    upload_id: str
    user_id: str
    working_directory: str
    xrd_measurement_entry: XRDMeasurementEntry
    analysis_settings: AnalysisSettingsInput


@dataclass
class SimulateReferencePatternsInput:
    """Class to represent input for simulating reference patterns activity."""

    user_id: str
    model_upload_id: str
    cif_paths: list[str]
    wavelength: float
    min_angle: float
    max_angle: float


@dataclass
class SimulatedReferencePattern:
    """
    Class to represent a simulated XRD pattern for a reference phase.

    Attributes:
    - cif_path: Path to the CIF file used for simulation.
    - two_theta: List of two theta angles in degrees.
    - intensity: List of intensity values corresponding to the two theta angles.
    """

    cif_path: str
    two_theta: list[float]
    intensity: list[float]


@dataclass
class UpdateAnalysisEntryInput:
    """Class to represent input for updating an analysis entry."""

    upload_id: str
    user_id: str
    mainfile: str
    action_id: str
    xrd_measurement_entries: list[XRDMeasurementEntry]
    analysis_results: list[AnalysisResult]
    simulated_reference_patterns: list[SimulatedReferencePattern]
