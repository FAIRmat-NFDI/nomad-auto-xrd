from dataclasses import dataclass

from nomad_auto_xrd.common.models import (
    AnalysisInput,
    AnalysisResult,
    AnalysisSettingsInput,
)


@dataclass
class UserInput:
    """Class to represent user input for analysis."""

    upload_id: str
    user_id: str
    mainfile: str

    analysis_settings: AnalysisSettingsInput
    analysis_inputs: list[AnalysisInput]


@dataclass
class AnalyzeInput(UserInput):
    """Class to represent input for analyze activity."""

    working_directory: str


@dataclass
class UpdateAnalysisEntryInput(UserInput):
    """Class to represent input for updating an analysis entry."""

    analysis_result: AnalysisResult
