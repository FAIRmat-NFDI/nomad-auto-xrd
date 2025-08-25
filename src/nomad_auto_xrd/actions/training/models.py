from dataclasses import dataclass

from nomad_auto_xrd.common.models import SimulationSettingsInput, TrainingSettingsInput


@dataclass
class UserInput:
    """Class to represent user input for model training."""

    upload_id: str
    user_id: str
    mainfile: str

    simulation_settings: SimulationSettingsInput
    training_settings: TrainingSettingsInput


@dataclass
class TrainModelInput(UserInput):
    """Class to represent training input for model training."""

    working_directory: str
    includes_pdf: bool


@dataclass
class CreateTrainedModelEntryInput(TrainModelInput):
    """Class to represent input for creating a trained model entry."""

    xrd_model_path: str
    pdf_model_path: str | None = None
    wandb_run_url_xrd: str | None = None
    wandb_run_url_pdf: str | None = None
    reference_structure_paths: list[str] = None
