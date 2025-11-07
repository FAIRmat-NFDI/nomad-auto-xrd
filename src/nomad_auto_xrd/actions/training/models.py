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
class SetupTrainingArtifactsInput:
    """Class to represent input for setting up training artifacts."""

    upload_id: str
    user_id: str
    working_directory: str
    simulation_settings: SimulationSettingsInput
    test_fraction: float
    includes_pdf: bool


@dataclass
class TrainModelInput:
    """Class to represent training input for model training."""

    upload_id: str
    user_id: str
    training_settings: TrainingSettingsInput
    working_directory: str
    xrd_dataset_path: str
    pdf_dataset_path: str | None = None


@dataclass
class CreateTrainedModelEntryInput:
    """Class to represent input for creating a trained model entry."""

    upload_id: str
    user_id: str
    mainfile: str
    action_instance_id: str

    working_directory: str
    includes_pdf: bool
    simulation_settings: SimulationSettingsInput
    training_settings: TrainingSettingsInput
    reference_structure_paths: list[str]
    xrd_model_path: str
    pdf_model_path: str | None = None
    wandb_run_url_xrd: str | None = None
    wandb_run_url_pdf: str | None = None
