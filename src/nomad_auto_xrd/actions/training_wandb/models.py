from pydantic import BaseModel, Field


class UserInput(BaseModel):
    """Class to represent user input for model training."""

    upload_id: str = Field(
        ...,
        description='ID of the NOMAD upload for reading and creating entries.',
    )
    user_id: str = Field(..., description='ID of the user making the request.')
    mainfile: str = Field(
        ...,
        description='Path to the "*.archive.json" of a "AutoXRDTrainingAction" entry',
    )
    wandb_api_key: str = Field(..., description='The WandB API key.')
    wandb_entity: str = Field(None, description='The WandB entity name (optional).')
    wandb_project: str = Field(None, description='The WandB project name (optional).')
