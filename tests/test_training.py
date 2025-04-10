import os

from nomad_auto_xrd.schema import (
    AutoXRDModel,
    SimulationSettings,
    TrainingSettings,
)
from nomad_auto_xrd.training import train


def test_train():
    model = AutoXRDModel(
        name='test_model',
        description='A test model for training.',
        working_directory='training',
        training_settings=TrainingSettings(num_epochs=2),
        simulation_settings=SimulationSettings(),
        includes_pdf=True,
    )
    structure_files = os.listdir('./tests/data/structure_files')
    structure_files = [
        os.path.join('data', 'structure_files', path) for path in structure_files
    ]
    model.simulation_settings.structure_files = structure_files

    os.chdir('tests')
    train(model)
    # Check if the model was trained successfully
    assert os.path.exists('training/')
