import os

from nomad_auto_xrd.schema import (
    AutoXRDModel,
    SimulationSettings,
    TrainingSettings,
)
from nomad_auto_xrd.training import train


def test_train():
    training_settings = TrainingSettings(num_epochs=2)
    simulation_settings = SimulationSettings()
    model = AutoXRDModel(
        working_directory='.',
        training_settings=training_settings,
        simulation_settings=simulation_settings,
        includes_pdf=True,
    )

    os.chdir('./tests')
    structure_files = os.listdir('data/All_CIFs')
    structure_files = [
        os.path.join('data', 'All_CIFs', path) for path in structure_files
    ]
    print(structure_files)
    model.simulation_settings.structure_files = structure_files

    train(model)
